import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from hmt_src.segment.models import load_checkpoint
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if repo_root.as_posix() not in sys.path:
        sys.path.insert(0, repo_root.as_posix())
    from hmt_src.segment.models import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Segment a txt document using a trained checkpoint.")
    parser.add_argument("--input_txt", type=str, required=True, help="Path to input .txt document")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Trained checkpoint directory")
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "bert", "lstm", "reranker"],
        help="Segmentation model type. Use auto to infer from checkpoint files for bert/lstm.",
    )
    parser.add_argument(
        "--reranker_model_name",
        type=str,
        default="BAAI/bge-reranker-base",
        help="Hugging Face reranker model used when --model_type reranker.",
    )
    parser.add_argument(
        "--reranker_threshold",
        type=float,
        default=0.0,
        help="Boundary decision threshold for reranker score: score < threshold => boundary=1.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Optional tokenizer name fallback if tokenizer files are not in checkpoint.",
    )
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--boundary_threshold",
        type=float,
        default=0.5,
        help="Probability threshold for boundary label=1.",
    )
    parser.add_argument("--output_json", type=str, default=None, help="Optional output json path")
    return parser.parse_args()


def detect_model_type(checkpoint_dir: Optional[Path], model_type: str) -> str:
    if model_type != "auto":
        return model_type
    if checkpoint_dir is None:
        raise ValueError("--model_type auto requires --checkpoint_dir to infer bert/lstm.")
    if (checkpoint_dir / "lstm_config.json").exists():
        return "lstm"
    return "bert"


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fin:
        return fin.read()


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    sentences: List[str] = []
    for para in paragraphs:
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|[\n\r]+", para)
        cleaned = [p.strip() for p in parts if p and p.strip()]
        if cleaned:
            sentences.extend(cleaned)

    if not sentences:
        sentences = [line.strip() for line in text.splitlines() if line.strip()]
    return sentences


def build_pairs(sentences: List[str]) -> Tuple[List[str], List[str]]:
    left = []
    right = []
    for i in range(len(sentences) - 1):
        left.append(sentences[i])
        right.append(sentences[i + 1])
    return left, right


def load_model_and_tokenizer(args, device: torch.device):
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    model_type = detect_model_type(checkpoint_dir, args.model_type)

    if model_type == "reranker":
        tokenizer = AutoTokenizer.from_pretrained(args.reranker_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(args.reranker_model_name).to(device)
        model.eval()
        return model, tokenizer, model_type

    if checkpoint_dir is None:
        raise ValueError("--checkpoint_dir is required for bert/lstm segmentation models.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir.as_posix())
    except Exception:
        if not args.tokenizer_name:
            raise
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if model_type == "bert":
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir.as_posix())
        if model.get_input_embeddings().num_embeddings != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
        model = model.to(device)
    else:
        model = load_checkpoint(
            checkpoint_dir=checkpoint_dir.as_posix(),
            model_type="lstm",
            args=args,
            tokenizer=tokenizer,
            device=device,
        )

    model.eval()
    return model, tokenizer, model_type


@torch.no_grad()
def predict_boundaries(args, model, tokenizer, sentences: List[str], device: torch.device):
    if len(sentences) <= 1:
        return [0], []

    left, right = build_pairs(sentences)
    scores: List[float] = []
    preds: List[int] = []

    for start in range(0, len(left), args.batch_size):
        end = min(start + args.batch_size, len(left))
        batch = tokenizer(
            left[start:end],
            right[start:end],
            truncation=True,
            max_length=args.max_length,
            padding=True,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        if args.model_type == "reranker":
            # BGE reranker returns a relevance score (larger means more related).
            batch_scores = logits.squeeze(-1).detach().cpu().tolist()
            scores.extend(batch_scores)
            preds.extend([1 if s < args.reranker_threshold else 0 for s in batch_scores])
        else:
            prob = torch.softmax(logits, dim=-1)[:, 1]
            batch_scores = prob.detach().cpu().tolist()
            scores.extend(batch_scores)
            preds.extend([1 if p >= args.boundary_threshold else 0 for p in batch_scores])

    boundaries = [0] + preds
    return boundaries, scores


def boundaries_to_segments(sentences: List[str], boundaries: List[int]) -> List[List[str]]:
    if not sentences:
        return []
    segments: List[List[str]] = [[sentences[0]]]
    for idx in range(1, len(sentences)):
        if boundaries[idx] == 1:
            segments.append([sentences[idx]])
        else:
            segments[-1].append(sentences[idx])
    return segments


def print_segments(sentences: List[str], boundaries: List[int], pair_scores: List[float]):
    if not sentences:
        return

    segment_start = 0
    segment_idx = 1
    num_sentences = len(sentences)

    for idx in range(1, num_sentences + 1):
        is_new_segment = idx < num_sentences and boundaries[idx] == 1
        if not is_new_segment and idx != num_sentences:
            continue

        segment_text = " ".join(sentences[segment_start:idx])
        if segment_idx == 1:
            print(f"\n===== Segment {segment_idx} =====")
        else:
            start_sentence_idx = segment_start
            score = pair_scores[start_sentence_idx] if start_sentence_idx < len(pair_scores) else float("nan")
            print(f"\n===== Segment {segment_idx} (boundary score={score:.6f}) =====")
        print(segment_text)

        segment_idx += 1
        segment_start = idx


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text = read_text(args.input_txt)
    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("Input document is empty after sentence splitting.")

    model, tokenizer, model_type = load_model_and_tokenizer(args, device)
    args.model_type = model_type
    boundaries, probs = predict_boundaries(args, model, tokenizer, sentences, device)
    pair_scores = [0.0] + probs
    segments = boundaries_to_segments(sentences, boundaries)

    print(f"Model type: {model_type}")
    print(f"Sentences: {len(sentences)}")
    print(f"Segments: {len(segments)}")
    boundary_indices = [i for i, b in enumerate(boundaries) if b == 1]
    print(f"Boundary sentence indices: {boundary_indices}")
    if boundary_indices:
        print("Boundary scores:")
        for i in boundary_indices:
            print(f"  sentence_index={i}, score={pair_scores[i]:.6f}")
    print_segments(sentences, boundaries, pair_scores)

    if args.output_json:
        output = {
            "input_txt": args.input_txt,
            "checkpoint_dir": args.checkpoint_dir,
            "model_type": model_type,
            "num_sentences": len(sentences),
            "num_segments": len(segments),
            "boundaries": boundaries,
            "pair_scores_for_sentence_i": pair_scores,
            "segments": [" ".join(seg) for seg in segments],
        }
        with open(args.output_json, "w", encoding="utf-8") as fout:
            json.dump(output, fout, indent=2, ensure_ascii=False)
        print(f"\nSaved segmentation json to: {args.output_json}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
