import os
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_SEGMENTER_CACHE: Dict[Tuple[str, str], Tuple[AutoModelForSequenceClassification, AutoTokenizer]] = {}


class SegmentIterator:
    def __init__(self, **kwargs):
        self.iter_content = kwargs
        self.pointer = 0
        self.empty = False

    def next(self, segment_length):
        segment = {}
        for k, tensor in self.iter_content.items():
            if tensor is not None:
                if self.pointer >= tensor.shape[1]:
                    self.empty = True
                    return None
                segment[k] = tensor[:, self.pointer : self.pointer + segment_length]

        self.pointer += segment_length
        return segment

    def is_empty(self):
        for _, tensor in self.iter_content.items():
            if tensor is not None:
                if self.pointer >= tensor.shape[1]:
                    self.empty = True
                    return True
                else:
                    return False


def _safe_pad_token_id(tokenizer):
    if tokenizer is None:
        return 0
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    if tokenizer.unk_token_id is not None:
        return tokenizer.unk_token_id
    return 0


def _build_sentence_delimiter_id_patterns(tokenizer) -> List[List[int]]:
    if tokenizer is None:
        return []

    # Match split_sentences separators by converting them to tokenizer IDs directly.
    delimiter_texts = [".", "!", "?", "\n", "\r", "\r\n", "\n\n"]
    patterns: List[List[int]] = []
    seen = set()
    for text in delimiter_texts:
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if not ids:
            continue
        key = tuple(ids)
        if key in seen:
            continue
        seen.add(key)
        patterns.append(ids)

    # Longest-first to avoid partial matching (e.g., "\n\n" before "\n").
    patterns.sort(key=len, reverse=True)
    return patterns


def _load_frozen_segmenter(checkpoint: str, device: torch.device):
    key = (checkpoint, str(device))
    if key in _SEGMENTER_CACHE:
        return _SEGMENTER_CACHE[key]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    _SEGMENTER_CACHE[key] = (model, tokenizer)
    return model, tokenizer


def _is_rank0_process() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            return torch.distributed.get_rank() == 0
        except Exception:
            pass

    rank = os.environ.get("RANK")
    if rank is not None:
        try:
            return int(rank) == 0
        except ValueError:
            pass

    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        try:
            return int(local_rank) == 0
        except ValueError:
            pass

    return True


@torch.no_grad()
def _predict_boundaries(
    model,
    tokenizer,
    sentences: List[str],
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 256,
    threshold: float = 0.5,
):
    if len(sentences) <= 1:
        return [0]

    left = [sentences[i] for i in range(len(sentences) - 1)]
    right = [sentences[i + 1] for i in range(len(sentences) - 1)]
    preds: List[int] = []

    for start in range(0, len(left), batch_size):
        end = min(start + batch_size, len(left))
        batch = tokenizer(
            left[start:end],
            right[start:end],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        if logits.shape[-1] == 1:
            scores = torch.sigmoid(logits.squeeze(-1))
            preds.extend([1 if p >= threshold else 0 for p in scores.detach().cpu().tolist()])
        else:
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds.extend([1 if p >= threshold else 0 for p in probs.detach().cpu().tolist()])

    return [0] + preds


class _BertSegmentIterator:
    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        segment_length: int = 1024,
        lm_tokenizer=None,
        seg_checkpoint: Optional[str] = None,
        boundary_threshold: float = 0.5,
        pred_batch_size: int = 64,
        pred_max_length: int = 256,
        debug: bool = False,
    ):
        self.iter_content = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
        }
        self.pointer = 0
        self.empty = False
        self.segment_length = max(int(segment_length), 1)
        self.lm_tokenizer = lm_tokenizer
        self.pad_token_id = _safe_pad_token_id(lm_tokenizer)
        self.batch_size = input_ids.shape[0]
        self.seq_len = input_ids.shape[1]
        self.device = input_ids.device
        self.boundary_threshold = boundary_threshold
        self.pred_batch_size = pred_batch_size
        self.pred_max_length = pred_max_length
        self.debug = bool(debug)
        self.sentence_delimiter_id_patterns = _build_sentence_delimiter_id_patterns(self.lm_tokenizer)

        if seg_checkpoint is None:
            raise ValueError("seg_checkpoint is required for Bert_SegmentIterator")
        self.seg_model, self.seg_tokenizer = _load_frozen_segmenter(seg_checkpoint, self.device)

        self.sample_spans = self._build_all_sample_spans(input_ids, attention_mask)
        self.num_steps = max((len(spans) for spans in self.sample_spans), default=0)

    def _split_sentences_by_delimiter_ids(self, token_ids: List[int]) -> List[Tuple[int, int]]:
        total_len = len(token_ids)
        if total_len == 0:
            return []
        if not self.sentence_delimiter_id_patterns:
            return [(0, total_len)]

        sent_spans: List[Tuple[int, int]] = []
        sent_start = 0
        idx = 0
        while idx < total_len:
            matched_len = 0
            for pat in self.sentence_delimiter_id_patterns:
                pat_len = len(pat)
                if pat_len == 0 or idx + pat_len > total_len:
                    continue
                if token_ids[idx : idx + pat_len] == pat:
                    matched_len = pat_len
                    break

            if matched_len == 0:
                idx += 1
                continue

            sent_end = idx + matched_len
            if sent_end > sent_start:
                sent_spans.append((sent_start, sent_end))
            sent_start = sent_end
            idx = sent_end

        if sent_start < total_len:
            sent_spans.append((sent_start, total_len))

        if not sent_spans:
            sent_spans = [(0, total_len)]
        return sent_spans

    def _split_group_by_sentences(
        self,
        sent_spans: List[Tuple[int, int]],
        start_sent_idx: int,
        end_sent_idx: int,
    ):
        if start_sent_idx > end_sent_idx:
            return []

        out: List[Tuple[int, int]] = []
        cur_start = None
        cur_end = None
        for idx in range(start_sent_idx, end_sent_idx + 1):
            s_start, s_end = sent_spans[idx]
            if s_end <= s_start:
                continue

            sent_len = s_end - s_start

            # Hard split when a single sentence itself exceeds max segment length.
            if sent_len > self.segment_length:
                if cur_start is not None and cur_end is not None and cur_end > cur_start:
                    out.append((cur_start, cur_end))
                hard_start = s_start
                while hard_start < s_end:
                    hard_end = min(hard_start + self.segment_length, s_end)
                    out.append((hard_start, hard_end))
                    hard_start = hard_end
                cur_start = None
                cur_end = None
                continue

            if cur_start is None:
                cur_start = s_start
                cur_end = s_end
                continue

            # If exceeding max length, move the overflow sentence (including the first
            # exceeded one) into a new segment.
            if (s_end - cur_start) > self.segment_length:
                out.append((cur_start, cur_end))
                cur_start = s_start
                cur_end = s_end
            else:
                cur_end = s_end

        if cur_start is not None and cur_end is not None and cur_end > cur_start:
            out.append((cur_start, cur_end))
        return out

    def _build_spans_for_sample(self, token_ids: List[int]):
        if not token_ids:
            return []
        sent_spans = self._split_sentences_by_delimiter_ids(token_ids)
        if not sent_spans:
            return [(0, len(token_ids))]

        sentences = [
            self.lm_tokenizer.decode(
                token_ids[start:end],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            for start, end in sent_spans
        ]
        if not sentences:
            return [(0, len(token_ids))]

        boundaries = _predict_boundaries(
            self.seg_model,
            self.seg_tokenizer,
            sentences,
            device=self.device,
            batch_size=self.pred_batch_size,
            max_length=self.pred_max_length,
            threshold=self.boundary_threshold,
        )
        if len(boundaries) != len(sent_spans):
            return [(0, len(token_ids))]

        group_starts = [0] + [i for i in range(1, len(sentences)) if boundaries[i] == 1]
        groups = []
        for gidx, gstart in enumerate(group_starts):
            gend = group_starts[gidx + 1] - 1 if gidx + 1 < len(group_starts) else len(sentences) - 1
            groups.append((gstart, gend))

        spans = []
        for gstart, gend in groups:
            spans.extend(self._split_group_by_sentences(sent_spans, gstart, gend))

        if not spans:
            return [(0, len(token_ids))]
        return spans

    def _build_all_sample_spans(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        all_spans = []
        for bidx in range(input_ids.shape[0]):
            valid_len = int(attention_mask[bidx].sum().item()) if attention_mask is not None else input_ids.shape[1]
            valid_len = max(0, min(valid_len, input_ids.shape[1]))
            if valid_len == 0:
                all_spans.append([])
                continue
            token_ids = input_ids[bidx, :valid_len].detach().cpu().tolist()
            spans = self._build_spans_for_sample(token_ids)
            if not spans:
                spans = [(0, valid_len)]
            normalized_spans = []
            for start, end in spans:
                start = max(0, min(int(start), valid_len))
                end = max(start, min(int(end), valid_len))
                if end <= start:
                    continue
                cur_start = start
                while (end - cur_start) > self.segment_length:
                    normalized_spans.append((cur_start, cur_start + self.segment_length))
                    cur_start += self.segment_length
                if end > cur_start:
                    normalized_spans.append((cur_start, end))
            if not normalized_spans:
                normalized_spans = [(0, valid_len)]
            all_spans.append(normalized_spans)
        return all_spans

    def next(self, segment_length):
        del segment_length
        if self.pointer >= self.num_steps:
            self.empty = True
            return None

        step_spans = []
        step_len = 0
        for spans in self.sample_spans:
            if self.pointer < len(spans):
                start, end = spans[self.pointer]
            else:
                start, end = 0, 0
            step_spans.append((start, end))
            step_len = max(step_len, max(0, end - start))

        if step_len == 0:
            self.pointer += 1
            return self.next(segment_length=0)

        segment = {}
        real_token_mask = torch.zeros(
            (self.batch_size, step_len), dtype=torch.bool, device=self.device
        )

        for key, tensor in self.iter_content.items():
            if tensor is None:
                continue

            if tensor.dim() == 2:
                if key == "input_ids":
                    padded = tensor.new_full((self.batch_size, step_len), self.pad_token_id)
                else:
                    padded = tensor.new_zeros((self.batch_size, step_len))
            elif tensor.dim() == 3:
                padded = tensor.new_zeros((self.batch_size, step_len, tensor.shape[-1]))
            else:
                raise ValueError(f"Unsupported tensor dimension {tensor.dim()} for key={key}")

            for bidx, (start, end) in enumerate(step_spans):
                cur_len = max(0, end - start)
                if cur_len == 0:
                    continue
                if tensor.dim() == 2:
                    padded[bidx, :cur_len] = tensor[bidx, start:end]
                else:
                    padded[bidx, :cur_len, :] = tensor[bidx, start:end, :]
                real_token_mask[bidx, :cur_len] = True

            segment[key] = padded

        self.pointer += 1
        if self.debug and _is_rank0_process():
            print("segment!: ", self.pointer)
            if "input_ids" not in segment or segment["input_ids"].shape[0] == 0:
                print("[dynamic_seg] empty segment input_ids\n")
            else:
                sample_ids = segment["input_ids"][0]
                sample_mask = real_token_mask[0] if real_token_mask.shape[0] > 0 else None
                if sample_mask is not None and sample_mask.numel() == sample_ids.shape[0]:
                    filtered_ids = sample_ids[sample_mask].detach().cpu().tolist()
                else:
                    filtered_ids = sample_ids.detach().cpu().tolist()

                if len(filtered_ids) == 0:
                    print("[dynamic_seg] sample0 empty after masking\n")
                elif self.lm_tokenizer is None:
                    print(f"[dynamic_seg] sample0 token_len={len(filtered_ids)} token_ids={filtered_ids}\n")
                else:
                    decoded_text = self.lm_tokenizer.decode(
                        filtered_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    print(f"[dynamic_seg] sample0 token_len={len(filtered_ids)}")
                    print(f"[dynamic_seg] sample0 text: {decoded_text}\n")
        return segment, step_len, real_token_mask

    def is_empty(self):
        if self.pointer >= self.num_steps:
            self.empty = True
            return True
        return False


def Bert_SegmentIterator(**kwargs):
    return _BertSegmentIterator(**kwargs)
