import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed as accelerate_set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

try:
    from hmt_src.segment.data import load_segmentation_pair_datasets
    from hmt_src.segment.models import create_model, load_checkpoint, save_checkpoint
except ModuleNotFoundError:
    # Support running as: python hmt_src/segment/train.py ...
    repo_root = Path(__file__).resolve().parents[2]
    if repo_root.as_posix() not in sys.path:
        sys.path.insert(0, repo_root.as_posix())
    from hmt_src.segment.data import load_segmentation_pair_datasets
    from hmt_src.segment.models import create_model, load_checkpoint, save_checkpoint


logging.basicConfig(
    format="[%(levelname)s] (%(asctime)s): %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train sentence segmentation on Wiki-727K")

    parser.add_argument("--model_type", type=str, default="bert", choices=["bert", "lstm"])
    parser.add_argument("--model_name", type=str, default="prajjwal1/bert-small")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="TankNee/wiki-727k")
    parser.add_argument("--dataset_config", type=str, default=None)

    parser.add_argument("--drop_titles", action="store_true", default=False)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to pass trust_remote_code when loading dataset from Hugging Face.",
    )

    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=None,
        help="Positive class weight for weighted CE (label=1).",
    )
    parser.add_argument(
        "--auto_pos_weight",
        action="store_true",
        default=False,
        help="Auto-set positive class weight to neg_count/pos_count from train split.",
    )
    parser.add_argument(
        "--max_pos_weight",
        type=float,
        default=20.0,
        help="Upper bound for auto positive class weight.",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument("--max_pairs_per_doc", type=int, default=None)
    parser.add_argument("--max_train_docs", type=int, default=None)
    parser.add_argument("--max_valid_docs", type=int, default=None)
    parser.add_argument("--max_test_docs", type=int, default=None)
    parser.add_argument("--max_train_pairs", type=int, default=None)
    parser.add_argument("--max_valid_pairs", type=int, default=None)
    parser.add_argument("--max_test_pairs", type=int, default=None)

    parser.add_argument("--lstm_embedding_dim", type=int, default=256)
    parser.add_argument("--lstm_hidden_dim", type=int, default=256)
    parser.add_argument("--lstm_num_layers", type=int, default=2)
    parser.add_argument("--lstm_dropout", type=float, default=0.1)

    parser.add_argument("--save_checkpoint", action="store_true", default=False)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/wiki727k_segment")
    parser.add_argument("--test_best_checkpoint", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=50)

    return parser.parse_args()


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def make_model_tensors_contiguous(model: torch.nn.Module):
    for param in model.parameters():
        if not param.data.is_contiguous():
            param.data = param.data.contiguous()
    for buffer in model.buffers():
        if not buffer.data.is_contiguous():
            buffer.data = buffer.data.contiguous()


def compute_classification_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def count_train_labels(train_dataset) -> Tuple[int, int]:
    pos_count = 0
    neg_count = 0
    if hasattr(train_dataset, "iter"):
        for batch in train_dataset.iter(batch_size=10000):
            for label in batch["labels"]:
                if int(label) == 1:
                    pos_count += 1
                else:
                    neg_count += 1
    else:
        for sample in train_dataset:
            label = int(sample["labels"])
            if label == 1:
                pos_count += 1
            else:
                neg_count += 1
    return neg_count, pos_count


def build_ce_criterion(args, train_dataset, device: torch.device) -> Tuple[torch.nn.CrossEntropyLoss, float]:
    effective_pos_weight = 1.0
    if args.pos_weight is not None:
        effective_pos_weight = float(args.pos_weight)
    elif args.auto_pos_weight:
        neg_count, pos_count = count_train_labels(train_dataset)
        if pos_count == 0:
            logger.warning("No positive samples found in train split. Falling back to pos_weight=1.0.")
            effective_pos_weight = 1.0
        else:
            auto_weight = neg_count / pos_count
            effective_pos_weight = min(max(auto_weight, 1.0), args.max_pos_weight)
            logger.info(
                "Auto pos_weight computed from train split: neg=%d pos=%d raw=%.4f clipped=%.4f",
                neg_count,
                pos_count,
                auto_weight,
                effective_pos_weight,
            )

    if effective_pos_weight != 1.0:
        weights = torch.tensor([1.0, effective_pos_weight], dtype=torch.float, device=device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    return criterion, effective_pos_weight


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    accelerator: Accelerator,
    criterion: torch.nn.CrossEntropyLoss,
) -> Dict[str, float]:
    model.eval()
    losses = []
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False, disable=not accelerator.is_local_main_process):
        batch = move_batch_to_device(batch, accelerator.device)
        labels = batch["labels"]
        model_inputs = {k: v for k, v in batch.items() if k != "labels"}
        outputs = model(**model_inputs)
        loss = criterion(outputs.logits.float(), labels)
        logits = outputs.logits

        gathered_loss = accelerator.gather_for_metrics(loss.detach().reshape(1))
        losses.append(gathered_loss.mean().item())
        preds = torch.argmax(logits, dim=-1)
        all_preds.append(accelerator.gather_for_metrics(preds.detach()).cpu())
        all_labels.append(accelerator.gather_for_metrics(labels.detach()).cpu())

    preds_np = torch.cat(all_preds).numpy() if all_preds else np.array([])
    labels_np = torch.cat(all_labels).numpy() if all_labels else np.array([])
    metrics = compute_classification_metrics(preds_np, labels_np) if len(labels_np) > 0 else {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def run_training(
    args,
    accelerator: Accelerator,
    model,
    tokenizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    model_metadata: Dict,
    criterion: torch.nn.CrossEntropyLoss,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    model, optimizer, train_loader, valid_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, test_loader
    )

    total_updates = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.num_epochs
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_updates, 1),
    )
    scheduler = accelerator.prepare(scheduler)

    best_valid_metrics = None
    best_valid_f1 = -1.0
    best_checkpoint_path = Path(args.checkpoint_dir) / "best"
    global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.num_epochs}",
            leave=False,
            disable=not accelerator.is_local_main_process,
        )
        for step, batch in enumerate(progress, start=1):
            batch = move_batch_to_device(batch, accelerator.device)
            with accelerator.accumulate(model):
                labels = batch["labels"]
                model_inputs = {k: v for k, v in batch.items() if k != "labels"}
                outputs = model(**model_inputs)
                loss = criterion(outputs.logits.float(), labels)
                running_loss += loss.detach().item()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.log_every == 0 and accelerator.is_local_main_process:
                    progress.set_postfix(loss=f"{running_loss / step:.4f}")

        train_loss_local = running_loss / max(len(train_loader), 1)
        train_loss = accelerator.gather_for_metrics(
            torch.tensor([train_loss_local], device=accelerator.device)
        ).mean().item()
        valid_metrics = evaluate(model, valid_loader, accelerator, criterion)
        if accelerator.is_main_process:
            logger.info(
                "Epoch %d | train_loss=%.4f | valid_loss=%.4f | valid_acc=%.4f | valid_f1=%.4f",
                epoch,
                train_loss,
                valid_metrics["loss"],
                valid_metrics["accuracy"],
                valid_metrics["f1"],
            )

        if valid_metrics["f1"] > best_valid_f1:
            best_valid_f1 = valid_metrics["f1"]
            best_valid_metrics = valid_metrics
            if args.save_checkpoint and accelerator.is_main_process:
                metadata = {
                    **model_metadata,
                    "epoch": epoch,
                    "best_valid_metrics": best_valid_metrics,
                    "model_type": args.model_type,
                    "model_name": args.model_name,
                }
                save_checkpoint(
                    model=accelerator.unwrap_model(model),
                    tokenizer=tokenizer,
                    output_dir=best_checkpoint_path.as_posix(),
                    model_type=args.model_type,
                    metadata=metadata,
                )
                logger.info("Saved best checkpoint to %s", best_checkpoint_path.as_posix())
            accelerator.wait_for_everyone()

    if args.save_checkpoint and accelerator.is_main_process:
        final_checkpoint_path = Path(args.checkpoint_dir) / "final"
        metadata = {
            **model_metadata,
            "best_valid_metrics": best_valid_metrics,
            "model_type": args.model_type,
            "model_name": args.model_name,
        }
        save_checkpoint(
            model=accelerator.unwrap_model(model),
            tokenizer=tokenizer,
            output_dir=final_checkpoint_path.as_posix(),
            model_type=args.model_type,
            metadata=metadata,
        )
        logger.info("Saved final checkpoint to %s", final_checkpoint_path.as_posix())
    accelerator.wait_for_everyone()

    model_for_test = model
    if args.save_checkpoint and args.test_best_checkpoint and best_checkpoint_path.exists():
        model_for_test = load_checkpoint(
            checkpoint_dir=best_checkpoint_path.as_posix(),
            model_type=args.model_type,
            args=args,
            tokenizer=tokenizer,
            device=accelerator.device,
        )

    test_metrics = evaluate(model_for_test, test_loader, accelerator, criterion)
    if accelerator.is_main_process:
        logger.info(
            "Test | loss=%.4f | acc=%.4f | precision=%.4f | recall=%.4f | f1=%.4f",
            test_metrics["loss"],
            test_metrics["accuracy"],
            test_metrics["precision"],
            test_metrics["recall"],
            test_metrics["f1"],
        )

    return best_valid_metrics or {}, test_metrics


def main():
    args = parse_args()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    accelerate_set_seed(args.seed)

    if accelerator.is_main_process:
        logger.info("Using device: %s", accelerator.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    added_pad_token = False
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        added_pad_token = True

    train_dataset, valid_dataset, test_dataset = load_segmentation_pair_datasets(args, tokenizer)
    criterion, effective_pos_weight = build_ce_criterion(args, train_dataset, accelerator.device)
    if accelerator.is_main_process:
        logger.info("Using CE positive class weight: %.4f", effective_pos_weight)
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if accelerator.device.type == "cuda" else None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=data_collator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=data_collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=data_collator,
    )

    model, model_metadata = create_model(args, tokenizer)
    if added_pad_token and args.model_type == "bert":
        model.resize_token_embeddings(len(tokenizer))
    make_model_tensors_contiguous(model)

    if args.eval_only:
        if args.load_checkpoint is None:
            raise ValueError("--eval_only requires --load_checkpoint")
        model = load_checkpoint(
            checkpoint_dir=args.load_checkpoint,
            model_type=args.model_type,
            args=args,
            tokenizer=tokenizer,
            device=accelerator.device,
        )
        make_model_tensors_contiguous(model)
        model, valid_loader, test_loader = accelerator.prepare(model, valid_loader, test_loader)
        valid_metrics = evaluate(model, valid_loader, accelerator, criterion)
        test_metrics = evaluate(model, test_loader, accelerator, criterion)
    else:
        valid_metrics, test_metrics = run_training(
            args=args,
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            model_metadata=model_metadata,
            criterion=criterion,
        )

    run_metrics = {
        "validation": valid_metrics,
        "test": test_metrics,
        "effective_pos_weight": effective_pos_weight,
        "args": vars(args),
    }

    if accelerator.is_main_process:
        logger.info("Validation metrics: %s", json.dumps(valid_metrics, indent=2))
        logger.info("Test metrics: %s", json.dumps(test_metrics, indent=2))

    if args.save_checkpoint and accelerator.is_main_process:
        output_dir = Path(args.checkpoint_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "run_metrics.json", "w", encoding="utf-8") as fout:
            json.dump(run_metrics, fout, indent=2)
        logger.info("Saved run metrics to %s", (output_dir / "run_metrics.json").as_posix())


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
