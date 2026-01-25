import logging

import datasets

from hmt_src.data.text_datasets import create_text_dataloaders
from hmt_src.data.utils import apply_train_set_split

logger = logging.getLogger(__name__)


def _normalize_qmsum_text(example):
    dialogue = example.get("dialogue") or example.get("meeting") or ""
    summary = example.get("summary") or example.get("answer") or ""

    if isinstance(dialogue, list):
        dialogue = " ".join(dialogue)
    if isinstance(summary, list):
        summary = " ".join(summary)

    if dialogue and summary:
        text = f"{dialogue}\nSummary:\n{summary}"
    else:
        text = dialogue or summary
    return {"text": text}


def qmsum_loader(args, include_train=True):
    if args.streaming:
        train_ds = None
        if include_train:
            train_ds = datasets.load_dataset(
                "ioeddk/qmsum", split="train", streaming=True
            )
        test_ds = datasets.load_dataset(
            "ioeddk/qmsum", split="test", streaming=True
        )
        valid_ds = test_ds
        base_ds = None
    else:
        dataset = datasets.load_dataset("ioeddk/qmsum")
        train_ds = dataset["train"] if include_train else None
        test_ds = dataset["test"]
        valid_ds = test_ds
        base_ds = None

    reference_ds = train_ds or valid_ds or test_ds
    column_names = reference_ds.column_names
    if train_ds is not None:
        train_ds = train_ds.map(
            _normalize_qmsum_text,
            remove_columns=column_names,
            desc="Prepare QMSum train text",
        )
    valid_ds = valid_ds.map(
        _normalize_qmsum_text,
        remove_columns=column_names,
        desc="Prepare QMSum valid text",
    )
    test_ds = test_ds.map(
        _normalize_qmsum_text,
        remove_columns=column_names,
        desc="Prepare QMSum test text",
    )

    train_ds, valid_ds, test_ds = apply_train_set_split(
        train_ds, valid_ds, test_ds, args, base_dataset=base_ds
    )

    return train_ds, valid_ds, test_ds


def load_longbench_dataloaders(
    args, tokenizer, batch_size, block_size, history_size, include_train=True
):
    if args.task_name == "ioeddk/qmsum":
        train_ds, valid_ds, test_ds = qmsum_loader(args, include_train=include_train)
    else:
        raise ValueError(f"Unknown LongBench task: {args.task_name}")

    train_dataloader, valid_dataloader, test_dataloader = create_text_dataloaders(
        train_ds,
        valid_ds,
        test_ds,
        tokenizer,
        batch_size,
        args,
        block_size,
        history_size,
    )
    if not include_train:
        train_dataloader = None
    logger.info("Prepared LongBench dataloaders")
    return train_dataloader, valid_dataloader, test_dataloader
