import logging

import datasets

from hmt_src.data.text_datasets import create_text_dataloaders
from hmt_src.data.utils import apply_train_set_split

logger = logging.getLogger(__name__)


def load_fineweb_dataloaders(
    args, tokenizer, batch_size, block_size, history_size, include_train=True
):
    task_subset = args.task_subset
    if args.streaming:
        base_ds = datasets.load_dataset(
            args.task_name, task_subset, split="train", streaming=True
        )
        train_ds = base_ds if include_train else None
        valid_ds = test_ds = base_ds
        base_for_split = base_ds if include_train else None
    else:
        train_ds = None
        if include_train:
            train_ds = datasets.load_dataset(
                args.task_name, task_subset, split="train[:5%]"
            )
        valid_ds = datasets.load_dataset(
            args.task_name, task_subset, split="train[5%:7%]"
        )
        test_ds = datasets.load_dataset(
            args.task_name, task_subset, split="train[7%:8%]"
        )
        base_for_split = None

    train_ds, valid_ds, test_ds = apply_train_set_split(
        train_ds, valid_ds, test_ds, args, base_dataset=base_for_split
    )

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
    logger.info("Prepared FineWeb dataloaders")
    return train_dataloader, valid_dataloader, test_dataloader
