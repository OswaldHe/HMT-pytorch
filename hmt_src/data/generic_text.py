import logging

import datasets

from hmt_src.data.text_datasets import create_text_dataloaders
from hmt_src.data.utils import apply_train_set_split

logger = logging.getLogger(__name__)


def load_generic_text_dataloaders(
    args, tokenizer, batch_size, block_size, history_size, include_train=True
):
    train_ds = None
    if include_train:
        train_ds = datasets.load_dataset(
            args.task_name,
            args.task_subset,
            split="train",
            streaming=args.streaming,
            trust_remote_code=True,
        )
    valid_ds = datasets.load_dataset(
        args.task_name,
        args.task_subset,
        split="validation",
        streaming=args.streaming,
        trust_remote_code=True,
    )
    test_ds = datasets.load_dataset(
        args.task_name,
        args.task_subset,
        split="test",
        streaming=args.streaming,
        trust_remote_code=True,
    )

    train_ds, valid_ds, test_ds = apply_train_set_split(train_ds, valid_ds, test_ds, args)

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
    logger.info("Prepared generic text dataloaders")
    return train_dataloader, valid_dataloader, test_dataloader
