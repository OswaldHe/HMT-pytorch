import logging

import datasets

from hmt_src.data.text_datasets import create_text_dataloaders
from hmt_src.data.utils import apply_train_set_split

logger = logging.getLogger(__name__)


def load_fineweb_dataloaders(args, tokenizer, batch_size, block_size, history_size):
    task_subset = args.task_subset
    if args.streaming:
        base_ds = datasets.load_dataset(
            args.task_name, task_subset, split="train", streaming=True
        )
        train_ds = valid_ds = test_ds = base_ds
    else:
        train_ds = datasets.load_dataset(
            args.task_name, task_subset, split="train[:5%]"
        )
        valid_ds = datasets.load_dataset(
            args.task_name, task_subset, split="train[5%:7%]"
        )
        test_ds = datasets.load_dataset(
            args.task_name, task_subset, split="train[7%:8%]"
        )
        base_ds = None

    train_ds, valid_ds, test_ds = apply_train_set_split(
        train_ds, valid_ds, test_ds, args, base_dataset=base_ds
    )

    loaders = create_text_dataloaders(
        train_ds,
        valid_ds,
        test_ds,
        tokenizer,
        batch_size,
        args,
        block_size,
        history_size,
    )
    logger.info("Prepared FineWeb dataloaders")
    return loaders
