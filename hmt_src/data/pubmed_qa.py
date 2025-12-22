import logging

import datasets

from hmt_src.data.pubmedqa_ds_preprocess import PubMedQA
from hmt_src.data.utils import apply_train_set_split

logger = logging.getLogger(__name__)


def load_pubmedqa_dataloaders(args, tokenizer, batch_size):
    task_subset = args.task_subset
    if args.streaming:
        base_ds = datasets.load_dataset(
            args.task_name, task_subset, split="train", streaming=True
        )
        train_ds = valid_ds = test_ds = base_ds
    else:
        train_ds = datasets.load_dataset(
            args.task_name, task_subset, split="train[:75%]"
        )
        valid_ds = datasets.load_dataset(
            args.task_name, task_subset, split="train[75%:90%]"
        )
        test_ds = datasets.load_dataset(
            args.task_name, task_subset, split="train[90%:]"
        )
        base_ds = None

    train_ds, valid_ds, test_ds = apply_train_set_split(
        train_ds, valid_ds, test_ds, args, base_dataset=base_ds
    )

    train_dataloader = PubMedQA(
        train_ds,
        tokenizer,
        fuse_size=args.fuse_size,
        batch_size=batch_size,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    valid_dataloader = PubMedQA(
        valid_ds, tokenizer, fuse_size=args.fuse_size, batch_size=batch_size
    )
    test_dataloader = PubMedQA(
        test_ds,
        tokenizer,
        fuse_size=args.fuse_size,
        batch_size=batch_size,
    )

    logger.info("Prepared PubMedQA dataloaders")
    return train_dataloader, valid_dataloader, test_dataloader
