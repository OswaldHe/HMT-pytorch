import logging

import datasets

from hmt_src.data.text_datasets import create_text_dataloaders
from hmt_src.data.utils import apply_train_set_split

logger = logging.getLogger(__name__)


def load_pile_arxiv_dataloaders(args, tokenizer, batch_size, block_size, history_size):
    valid_ds = datasets.load_dataset(
        args.task_name, args.task_subset, split="validation", streaming=args.streaming
    )
    test_ds = datasets.load_dataset(
        args.task_name, args.task_subset, split="test", streaming=args.streaming
    )

    train_ds, valid_ds, test_ds = apply_train_set_split(None, valid_ds, test_ds, args)

    loaders = create_text_dataloaders(
        train_ds,
        valid_ds,
        test_ds,
        tokenizer,
        batch_size,
        args,
        block_size,
        history_size,
        group_test=False,
    )
    logger.info("Prepared Pile ArXiv dataloaders")
    return loaders
