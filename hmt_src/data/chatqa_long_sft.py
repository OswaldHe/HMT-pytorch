import logging

import datasets

from hmt_src.data.long_sft_ds_preprocess import LongSFT
from hmt_src.data.utils import apply_train_set_split

logger = logging.getLogger(__name__)


def load_chatqa_long_sft_dataloaders(args, tokenizer, batch_size):
    train_ds = datasets.load_dataset(
        args.task_name, args.task_subset, split="train", streaming=args.streaming
    )
    valid_ds = datasets.load_dataset(
        args.task_name, args.task_subset, split="test", streaming=args.streaming
    )
    test_ds = valid_ds

    train_ds, valid_ds, test_ds = apply_train_set_split(train_ds, valid_ds, test_ds, args)

    train_dataloader = LongSFT(
        train_ds,
        tokenizer,
        batch_size=batch_size,
        clip=True,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    valid_dataloader = LongSFT(
        valid_ds,
        tokenizer,
        batch_size=batch_size,
        clip=True,
    )
    test_dataloader = LongSFT(test_ds, tokenizer, batch_size=batch_size)

    logger.info("Prepared ChatQA2 Long SFT dataloaders")
    return train_dataloader, valid_dataloader, test_dataloader
