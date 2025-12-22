import logging

import datasets

from hmt_src.data.text_datasets import create_text_dataloaders
from hmt_src.data.utils import apply_train_set_split

logger = logging.getLogger(__name__)


def load_eda_corpus_dataloaders(args, tokenizer, batch_size, block_size, history_size):
    full_ds = datasets.load_dataset(
        "json",
        data_files="/home/jovyan/workspace/RAG-EDA/training_dataset/generator_dataset/eda_corpus_pretrain.jsonl",
    )["train"]
    split_ds = full_ds.train_test_split(test_size=0.05, seed=42)
    train_ds = split_ds["train"]
    valid_ds = split_ds["test"]
    test_ds = valid_ds

    train_ds, valid_ds, test_ds = apply_train_set_split(train_ds, valid_ds, test_ds, args)

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
    logger.info("Prepared EDA corpus dataloaders")
    return loaders
