import random

import t5
import torch
import tensorflow.compat.v1 as tf
from t5.models.hf_model import tokens_to_batches
from torch.utils.data import IterableDataset


def sharded_dataset_fn(shards, split=None, shuffle_files=False):
    if shuffle_files:
        random.shuffle(shards)
    dataset = tf.data.TextLineDataset(shards)
    return dataset


def text_preprocessor(ds):
    return ds.map(lambda text: {'targets': text})


DEFAULT_OUTPUT_FEATURES = {
    "inputs": t5.seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True,
        required=False),
    "targets": t5.seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True)
}


class T5PretrainingDataset(IterableDataset):
    def __init__(self, shards, batch_size, inputs_len=32, targets_len=32):
        self.sequence_length = {"inputs": inputs_len, "targets": targets_len}
        self.shards = shards
        self.batch_size = batch_size
        self.task = t5.data.Task("span_corruption",
                                 splits=[],
                                 dataset_fn=lambda split, shuffle_files: sharded_dataset_fn(self.shards, split,
                                                                                            shuffle_files),
                                 text_preprocessor=[text_preprocessor],
                                 token_preprocessor=t5.data.preprocessors.span_corruption,
                                 output_features=list(DEFAULT_OUTPUT_FEATURES.keys()),
                                 metric_fns=[])
        self.tfdataset = self.task.get_dataset(split='', sequence_length=self.sequence_length)
        self.tfdataset = tokens_to_batches(self.tfdataset,
                                           sequence_length=self.sequence_length,
                                           batch_size=self.batch_size,
                                           output_features=list(DEFAULT_OUTPUT_FEATURES.keys()))

    def __iter__(self):
        # todo: make dataset infinite?
        for x in self.tfdataset:
            if 'targets_mask' in x:  # do not compute loss on paddings
                x['targets'] -= (1 - x['targets_mask']) * 100
            yield {k: torch.from_numpy(x[k]).type(torch.long) for k in x}


def assert_vocabs(tokenizer):
    """Asserts that default vocabulary from t5 repo is the same as HFTransformers tokenizer

    Args:
        tokenizer: HuggingFace Transformers tokenizer to check
    """
    vocab = t5.data.get_default_vocabulary()
    assert vocab.vocab_size == tokenizer.vocab_size
    assert vocab.unk_id == tokenizer.unk_token_id
    assert vocab.eos_id == tokenizer.eos_token_id
    assert vocab.pad_id == tokenizer.pad_token_id
