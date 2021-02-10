import logging
import random
from pathlib import Path

import t5
import tensorflow.compat.v1 as tf
tf.config.set_visible_devices([], 'GPU')

import horovod.torch as hvd 
import torch
from t5.models.hf_model import tokens_to_batches
from torch.utils.data import DataLoader, IterableDataset


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def dataset_fn(shards, split=None, shuffle_files=False):
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


class T5Dataset(IterableDataset):
    def __init__(self, shards, batch_size):
        self.sequence_length = {"inputs": 32, "targets": 32}
        self.shards = shards
        self.batch_size = batch_size
        self.task = t5.data.Task("span_corruption",
                                 splits=[],
                                 dataset_fn=lambda split, shuffle_files: dataset_fn(self.shards, split, shuffle_files),
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
            yield x


if __name__ == '__main__':
    # run with horovod:
    # horovodrun --gloo -np 3 python check_tf_data.py
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    if hvd.local_rank() == 0:
        logger.info(f'hvd size: {hvd.size()}')

    data_path = Path('./data/toy_pretraining_data/').expanduser().absolute()
    shards = list(sorted([sh.name for sh in data_path.glob('*.txt')]))
    # split shards across workers, drop remainders
    shards_per_worker = len(shards) // hvd.size()
    shards = shards[hvd.local_rank() * shards_per_worker:(hvd.local_rank() + 1) * shards_per_worker]
    logger.info(f'worker {hvd.local_rank()} shards: {shards}')
    # absolute path to shards
    shards = [str(data_path / sh) for sh in shards]

    t5dataset = T5Dataset(shards, batch_size=2)

    # fails to work if num_workes > 0 cause we are using tf.datasets
    dl = DataLoader(t5dataset, num_workers=0, batch_size=None)
    vocab = t5.data.get_default_vocabulary()

    print(f'from dataloader {hvd.local_rank()}:')
    k = 0
    last_el = None

    for x in dl:
        last_el = x
        k += 1

    logger.info(f'n batches at dataloader {hvd.local_rank()}: {k}')
    logger.info(f"last batch at dataloader {hvd.local_rank()}: {vocab.decode(last_el['inputs'][0].tolist())}")
