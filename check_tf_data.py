import logging
from pathlib import Path

import t5
import tensorflow.compat.v1 as tf

import horovod.torch as hvd
import torch
from torch.utils.data import DataLoader

from data_utils import T5PretrainingDataset

tf.config.set_visible_devices([], 'GPU')  # turn off GPUs for tf operations

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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

    t5dataset = T5PretrainingDataset(shards, batch_size=2)

    # fails to work if num_workes > 0 cause we are using tf.datasets
    dl = DataLoader(t5dataset, num_workers=0, batch_size=None)
    vocab = t5.data.get_default_vocabulary()

    k = 0
    last_el = None
    for x in dl:
        last_el = x
        k += 1

    logger.info(f'n batches at dataloader {hvd.local_rank()}: {k}')
    logger.info(f"last batch at dataloader {hvd.local_rank()}: {vocab.decode(last_el['inputs'][0].tolist())}")
