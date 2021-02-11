import argparse
import logging
from pathlib import Path

import t5  # noqa: F401 core_dump without t5 import here 🤦‍♂️

import numpy as np
import horovod.torch as hvd
import tensorflow.compat.v1 as tf
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from data_utils import T5PretrainingDataset, assert_vocabs, jsonl_preprocessor

tf.config.set_visible_devices([], 'GPU')  # turn off GPUs for tf operations

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=10, help='input batch size for training (default: 10)')
parser.add_argument('--model_path', type=str, default='./model', help='path where to save model')
parser.add_argument('--iters', type=int, default=100,
                    help='number of iterations to train (default: 100).')
parser.add_argument('--lr', type=float, default=5e-05, help='learning rate (default: 3e-04)')
parser.add_argument('--log_interval', type=int, default=10,
                    help='how many batches to wait for logging training status')
parser.add_argument('--save_interval', type=int, default=5000, help='save model every steps')


def validate(iter):
    if hvd.local_rank() == 0:
        logger.info(f'start validation at iter {iter}')


if __name__ == '__main__':
    # run with horovod:
    # export CUDA_VISIBLE_DEVICES=0,1,2; horovodrun --gloo -np 3 python run_t5_pretraining.py
    args = parser.parse_args()
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    if hvd.local_rank() == 0:
        logger.info(f'hvd size: {hvd.size()}')

    kwargs = {'pin_memory': False}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    if hvd.local_rank() == 0:
        if not Path(args.model_path).exists():
            Path(args.model_path).mkdir(parents=True)

    # specify datasets
    data_path = Path('/home/kuratov/data/ThePile/Wikipedia/preprocessed_shards').expanduser().absolute()
    shards = list(sorted([sh.name for sh in data_path.glob('*.jsonl')]))
    # split shards across workers, drop remainders
    shards_per_worker = len(shards) // hvd.size()
    shards = shards[hvd.local_rank() * shards_per_worker:(hvd.local_rank() + 1) * shards_per_worker]
    logger.info(f'worker {hvd.local_rank()} shards: {shards}')
    # absolute path to shards
    shards = [str(data_path / sh) for sh in shards]

    t5dataset = T5PretrainingDataset(shards, batch_size=args.batch_size, text_preprocessor=jsonl_preprocessor,
                                     inputs_len=128, targets_len=128)
    # fails to work if num_workes > 0 cause we are using tf.datasets
    t5dataloader = DataLoader(t5dataset, num_workers=0, batch_size=None, **kwargs)

    # define model
    model_name = 't5-base'
    config = T5Config.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    if hvd.local_rank() == 0:
        assert_vocabs(tokenizer)

    model = T5ForConditionalGeneration(config=T5Config.from_pretrained(model_name))
    model = model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=hvd.Compression.none,
                                         op=hvd.Average,
                                         gradient_predivide_factor=1.0)

    # train loop
    pbar = None
    if hvd.local_rank() == 0:
        pbar = tqdm(total=args.iters)

    i = 0
    losses = []
    while i < args.iters:
        for batch in t5dataloader:
            if i >= args.iters:
                break

            optimizer.zero_grad()
            outputs = model(input_ids=batch['inputs'].cuda(),
                            attention_mask=batch['inputs_mask'].cuda(),
                            labels=batch['targets'].cuda())
            outputs.loss.backward()
            losses += [outputs.loss.detach().item()]
            optimizer.step()

            if i % args.log_interval == 0:
                mean_loss = np.mean(losses)
                if hvd.local_rank() == 0:
                    logger.info(f'step: {i}/{args.iters} loss: {mean_loss:.4f}')
                losses = []
                validate(i)

            if i % args.save_interval == 0 and hvd.local_rank() == 0:
                torch.save({
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict()
                           }, f'{args.model_path}/model_{i}.pth')

            i += 1
            if hvd.local_rank() == 0:
                pbar.update(1)

    if hvd.local_rank() == 0:
        pbar.close()
    logger.info('Done!')
