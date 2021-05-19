import argparse
import json
import logging
import os
from pathlib import Path
import platform
import importlib

import t5  # noqa: F401 core_dump without t5 import here 🤦‍♂️
import horovod.torch as hvd

import numpy as np
import tensorflow.compat.v1 as tf
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# if CUDA_VISIBLE_DEVICES is not set make all gpus visible
if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])

hvd.init()
# set 1 gpu visible per process, should be before transformers import
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[hvd.local_rank()]

import transformers  # noqa: E402
from transformers import T5Config, T5Tokenizer  # noqa: E402

from data_utils import T5PretrainingDataset, assert_vocabs, jsonl_preprocessor  # noqa: E402
from utils import get_cls_by_name, get_git_hash_commit  # noqa: E402

tf.config.set_visible_devices([], 'GPU')  # turn off GPUs for tf operations
# limit cpu threads for tf
# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)

# limit # of CPU threads to be used per pytorch worker, otherwise it will use all cpus and throttle gpus
torch.set_num_threads(4)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# apex.amp
amp = None

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./model', help='path where to save model')
parser.add_argument('--data_path', type=str, help='path with the sharded data in jsonl format')
parser.add_argument('--log_interval', type=int, default=10,
                    help='how many batches to wait for logging training status')
parser.add_argument('--save_interval', type=int, default=5000, help='save model every steps')
parser.add_argument('--working_dir', type=str, default='.',
                    help='working dir, should be a dir with t5-experiments repo (default: .)')

# model args
parser.add_argument('--base_model', type=str, default='t5-base',
                    help='base model name (from huggingface) (default: t5-base)')
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: None)')
parser.add_argument('--model_cls', type=str, default='transformers:T5ForConditionalGeneration',
                    help='model class name to use (default: transformers:T5ForConditionalGeneration)')

parser.add_argument('--init_checkpoint', type=str, help='path to init checkpoint to load a model from (default: None).')
parser.add_argument('--input_seq_len', type=int, default=128, help='input sequnce length (default: 128).')
parser.add_argument('--target_seq_len', type=int, default=128, help='target sequnce length (default: 128).')
parser.add_argument('--vocab', type=str, default='./vocabs/sentencepiece.model',
                    help='path to vocabulary file with sentencepiece model (default: ./vocabs/sentencepiece.model)')

# training args
parser.add_argument('--lr', type=float, default=None, help='learning rate (default: None)')
parser.add_argument('--batch_size', type=int, default=10, help='input batch size for training (default: 10)')
parser.add_argument('--iters', type=int, default=100,
                    help='number of training steps (i.e., gradient updates) (default: 100).')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help='number of batches to accumulate gradients for each worker; it multiplies total batch size.')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--fp16', action='store_true', default=False, help='use torch.amp for fp16 training')
parser.add_argument('--apex_opt_lvl', type=str, default='O1', help='apex opt level, O1, O2. (default: O1)')
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (default: 0.0)')
parser.add_argument('--scale_parameter', action='store_true', default=False,
                    help='Adafactor scale_parameter (default: False)')
parser.add_argument('--relative_step', action='store_true', default=False,
                    help='Adafactor relative_step (default: False)')
parser.add_argument('--warmup_init', action='store_true', default=False,
                    help='Adafactor warmup_init (default: False)')


def validate(iter):
    # todo: model.eval()
    if hvd.local_rank() == 0:
        logger.info(f'start validation at iter {iter}')


if __name__ == '__main__':
    # run with horovod:
    # export CUDA_VISIBLE_DEVICES=0,1,2; horovodrun --gloo -np 3 python run_t5_pretraining.py
    args = parser.parse_args()
    # set current working dir
    # todo: maybe take path to run_t5_pretraining.py?
    args.working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(args.working_dir)
    if hvd.local_rank() == 0:
        logger.info(f'hvd size: {hvd.size()}')
        logger.info(f'FP16: {args.fp16}')

    if args.fp16:
        try:
            amp = importlib.import_module('apex.amp')
        except ImportError:
            raise ImportError('Install NVIDIA APEX to use fp16 training! Check README.md for instructions.')

    kwargs = {'pin_memory': True}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    # create model_path and save configuration, init tensorboard logs
    tb_writer = None
    if hvd.local_rank() == 0:
        # todo: if model path exists and there is config file, write new config file
        model_path = Path(args.model_path)
        if not model_path.exists():
            Path(model_path).mkdir(parents=True)
        args_dict = dict(vars(args))
        args_dict['ENV'] = {}
        for env_var in ['CUDA_VISIBLE_DEVICES']:
            args_dict['ENV'][env_var] = os.environ.get(env_var, '')
        args_dict['MACHINE'] = platform.node()
        args_dict['COMMIT'] = get_git_hash_commit()
        json.dump(args_dict, open(model_path/'config.json', 'w'), indent=4)
        tb_writer = SummaryWriter(log_dir=args.model_path)

    # get dataset shards
    data_path = Path(args.data_path).expanduser().absolute()
    shards = list(sorted([sh.name for sh in data_path.glob('*.jsonl')]))
    # split shards across workers, drop remainders
    shards_per_worker = len(shards) // hvd.size()
    shards = shards[hvd.local_rank() * shards_per_worker:(hvd.local_rank() + 1) * shards_per_worker]
    logger.info(f'worker {hvd.local_rank()} shards: {shards}')
    # absolute path to shards
    shards = [str(data_path / sh) for sh in shards]

    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps

    t5dataset = T5PretrainingDataset(shards, batch_size=per_worker_batch_size, text_preprocessor=jsonl_preprocessor,
                                     inputs_len=args.input_seq_len, targets_len=args.target_seq_len,
                                     vocab_path=args.vocab)
    # fails to work if num_workes > 0 cause we are using tf.datasets
    t5dataloader = DataLoader(t5dataset, num_workers=0, batch_size=None, **kwargs)

    # define model
    if not args.model_cfg:
        t5config = T5Config.from_pretrained(args.base_model)
    else:
        t5config = T5Config.from_json_file(args.model_cfg)
        # todo: get tokenizer from config
        logger.warning(f'Model configuration was taken from {args.model_cfg}, but tokenizer from {args.base_model}')
    # define tokenizer
    t5tokenizer = T5Tokenizer.from_pretrained(args.base_model)
    if hvd.local_rank() == 0:
        assert_vocabs(t5tokenizer, args.vocab)

    model_cls = get_cls_by_name(args.model_cls)  # transfomers:T5ForConditionalGeneration or modeling_t5:my_class
    if hvd.local_rank() == 0:
        logger.info(f'Using model class: {model_cls}')
    model = model_cls(config=t5config)

    if hasattr(torch.optim, args.optimizer):
        optimizer_cls = getattr(torch.optim, args.optimizer)
    elif hasattr(transformers.optimization, args.optimizer):
        optimizer_cls = getattr(transformers.optimization, args.optimizer)
    else:
        raise RuntimeError(f'Optimizer {args.optimizer} was not found in torch.optim, transformers.optimization')
    if optimizer_cls == transformers.optimization.Adafactor:
        if args.lr is not None and not args.scale_parameter and not args.relative_step:
            raise RuntimeError('To use a manual (external) learning rate schedule you should set '
                               '`scale_parameter=False` and `relative_step=False`')

        optimizer = optimizer_cls(model.parameters(), lr=args.lr,
                                  scale_parameter=args.scale_parameter,
                                  relative_step=args.relative_step,
                                  warmup_init=args.warmup_init,
                                  weight_decay=args.weight_decay)
    else:
        optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model = model.cuda()

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression,
                                         op=hvd.Average,
                                         gradient_predivide_factor=1.0,
                                         backward_passes_per_step=args.gradient_accumulation_steps,
                                         )
    # Apex
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, enabled=args.fp16, opt_level=args.apex_opt_lvl)

    init_iteration = 0
    if args.init_checkpoint:
        # todo: use iteration number to restore position in dataset?
        # todo: if there is checkpoint in model_path load model from the latest checkpoint (init_checkpoint is None)
        checkpoint = torch.load(args.init_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if 'amp' in checkpoint and args.fp16:
            amp.load_state_dict(checkpoint['amp'])
        init_iteration = checkpoint.get('iteration', 0)
        logger.info(f'Model was loaded from: {args.init_checkpoint}')

    # train loop
    i = init_iteration
    pbar = None
    if hvd.local_rank() == 0:
        pbar = tqdm(total=args.iters)
        pbar.update(i)

    losses = []
    while i <= args.iters:
        for batch in t5dataloader:
            model.train()
            if i > args.iters:
                break
            # move batch data to gpu
            for k in batch:
                batch[k] = batch[k].cuda()
            # train step
            optimizer.zero_grad()
            batch_loss = 0
            # iterations over sub-batches (for gradient accumulation)
            for j in range(0, len(batch['inputs']), args.batch_size):
                outputs = model(input_ids=batch['inputs'][j: j + args.batch_size],
                                attention_mask=batch['inputs_mask'][j: j + args.batch_size],
                                # todo: use decoder_attention mask!
                                labels=batch['targets'][j: j + args.batch_size])
                if args.fp16 and args.apex_opt_lvl == 'O2':
                    loss = outputs['loss']
                else:
                    loss = outputs.loss

                # divide loss on gradient_accumulation_steps to get average loss for sub-batches
                loss = loss / args.gradient_accumulation_steps
                batch_loss += loss.detach().item()
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                        # last sub-batch, call synchronize within amp.scale_loss scope
                        # mb move to just above with optimizer.skip_synchronize()
                        if j == (len(batch['inputs']) // args.batch_size - 1) * args.batch_size:
                            optimizer.synchronize()
                else:
                    loss.backward()

            losses += [batch_loss]

            if args.fp16:
                with optimizer.skip_synchronize():
                    optimizer.step()
            else:
                optimizer.step()

            # logging / validation
            if i % args.log_interval == 0:
                mean_loss = np.mean(losses)
                if hvd.local_rank() == 0:
                    # log train loss
                    logger.info(f'step: {i}/{args.iters} loss: {mean_loss:.4f}')
                    tb_writer.add_scalar('loss/train', mean_loss, i)
                    # log learning rate
                    for j, param_group in enumerate(optimizer.param_groups):
                        tb_writer.add_scalar(f'lr/param_group_{j}', param_group['lr'], i)
                losses = []
                validate(i)

            # saving model
            if i % args.save_interval == 0 and hvd.local_rank() == 0:
                save_path = f'{args.model_path}/model_{i}.pth'
                to_save = {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "iteration": i,
                           }
                if args.fp16:
                    to_save['amp'] = amp.state_dict(),
                torch.save(to_save, save_path)
                logger.info(f'Model was saved to {save_path}')

            i += 1
            if hvd.local_rank() == 0:
                pbar.update(1)

    if hvd.local_rank() == 0:
        pbar.close()
    logger.info('Done!')
