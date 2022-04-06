import argparse
import json
import logging
import os
from pathlib import Path

from megatron.data.dataset_utils import get_indexed_dataset_
from megatron.data.bert_dataset import BertDataset
from megatron.tokenizer.tokenizer import _HFAutoTokenizer

import horovod.torch as hvd
from dotenv import load_dotenv
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from trainer import Trainer

load_dotenv()

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# if CUDA_VISIBLE_DEVICES is not set make all gpus visible
if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])

logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
# first call to torch.cuda.device_count() sets visible gpus, following calls will not change the result
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

hvd.init()

import transformers  # noqa: E402
from transformers import AutoConfig  # noqa: E402

from utils import collect_run_configuration, get_cls_by_name, get_optimizer  # noqa: E402
import optimizers  # noqa: E402

# limit # of CPU threads to be used per pytorch worker, otherwise it might use all cpus and throttle gpus
torch.set_num_threads(4)
# all gpus set with CUDA_VISIBLE_DEVICES are visible to process, indexing from 0 to ...
torch.cuda.set_device(hvd.local_rank())

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./model', help='path where to save model')
parser.add_argument('--data_path', type=str, help='path with the indexed data in bin format')
parser.add_argument('--valid_data_path', type=str, help='path with the indexed data in bin format')
parser.add_argument('--log_interval', type=int, default=10,
                    help='how many batches to wait for logging training status')
parser.add_argument('--valid_interval', type=int, default=None,
                    help='how many batches to wait for logging training status')
parser.add_argument('--save_interval', type=int, default=5000, help='save model every steps')
parser.add_argument('--save_best', action='store_true', default=False,
                    help='Save best checkpoint if validation set is provided.')
parser.add_argument('--working_dir', type=str, default='.',
                    help='working dir, should be a dir with t5-experiments repo (default: .)')
parser.add_argument('--seed', type=int, default=42, help='random seed')

# bert data args
parser.add_argument('--data_impl', type=str, default='mmap', choices=['lazy', 'cached', 'mmap'],
                    help='type of dataset produced by preprocess_data.py')
parser.add_argument('--data_name', type=str, default='', help='used to save/load samples mapping .npy index')
parser.add_argument('--data_n_epochs', type=int, default=None, help='pre-generate samples for data_epochs')
parser.add_argument('--data_n_samples', type=int, default=None, help='pre-generate data_n_samples')
parser.add_argument('--data_skip_warmup', action='store_true', default=False,
                    help='skip dataset warmup (default: False)')
parser.add_argument('--input_seq_len', type=int, default=128, help='input sequnce length (default: 128).')
parser.add_argument('--mlm_prob', type=float, default=0.15, help='MLM task prob (default: 0.15)')
parser.add_argument('--short_seq_prob', type=float, default=0.1, help='short sequence prob (default: 0.1)')
parser.add_argument('--use_nsp', type=int, default=1, help='use next sentence prediction task (default: 1)')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')

# model args
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: None)')
parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
                    help='model class name to use (default: transformers:BertForPreTraining)')
parser.add_argument('--init_checkpoint', type=str, help='path to init checkpoint to load a model from (default: None).')
parser.add_argument('--skip_used_data', action='store_true', default=False,
                    help='skip batches that were already seen by init_checkpoint (default: False)')

# tokenizer
# todo: add wordpiece tokenizers support?
parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')

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

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (default: 0.0)')
parser.add_argument('--scale_parameter', action='store_true', default=False,
                    help='Adafactor scale_parameter (default: False)')
parser.add_argument('--relative_step', action='store_true', default=False,
                    help='Adafactor relative_step (default: False)')
parser.add_argument('--warmup_init', action='store_true', default=False,
                    help='Adafactor warmup_init (default: False)')

# scheduler args
parser.add_argument('--lr_scheduler', type=str, default=None,
                    help='scheduler name from transformers.optimization: linear, cosine, cosine_with_restarts, '
                    'polynomial, constant, constant_with_warmup (default: None)')
parser.add_argument('--num_warmup_steps', type=int, default=None,
                    help='number of warming steps to get to lr (default: None)')
parser.add_argument('--num_training_steps', type=int, default=None,
                    help='number of training steps, if not set iters will be used (default: None)')
parser.add_argument('--reset_lr', action='store_true', default=False,
                    help='Do not load lr_scheduler from checkpoint and setup new (default: False)')

if __name__ == '__main__':
    args = parser.parse_args()
    # set current working dir
    args.working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(args.working_dir)
    if hvd.rank() == 0:
        logger.info(f'hvd size: {hvd.size()}')
        logger.info(f'FP16: {args.fp16}')

    # create model path and save configuration
    if hvd.rank() == 0:
        model_path = Path(args.model_path)
        if not model_path.exists():
            Path(model_path).mkdir(parents=True)
        args_dict = collect_run_configuration(args)
        # todo: if model path exists and there is config file, write new config file aside
        json.dump(args_dict, open(model_path/'config.json', 'w'), indent=4)

    tokenizer = _HFAutoTokenizer(args.tokenizer)
    # get train dataset
    if hvd.rank() == 0:
        logger.info(f'preparing training data from: {args.data_path}')
    data_path = Path(args.data_path).expanduser().absolute()
    train_data_index = get_indexed_dataset_(str(data_path), args.data_impl, skip_warmup=args.data_skip_warmup)

    train_dataset = BertDataset(indexed_dataset=train_data_index, masked_lm_prob=args.mlm_prob,
                                short_seq_prob=args.short_seq_prob, binary_head=args.use_nsp, tokenizer=tokenizer,
                                name=args.data_name, data_prefix=str(data_path),
                                num_epochs=args.data_n_epochs, max_num_samples=args.data_n_samples,
                                max_seq_length=args.input_seq_len, mask_label_id=-100, seed=args.seed,
                                )
    # shuffle train data each epoch (one loop over train_dataset) & drop last batch
    train_sampler = DistributedSampler(train_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=True,
                                       drop_last=True, seed=args.seed)

    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    global_batch_size = per_worker_batch_size * hvd.size()
    kwargs = {'pin_memory': True}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'
    train_dataloader = DataLoader(train_dataset, num_workers=args.data_n_workers, batch_size=per_worker_batch_size,
                                  sampler=train_sampler, **kwargs)
    # get validation dataset
    if args.valid_data_path:
        if hvd.rank() == 0:
            logger.info(f'preparing validation data from: {args.valid_data_path}')
        valid_data_path = Path(args.valid_data_path).expanduser().absolute()
        valid_data_index = get_indexed_dataset_(str(valid_data_path), args.data_impl, skip_warmup=args.data_skip_warmup)
        valid_dataset = BertDataset(indexed_dataset=valid_data_index, masked_lm_prob=args.mlm_prob,
                                    short_seq_prob=args.short_seq_prob, binary_head=args.use_nsp, tokenizer=tokenizer,
                                    name=args.data_name, data_prefix=str(valid_data_path),
                                    num_epochs=1, max_num_samples=None,  # take all validation data
                                    max_seq_length=args.input_seq_len, mask_label_id=-100, seed=args.seed)
        valid_sampler = DistributedSampler(valid_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=False)
        valid_dataloader = DataLoader(valid_dataset, num_workers=args.data_n_workers, batch_size=per_worker_batch_size,
                                      sampler=valid_sampler, **kwargs)
        if args.valid_interval is None:
            args.valid_interval = args.log_interval
    else:
        valid_dataloader = None
        if hvd.rank() == 0:
            logger.info('No validation data is used.')

    # define model
    model_cfg = AutoConfig.from_pretrained(args.model_cfg)
    # todo: get model class from model_cfg?
    model_cls = get_cls_by_name(args.model_cls)
    if hvd.rank() == 0:
        logger.info(f'Using model class: {model_cls}')
    model = model_cls(config=model_cfg)

    # define optimizer
    # todo: move to trainer?
    optimizer_cls = get_optimizer(args.optimizer)
    if optimizer_cls is None:
        raise RuntimeError(f'{args.optimizer} was not found in optimizers, torch.optim, transformers.optimization')

    if hvd.rank() == 0:
        logger.info(f'Using optimizer class: {optimizer_cls}')

    # todo: group optimizer params
    if optimizer_cls in [transformers.optimization.Adafactor, optimizers.Adafactor]:
        # https://github.com/huggingface/transformers/pull/9751/files -> transformers 4.3.0
        optimizer = optimizer_cls(model.parameters(), lr=args.lr,
                                  scale_parameter=args.scale_parameter,
                                  relative_step=args.relative_step,
                                  warmup_init=args.warmup_init,
                                  weight_decay=args.weight_decay)
    else:
        optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def batch_transform_fn(batch):
        return {
            'input_ids': batch['text'],
            'token_type_ids': batch['types'],
            'attention_mask': batch['padding_mask'],
            'labels': batch['labels'],
            'next_sentence_label': batch['is_random'],
        }

    def get_metrics_fn(output):
        # output - result of model(batch) call
        # only stateless metrics could be get in such way - metrics are averaged over batches
        # loss is a default metric, this function should be used if other metrics than loss should be logged
        metrics = {'loss': output['loss']}
        if 'mlm_loss' in output:
            metrics['loss_mlm'] = output['mlm_loss']
        if 'nsp_loss' in output:
            metrics['loss_nsp'] = output['nsp_loss']
        return metrics

    trainer = Trainer(args, model, optimizer, train_dataloader, valid_dataloader, train_sampler,
                      batch_transform_fn, get_metrics_fn)

    # train loop
    trainer.train()
