import json
import logging
import os
from pathlib import Path

import t5  # noqa: F401 core_dump without t5 import here 🤦‍♂️
from t5.seqio.dataset_providers import ShardInfo
import horovod.torch as hvd
from dotenv import load_dotenv
import tensorflow.compat.v1 as tf
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from transformers import HfArgumentParser

from lm_experiments_tools import Trainer, TrainerArgs

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
from transformers import T5Config, T5Tokenizer  # noqa: E402

from data_utils import T5PretrainingDataset, assert_vocabs, jsonl_preprocessor  # noqa: E402
from lm_experiments_tools.utils import collect_run_configuration, get_cls_by_name, get_optimizer  # noqa: E402
from lm_experiments_tools.utils import get_git_diff  # noqa: E402
import lm_experiments_tools.optimizers as optimizers  # noqa: E402

tf.config.set_visible_devices([], 'GPU')  # turn off GPUs for tf operations
# limit cpu threads for tf
# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)

# limit # of CPU threads to be used per pytorch worker, otherwise it will use all cpus and throttle gpus
torch.set_num_threads(4)
# all gpus set with CUDA_VISIBLE_DEVICES are visible to process, indexing from 0 to ...
torch.cuda.set_device(hvd.local_rank())


parser = HfArgumentParser(TrainerArgs)
parser.add_argument('--data_path', type=str, help='path with the sharded data in jsonl format')
parser.add_argument('--valid_data_path', type=str, help='path with the sharded data in jsonl format for validation')
parser.add_argument('--working_dir', type=str, default='.',
                    help='working dir, should be a dir with t5-experiments repo (default: .)')

# model args
parser.add_argument('--base_model', type=str, default='t5-base',
                    help='base model name (from huggingface) (default: t5-base)')
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: None)')
parser.add_argument('--model_cls', type=str, default='transformers:T5ForConditionalGeneration',
                    help='model class name to use (default: transformers:T5ForConditionalGeneration)')

parser.add_argument('--input_seq_len', type=int, default=128, help='input sequnce length (default: 128).')
parser.add_argument('--target_seq_len', type=int, default=128, help='target sequnce length (default: 128).')
parser.add_argument('--vocab', type=str, default='./vocabs/sentencepiece.model',
                    help='path to vocabulary file with sentencepiece model (default: ./vocabs/sentencepiece.model)')

# training args
parser.add_argument('--task', type=str, default='span_corruption',
                    help='t5 task name, e.g. `span_corruption`, `prefix_lm`. (default: span_corruption)')

parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (default: 0.0)')
parser.add_argument('--scale_parameter', action='store_true', default=False,
                    help='Adafactor scale_parameter (default: False)')
parser.add_argument('--relative_step', action='store_true', default=False,
                    help='Adafactor relative_step (default: False)')
parser.add_argument('--warmup_init', action='store_true', default=False,
                    help='Adafactor warmup_init (default: False)')

if __name__ == '__main__':
    # run with horovod:
    # export CUDA_VISIBLE_DEVICES=0,1,2; horovodrun --gloo -np 3 python run_t5_pretraining.py
    args = parser.parse_args()
    # set current working dir
    # todo: maybe take path to run_t5_pretraining.py?
    args.working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(args.working_dir)
    if hvd.rank() == 0:
        logger.info(f'hvd size: {hvd.size()}')
        logger.info(f'FP16: {args.fp16}')

    kwargs = {'pin_memory': True}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    # create model_path and save configuration
    if hvd.rank() == 0:
        model_path = Path(args.model_path)
        if not model_path.exists():
            Path(model_path).mkdir(parents=True)
        args_dict = collect_run_configuration(args)
        # todo: if model path exists and there is config file, write new config file aside
        json.dump(args_dict, open(model_path / 'config.json', 'w'), indent=4)
        open(model_path / 'git.diff', 'w').write(get_git_diff())

    # get train dataset shards
    data_path = Path(args.data_path).expanduser().absolute()
    shards = list(sorted([sh.name for sh in data_path.glob('*.jsonl')]))
    # split shards across workers, drop remainders
    shards_per_worker = len(shards) // hvd.size()
    shards = shards[hvd.rank() * shards_per_worker:(hvd.rank() + 1) * shards_per_worker]
    logger.info(f'worker {hvd.rank()} shards: {shards}')
    # absolute path to shards
    shards = [str(data_path / sh) for sh in shards]

    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    global_batch_size = per_worker_batch_size * hvd.size()

    train_data = T5PretrainingDataset(shards, task=args.task, batch_size=per_worker_batch_size,
                                      text_preprocessor=jsonl_preprocessor,
                                      inputs_len=args.input_seq_len, targets_len=args.target_seq_len,
                                      vocab_path=args.vocab)
    # fails to work if num_workes > 0 cause we are using tf.datasets
    train_dataloader = DataLoader(train_data, num_workers=0, batch_size=None, **kwargs)

    # get validation dataset
    if args.valid_data_path:
        valid_data_path = Path(args.valid_data_path).expanduser().absolute()
        valid_shards = list(sorted([sh.name for sh in valid_data_path.glob('*.jsonl')]))
        # split shards across workers, drop remainders
        logger.info(f'worker {hvd.rank()} validation shards: {valid_shards}')
        # absolute path to shards
        valid_shards = [str(valid_data_path / sh) for sh in valid_shards]
        valid_data = T5PretrainingDataset(valid_shards, task=args.task, batch_size=per_worker_batch_size,
                                          text_preprocessor=jsonl_preprocessor,
                                          inputs_len=args.input_seq_len, targets_len=args.target_seq_len,
                                          vocab_path=args.vocab,
                                          shard_info=ShardInfo(index=hvd.rank(), num_shards=hvd.size()))
        valid_dataloader = DataLoader(valid_data, num_workers=0, batch_size=None, **kwargs)
        if args.valid_interval is None:
            args.valid_interval = args.log_interval
    else:
        valid_dataloader = None
        if hvd.rank() == 0:
            logger.info('No validation data is used.')

    # define model
    if not args.model_cfg:
        t5config = T5Config.from_pretrained(args.base_model)
    else:
        t5config = T5Config.from_json_file(args.model_cfg)
        # todo: get tokenizer from config
        logger.warning(f'Model configuration was taken from {args.model_cfg}, but tokenizer from {args.base_model}')
    # define tokenizer
    t5_default_tokenizer = T5Tokenizer.from_pretrained(args.base_model)
    if hvd.rank() == 0:
        assert_vocabs(t5_default_tokenizer, args.vocab)

    model_cls = get_cls_by_name(args.model_cls)  # transfomers:T5ForConditionalGeneration or modeling_t5:my_class
    if hvd.rank() == 0:
        logger.info(f'Using model class: {model_cls}')
    model = model_cls(config=t5config)

    # define optimizer
    optimizer_cls = get_optimizer(args.optimizer)
    if optimizer_cls is None:
        raise RuntimeError(f'{args.optimizer} was not found in optimizers, torch.optim, transformers.optimization')

    if hvd.rank() == 0:
        logger.info(f'Using optimizer class: {optimizer_cls}')

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
            'input_ids': batch['inputs'],
            'attention_mask': batch['inputs_mask'],
            'labels': batch['targets'],
            'decoder_attention_mask': batch['targets_mask'],
        }

    trainer = Trainer(args, model, optimizer, train_dataloader, valid_dataloader, batch_transform_fn=batch_transform_fn)

    # train loop
    trainer.train()
