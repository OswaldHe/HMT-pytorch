import json
import logging
import sys
import os
import math
import shutil
from pathlib import Path
from itertools import chain
from megatron.data.dataset_utils import get_indexed_dataset_

import horovod.torch as hvd
# from dotenv import load_dotenv
import torch
import numpy as np
import datasets
from torch.utils.data import DataLoader, DistributedSampler
from datasets import Dataset

from lm_experiments_tools import TrainerArgs
from lm_experiments_tools.trainer import Trainer

from torch.nn.utils.rnn import pad_sequence

# load_dotenv()

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
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser  # noqa: E402

from lm_experiments_tools.utils import collect_run_configuration, get_cls_by_name, get_optimizer  # noqa: E402
import lm_experiments_tools.optimizers as optimizers  # noqa: E402

# limit # of CPU threads to be used per pytorch worker, otherwise it might use all cpus and throttle gpus
# > 2 fails cause of https://github.com/pytorch/pytorch/issues/56615
# need to upgrade to torch>1.8.1
torch.set_num_threads(4)
# all gpus set with CUDA_VISIBLE_DEVICES are visible to process, indexing from 0 to ...
torch.cuda.set_device(hvd.local_rank())

parser = HfArgumentParser(TrainerArgs)
parser.add_argument('--task_name', type=str, help="Task name, wikitext, ...")
parser.add_argument('--validate_only', action='store_true', default=False,
                    help='Skip training and run only validation. (default: False)')
parser.add_argument('--working_dir', type=str, default='.',
                    help='working dir, should be a dir with t5-experiments repo (default: .)')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--show_valid_examples', type=int, default=0,
                    help='how many valid examples to show during training (default: 0)')
parser.add_argument('--input_seq_len', type=int, default=128, help='input sequnce length (default: 128).')
parser.add_argument('--target_seq_len', type=int, default=16, help='target sequnce length, should be set to '
                                                                   'max(len(target))+1 for EOS (default: 16).')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')

parser.add_argument('--input_prefix', type=str, default='', help='add task prefix to an input string (default: "")')
parser.add_argument('--sliding_window', action='store_true', help='use slinding window attentinon mask, '
                    'eval on last segment only', default=False)

# model args
parser.add_argument('--from_pretrained', type=str, help='model name in HF Model Hub (default: "")')
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: "")')
# parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
#                     help='model class name to use (default: transformers:BertForPreTraining)')
parser.add_argument('--memory_cell_cls', type=str, default=None, help='cell class for RMT')
parser.add_argument('--recurrent_wrapper_cls', type=str, default=None, help='recurrent wrapper class for RMT')
parser.add_argument('--model_cpt', type=str, default=None, help='pretrained model checkpoint path')
parser.add_argument('--backbone_cls', type=str, default=None,
                    help='backbone class name to use for RMT')
parser.add_argument('--model_type', type=str, default='encoder-decoder',
                    help='model type, encoder, encoder-decoder, decoder, affects preprocessing '
                         '(default: encoder-decoder)')


# Aydar # RMT args 
parser.add_argument('--input_size', type=int, default=None, help='maximal input size of the backbone model')
parser.add_argument('--num_mem_tokens', type=int, default=None, help='number of memory tokens.')
parser.add_argument('--max_n_segments', type=int, default=1, help='maximal segment number')
parser.add_argument('--k2', type=int, default=-1, help='number of last segments used by backward')
parser.add_argument('--freeze_model_weights', action='store_true', default=False,
                    help='Stop training all model weights except memory layers')
parser.add_argument('--backbone_cpt', type=str, default=None, help='backbone model checkpoint path')
parser.add_argument('--vary_n_segments', action='store_true', default=False, help='Randomly choose segment number from 0 to max_n_segments')


# tokenizer
# todo: add wordpiece tokenizers support?
parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (default: 0.0)')
parser.add_argument('--scale_parameter', action='store_true', default=False,
                    help='Adafactor scale_parameter (default: False)')
parser.add_argument('--relative_step', action='store_true', default=False,
                    help='Adafactor relative_step (default: False)')
parser.add_argument('--warmup_init', action='store_true', default=False,
                    help='Adafactor warmup_init (default: False)')



if __name__ == '__main__':
    args = parser.parse_args()
    # set current working dir
    args.working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(args.working_dir)
    if hvd.rank() == 0:
        logger.info(f'hvd size: {hvd.size()}')
        logger.info(f'FP16: {args.fp16}')

    if hvd.rank() == 0 and args.model_path is None:
        logger.warning('model_path is not set: config, logs and checkpoints will not be saved.')

    # create model path and save configuration
    if hvd.rank() == 0 and args.model_path is not None:
        model_path = Path(args.model_path)
        if not model_path.exists():
            Path(model_path).mkdir(parents=True)
        args_dict = collect_run_configuration(args)
        # todo: if model path exists and there is config file, write new config file aside
        json.dump(args_dict, open(model_path/'config.json', 'w'), indent=4)

    if not args.from_pretrained:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)

    # Prepare datasets
    if hvd.rank() == 0:
        logger.info(f'preparing dataset for {args.task_name}')
    
    if 'wikitext' in args.task_name:
        raw_datasets = datasets.load_dataset('wikitext', args.task_name)
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )
    elif 'arxiv' in args.task_name:
        # from datasets import load_from_disk
        tokenized_datasets = datasets.load_from_disk('/home/bulatov/bulatov/datasets/arxiv_pile/processed/')
    else:
        raise(ValueError(f"Unknown dataset {args.task_name}"))

    if args.sliding_window:
        block_size = args.input_size // args.max_n_segments - 2 * args.num_mem_tokens
        history_size = args.input_size - block_size
    else:        
        block_size = args.input_size 
        if args.num_mem_tokens is not None:
            block_size -= 2 * args.num_mem_tokens
        history_size = args.input_seq_len - block_size

    def group_texts(examples, block_size, history_size=None):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if history_size is None:
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
        else:
            result = {
                k: [t[max({0, i - history_size}) : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
        result["labels"] = result["input_ids"].copy()
        return result

    id_pad_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if args.sliding_window:
        def collate_fn(batch):
            raise NotImplementedError
            # input_ids = [torch.tensor(b['input_ids']) for b in batch]
            # input_lens = [el.shape[-1] for el in input_ids]

            # labels = [torch.tensor(b['labels']) for b in batch]
            # attention_mask = [torch.tensor(b['attention_mask']) for b in batch]
            # input_ids = pad_sequence(input_ids, padding_value=id_pad_value).T
            # labels = pad_sequence(labels, padding_value=-100).T
            # attention_mask = pad_sequence(attention_mask, padding_value=0).T

            # # make sliding window att mask
            # attention_mask = attention_mask[:, None, :].repeat(1, attention_mask.shape[1], 1)
            # attention_mask = (torch.tril(attention_mask, 0) * (1 - torch.tril(attention_mask, -block_size)))

            # collated = {'input_ids': input_ids,
            #             'labels': labels, 
            #             'attention_mask': attention_mask}

            # if input_ids.shape[1] != block_size:
            #     # take only labels for last block (maybe use all labels during training?)
            #     labels_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            #     for i, lens in enumerate(input_lens):
            #         labels_mask[i, max(lens - block_size, 0): lens] = True
            #     collated['labels_mask'] = labels_mask

            # return collated
    else:
        def collate_fn(batch):
            input_ids = [torch.tensor(b['input_ids']) for b in batch]
            labels = [torch.tensor(b['labels']) for b in batch]
            labels_mask = [torch.ones_like(l, dtype=bool) for l in labels]
            attention_mask = [torch.tensor(b['attention_mask']) for b in batch]

            input_ids = pad_sequence(input_ids, padding_value=id_pad_value).T
            labels = pad_sequence(labels, padding_value=-100).T
            labels_mask = pad_sequence(labels_mask, padding_value=False).T
            attention_mask = pad_sequence(attention_mask, padding_value=0).T

            collated = {'input_ids': input_ids,
                        'labels': labels, 
                        'labels_mask': labels_mask,
                        'attention_mask': attention_mask}

            if args.vary_n_segments:
                n_segments = np.random.randint(1, args.max_n_segments + 1)
                n_tokens = n_segments * block_size
                for k in collated:
                    collated[k] = collated[k][:, -n_tokens:]

            return collated


    train_dataset = tokenized_datasets["train"].map(lambda x: group_texts(x, block_size, history_size), 
                                            batched=True, desc=f"Grouping train in chunks of {block_size} and history {history_size}")
    valid_dataset = tokenized_datasets["validation"].map(lambda x: group_texts(x, block_size, history_size), 
                                            batched=True, desc=f"Grouping valid in chunks of {block_size} and history {history_size}")

    
    # shuffle train data each epoch (one loop over train_dataset)
    train_sampler = DistributedSampler(train_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=True,
                                       drop_last=False, seed=args.seed)
    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    global_batch_size = per_worker_batch_size * hvd.size()
    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}
    train_dataloader = DataLoader(train_dataset, batch_size=per_worker_batch_size, sampler=train_sampler,
                                  collate_fn=collate_fn, **kwargs)
    
    
    # dataloader for validation
    # batch sample i is a continuation of sample i of the previous batch
    class alignedDataLoader(DataLoader):
        def __iter__(self):
            all_inds = np.arange(len(self.dataset) // self.batch_size * self.batch_size)
            all_inds = all_inds.reshape(self.batch_size, -1)
            for batch_ind in range(all_inds.shape[1]):
                batch = [self.dataset[int(ind)] for ind in all_inds[:, batch_ind]]
                yield self.collate_fn(batch)

    # get validation dataset
    valid_dataloader = None
    if hvd.rank() == 0:
        logger.info(f'preparing validation data from babilong')
    valid_sampler = DistributedSampler(valid_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=False)
    valid_dataloader = alignedDataLoader(valid_dataset, batch_size=per_worker_batch_size, sampler=valid_sampler,
                                  collate_fn=collate_fn, drop_last=True, **kwargs)
    
    # get test dataset
    if 'test' in tokenized_datasets.keys():
        test_dataset = tokenized_datasets["test"].map(lambda x: group_texts(x, block_size, history_size), 
                                            batched=True, desc=f"Grouping test in chunks of {block_size} and history {history_size}")
        test_sampler = DistributedSampler(test_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=per_worker_batch_size, sampler=test_sampler,
                                    collate_fn=collate_fn, drop_last=True, **kwargs)
    
    if args.valid_interval is None:
        args.valid_interval = args.log_interval

    # define model
    model_cls = get_cls_by_name(args.backbone_cls)
    if hvd.rank() == 0:
        logger.info(f'Using model class: {model_cls}')
    
    if not args.from_pretrained:
        model_cfg = AutoConfig.from_pretrained(args.model_cfg)
        model = model_cls(config=model_cfg)
    else:
        if hvd.rank() == 0:
            logger.info(f'Loading pretrained model: {args.from_pretrained}')
        model = model_cls.from_pretrained(args.from_pretrained)

    ## load cpt of backbone model
    if args.backbone_cpt:
        backbone_cpt = os.path.join(args.backbone_cpt, "model_best.pth")
        cpt = torch.load(backbone_cpt, map_location='cpu')
        model.load_state_dict(cpt['model_state_dict'], strict=False)
        if hvd.rank() == 0:
            logger.info(f'Loaded baseline state dict from: {args.backbone_cpt}')

    # Pass memory settings to pretrained model
    if args.num_mem_tokens is not None:
        # if args.memory_forward_func is not None:
        #     args.memory_forward_func = get_cls_by_name(args.memory_forward_func)

        memory_cell_cls = get_cls_by_name(args.memory_cell_cls)
        recurrent_wrapper_cls = get_cls_by_name(args.recurrent_wrapper_cls)
        if hvd.rank() == 0:
            logger.info(f'Wrapping in: {memory_cell_cls} and {recurrent_wrapper_cls}')
        
        
        cell = memory_cell_cls(model, args.num_mem_tokens)
        model = recurrent_wrapper_cls(cell, 
                                      segment_size=block_size,
                                      max_n_segments=args.max_n_segments, 
                                      k2=args.k2
        )
                                    

        ## load cpt of rmt
        if args.model_cpt:
            model_cpt = os.path.join(args.model_cpt, "model_best.pth")
            cpt = torch.load(model_cpt, map_location='cpu')
            model.load_state_dict(cpt['model_state_dict'], strict=False)
            if hvd.rank() == 0:
                logger.info(f'Loaded RMT state dict from: {args.model_cpt}')

    if args.freeze_model_weights:
        for n, p in model.named_parameters():
            # if 'memory' not in n and 'wte' not in n:
            if 'memory' not in n and 'lora' not in n:
                p.requires_grad = False
        if hvd.rank() == 0:
            logger.info(f'Frozen moodel weights')
            logger.info(f'Remaining parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}')

    # fix the not-contiguous error with loralib and horovod
    def make_contiguous(module):
        with torch.no_grad():
            for param in module.parameters():
                param.set_(param.contiguous())
    make_contiguous(model)
    
    # define optimizer
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

    # for encoder only classification
    def keep_for_metrics_fn(batch, output):
        # select data from batch and model output that would be used to compute metrics
        data = {}
        data['labels'] = batch['labels']
        data['loss'] = output['loss']
        data['predictions'] = torch.argmax(output['logits'].detach(), dim=-1)
        return data

    # HF datasets can compute metrics on each gpu process and then aggregate them on process with rank 0
    # synchronization is done by using temporay files on a shared filesystem
    # rank and number of workers is set by num_process and process_id params
    # BUT our Trainer aggregates all prediction from all gpus!
    #   this will lead to computing metrics for predictions repeated xN_GPUS times
    # need to try:
    # - keep_in_memory=True, may lead to OOM for large validation sets, after sync predictions and targets for the full
    #       validation set would be stored on each GPU -> xN_GPUs RAM
    #   - implemented currently
    # - compute metrics on batch lvl
    # - add support of HF metrics and turn off aggregation in case if metric has .add_batch method
    # scrolls_metric = datasets.load_metric(scrolls_metric_path, args.task_name, keep_in_memory=True)

    def metrics_fn(data):
        # compute metrics based on stored labels, predictions, ...
        metrics = {}
        y, p = data['labels'], data['predictions']
        if hvd.rank() == 0 and args.show_valid_examples > 0:
            for i in range(min(args.show_valid_examples, len(y))):
                # logger.info(f'y: {tokenizer.decode(y[i])}')
                # logger.info(f'p: {tokenizer.decode(p[i])}')
                logger.info(f'y: {y[i]}')
                logger.info(f'p: {p[i]}')
                logger.info('-' * 50)
        try:
            perplexity = math.exp(data["loss"].mean())
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        return metrics

    batch_metrics_fn = lambda _, y: {key: y[key] for key in y.keys() if (('loss' in key) or ('!log' in key))}
    trainer = Trainer(args, model, optimizer, train_dataloader, valid_dataloader, train_sampler,
                      keep_for_metrics_fn=keep_for_metrics_fn, metrics_fn=metrics_fn,
                      ###booydar
                      batch_metrics_fn=batch_metrics_fn,
                      generate_kwargs={})

    if not args.validate_only:
        # train loop
        trainer.train()
        # make sure all workers are done
        hvd.barrier()
        # run validation after training
        if args.save_best:
            best_model_path = str(Path(args.model_path) / 'model_best.pth')
            if hvd.rank() == 0:
                logger.info(f'Loading best saved model from {best_model_path}')
            trainer.load(best_model_path)
        if valid_dataloader is not None:
            if hvd.rank() == 0:
                logger.info('Runnning validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=False)
        if test_dataloader is not None:
            if hvd.rank() == 0:
                logger.info('Runnning validation on test data:')
            trainer.validate(test_dataloader, write_tb=False)
    else:
        # run validation, do not write to tensorboard
        if hvd.rank() == 0:
            logger.info('Running validation on train set:')
        # trainer.validate(train_dataloader, split='train', write_tb=True)
        if valid_dataloader is not None:
            if hvd.rank() == 0:
                logger.info('Running validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=True)
        # if test_dataloader is not None:
        #     if hvd.rank() == 0:
        #         logger.info('Runnning validation on test data:')
        #     trainer.validate(test_dataloader, write_tb=True)
