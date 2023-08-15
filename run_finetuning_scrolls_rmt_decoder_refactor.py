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
from dotenv import load_dotenv
import torch
import numpy as np
import datasets
from torch.utils.data import DataLoader, DistributedSampler
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from sklearn.metrics import f1_score, accuracy_score

from lm_experiments_tools import TrainerArgs
from lm_experiments_tools.trainer_dec import Trainer

from torch.nn.utils.rnn import pad_sequence
from lm_experiments_tools.lm_datasets import get_lm_datasets

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
parser.add_argument('--task_name', type=str, help='Scrolls task name: "gov_report", "summ_screen_fd", "qmsum", '
                                                  '"narrative_qa", "qasper", "quality", "contract_nli"')
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
parser.add_argument('--sliding_window', action='store_true', help='use slinding window attention mask, '
                    'eval on last segment only', default=False)

# model args
parser.add_argument('--from_pretrained', type=str, help='model name in HF Model Hub (default: "")')
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: "")')
parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
                    help='model class name to use (default: transformers:BertForPreTraining)')
parser.add_argument('--memory_cell_cls', type=str, default=None, help='cell class for RMT')
parser.add_argument('--recurrent_wrapper_cls', type=str, default=None, help='recurrent wrapper class for RMT')
parser.add_argument('--model_cpt', type=str, default=None, help='pretrained model checkpoint path')
parser.add_argument('--model_type', type=str, default='encoder-decoder',
                    help='model type, encoder, encoder-decoder, decoder, affects preprocessing '
                         '(default: encoder-decoder)')


# Aydar # RMT args 
parser.add_argument('--input_size', type=int, default=None, help='maximal input size of the backbone model')
parser.add_argument('--num_mem_tokens', type=int, default=None, help='number of memory tokens.')
parser.add_argument('--max_n_segments', type=int, default=1, help='maximal segment number')
parser.add_argument('--vary_n_segments', action='store_true', default=False, help='Randomly choose segment number from 1 to max_n_segments')
parser.add_argument('--segment_alignment', type=str, default=None, help="How to align segments when splitting input")
# parser.add_argument('--sum_loss', action='store_true', default=False,
#                     help='with this flag task loss from all segments is summed')
# parser.add_argument('--bptt_depth', type=int, default=-1, help='max number of previous segments in gradient computation.')
# parser.add_argument('--segment_ordering', type=str, help='segment order', default='regular',
#                     choices=['regular', 'reversed', 'bidirectional', 'repeat_first', 'last_memory_only'])
# parser.add_argument('--memory_forward_func', type=str, help='path to memory forward funсtion script', default=None)
# parser.add_argument('--memory_layers', type=str, help='memory-augmented layer inds or "all" for all layers', default=None)
# parser.add_argument('--share_memory_layers', action='store_true', help='share weights of memory layers', default=False)
# parser.add_argument('--reconstruction_loss_coef', type=float, default=None,
#                     help='reconstuction loss ratio in total loss')
# # parser.add_argument('--segment_ordering', type=str,help='????', default='regular',
# #                     choices=['regular', 'reversed', 'bidirectional', 'repeat_first', 'last_memory_only'])
# parser.add_argument('--retain_graph', action='store_true', help='Retain computation graph during backward pass', default=False)
# parser.add_argument('--use_truncated_backward', action='store_true', default=False,
#                     help='whether to use RMT truncated bptt method in backward')
# parser.add_argument('--k1', type=int, default=-1, help='(not implemented) If not -1, gradient update is done each k1 segments')
parser.add_argument('--k2', type=int, default=-1, help='number of last segments used by backward')
parser.add_argument('--freeze_model_weights', action='store_true', default=False,
                    help='Stop training all model weights except memory layers')
parser.add_argument('--backbone_cpt', type=str, default=None, help='backbone model checkpoint path')


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


def download_metric():
    scrolls_metric_path = hf_hub_download(repo_id="tau/scrolls", filename="metrics/scrolls.py", repo_type="dataset")
    updated_scrolls_metric_path = (
        os.path.dirname(scrolls_metric_path) + os.path.basename(scrolls_metric_path).replace(".", "_") + ".py"
    )
    shutil.copy(scrolls_metric_path, updated_scrolls_metric_path)
    return updated_scrolls_metric_path


scrolls_metric_path = download_metric()

task_to_metric = {
    'gov_report': ['rouge/rouge1', 'rouge/rouge2', 'rouge/rougeL', 'rouge/rougeLsum', 'rouge/geometric_mean'],
    'summ_screen_fd': ['rouge/rouge1', 'rouge/rouge2', 'rouge/rougeL', 'rouge/rougeLsum', 'rouge/geometric_mean'],
    'qmsum': ['rouge/rouge1', 'rouge/rouge2', 'rouge/rougeL', 'rouge/rougeLsum', 'rouge/geometric_mean'],
    'narrative_qa': ['f1'],
    'qasper': ['f1'],
    'quality': ['exact_match'],
    'contract_nli': ['exact_match']
}

tasks_with_duplicates = {'narrative_qa', 'qasper'}


# https://github.com/tau-nlp/scrolls/blob/5bfb8dbaf3a0128ac8c65922096fd95a645f6ba2/baselines/src/utils/duplicates.py#L1
# some tasks have multiple possible labels for single input, drop_duplicates_in_input will collect such labels
def drop_duplicates_in_input(untokenized_dataset):
    indices_to_keep = []
    id_to_idx = {}
    outputs = []
    for i, (id_, output) in enumerate(zip(untokenized_dataset["id"], untokenized_dataset["output"])):
        if id_ in id_to_idx:
            outputs[id_to_idx[id_]].append(output)
            continue
        indices_to_keep.append(i)
        id_to_idx[id_] = len(outputs)
        outputs.append([output])
    untokenized_dataset = untokenized_dataset.select(indices_to_keep).flatten_indices()
    untokenized_dataset = untokenized_dataset.remove_columns("output")
    untokenized_dataset = untokenized_dataset.add_column("outputs", outputs)
    return untokenized_dataset


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

    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.model_type == 'encoder-decoder':
        raise NotImplementedError
        # global_attention_first_token = False  # should be True for LED
        # encode_plus_kwargs = {'truncation': True, 'padding': 'longest', 'pad_to_multiple_of': 1}
        # # generate_kwargs = {'max_length': args.target_seq_len, 'min_length': args.target_seq_len}
        # generate_kwargs = {}

        # def collate_fn(batch):
        #     # cut too long strings because they may slow down tokenization
        #     inputs = [b['input'][:args.input_seq_len * 10] for b in batch]
        #     if 'outputs' in batch[0]:
        #         # if we have more than 1 label per example (only in valid) take only one of them
        #         # to compute loss on valid
        #         labels = [b['outputs'][0][:args.target_seq_len * 10] for b in batch]
        #     else:
        #         labels = [b['output'][:args.target_seq_len * 10] for b in batch]
        #     if args.input_prefix:
        #         inputs = [args.input_prefix + inp for inp in inputs]
        #     features = tokenizer.batch_encode_plus(list(inputs), max_length=args.input_seq_len, return_tensors='pt',
        #                                            **encode_plus_kwargs)
        #     with tokenizer.as_target_tokenizer():
        #         labels = tokenizer.batch_encode_plus(list(labels), max_length=args.target_seq_len, return_tensors='pt',
        #                                              **encode_plus_kwargs).input_ids
        #     labels[labels == tokenizer.pad_token_id] = -100
        #     features['labels'] = labels
        #     features['id'] = [b['id'] for b in batch]
        #     if 'outputs' in batch[0]:
        #         features['target_text'] = [b['outputs'] for b in batch]
        #     else:
        #         features['target_text'] = [b['output'] for b in batch]
        #     if 'global_attention_mask' in features:
        #         raise RuntimeError('What global attention mask for Longformer and LongformerEncoder-Decoder should be?')
        #     return features

    elif args.model_type == 'encoder' and args.task_name == 'contract_nli':
        raise NotImplementedError
        # if args.use_generate_on_valid:
        #     raise RuntimeError('use_generate_on_valid should be set to False for encoder-only models')

        # encode_plus_kwargs = {'max_length': args.input_seq_len,
        #                       'truncation': True,
        #                       'padding': 'longest',
        #                       'pad_to_multiple_of': 1}
        # generate_kwargs = {}
        # labels_map = {'Contradiction': 0, 'Entailment': 1, 'Not mentioned': 2}
        # num_labels = len(labels_map)

        # def collate_fn(batch):
        #     # cut too long strings because they may slow down tokenization
        #     inputs = [b['input'][:args.input_seq_len * 10] for b in batch]
        #     labels = [b['output'][:args.target_seq_len * 10] for b in batch]
        #     if args.input_prefix:
        #         inputs = [args.input_prefix + inp for inp in inputs]
        #     features = tokenizer.batch_encode_plus(list(inputs), return_tensors='pt', **encode_plus_kwargs)
        #     labels = np.array([labels_map[t] for t in labels])
        #     features['labels'] = torch.from_numpy(labels)
        #     return features

    elif args.model_type == 'decoder':
        from torch.nn.utils.rnn import pad_sequence

        # tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.pad_token_id = tokenizer.eos_token
        tokenizer.add_special_tokens({'additional_special_tokens': ['[GEN]', '[PAD]']})
        gen_token = tokenizer.encode('[GEN]')[0]
        tokenizer.pad_token_id = tokenizer.encode('[PAD]')[0]
        id_pad_value = tokenizer.pad_token_id

        block_size = args.input_size
        if args.num_mem_tokens not in {0, None}:
            block_size -= 2 * args.num_mem_tokens

        def collate_fn(batch):
            inputs = [b['input'][:args.input_seq_len * 10] for b in batch]
            labels = [b['output'][:args.input_seq_len * 10] for b in batch]

            collated = {}
            inputs = tokenizer.batch_encode_plus(list(inputs), padding=False)
            labels = tokenizer.batch_encode_plus(list(labels), padding=False)

            full_inputs = [torch.tensor(i[:args.input_seq_len - len(l) - 1] + [gen_token] + l) for i, l in zip(inputs['input_ids'], labels['input_ids'])]
            full_inputs = pad_sequence(full_inputs, padding_value=tokenizer.pad_token_id).T

            gen_inputs = [torch.tensor(i[:args.input_seq_len - 1] + [gen_token]) for i in inputs['input_ids']]
            gen_inputs = pad_sequence(gen_inputs, padding_value=tokenizer.pad_token_id).T
            
            labels_mask = torch.zeros_like(full_inputs).bool()
            for i, l in enumerate(labels['input_ids']):
                labels_mask[i, -len(l) -1:] = True

            collated['input_ids'] = collated['labels'] = full_inputs
            collated['input_ids_generate'] = gen_inputs
            collated['labels_mask'] = labels_mask
            collated['attention_mask'] = (collated['input_ids'] != id_pad_value).bool()

            collated['id'] = [b['id'] for b in batch]
            # if 'outputs' in batch[0]:
            #     collated['target_text'] = [b['outputs'] for b in batch]
            # else:
            collated['target_text'] = [b['output'] for b in batch]
            return collated
        # def collate_train(batch):
        #     inputs = [b['input'][:args.input_seq_len * 10] for b in batch]
        #     labels = [b['output'][:args.input_seq_len * 10] for b in batch]

        #     collated = {}
        #     inputs = tokenizer.batch_encode_plus(list(inputs), padding=False)
        #     labels = tokenizer.batch_encode_plus(list(labels), padding=False)

        #     full_inputs = [torch.tensor(i[:args.input_seq_len - len(l) - 1] + [gen_token] + l) for i, l in zip(inputs['input_ids'], labels['input_ids'])]
        #     full_inputs = pad_sequence(full_inputs, padding_value=tokenizer.pad_token_id).T
            
        #     labels_mask = torch.zeros_like(full_inputs).bool()
        #     for i, l in enumerate(labels['input_ids']):
        #         labels_mask[i, -len(l) -1:] = True

        #     collated['input_ids'] = collated['labels'] = full_inputs
        #     collated['labels_mask'] = labels_mask
        #     collated['attention_mask'] = (collated['input_ids'] != id_pad_value).bool()
        #     return collated

        # def collate_valid(batch):
        #     inputs = [b['input'][:args.input_seq_len * 10] for b in batch]

        #     collated = {}
        #     inputs = tokenizer.batch_encode_plus(list(inputs), padding=False)
        #     full_inputs = [torch.tensor(i[:args.input_seq_len - 1] + [gen_token]) for i in inputs['input_ids']]
        #     full_inputs = pad_sequence(full_inputs, padding_value=tokenizer.pad_token_id).T
            
        #     collated['input_ids'] = full_inputs
        #     collated['attention_mask'] = (collated['input_ids'] != id_pad_value).bool()
        #     collated['target_text'] = [b['output'] for b in batch]
        #     return collated
            
    else:
        raise NotImplementedError(f'Unknown model type {args.model_type}')

    # get train dataset
    if hvd.rank() == 0:
        logger.info(f'preparing dataset for: {args.task_name}')
    dataset = datasets.load_dataset('tau/scrolls', args.task_name)
    train_dataset = dataset['train']
    # shuffle train data each epoch (one loop over train_dataset)
    train_sampler = DistributedSampler(train_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=True,
                                       drop_last=False, seed=args.seed)
    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    global_batch_size = per_worker_batch_size * hvd.size()
    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}
    train_dataloader = DataLoader(train_dataset, batch_size=per_worker_batch_size, sampler=train_sampler,
                                  collate_fn=collate_fn, **kwargs)
                                #   collate_fn=collate_train, **kwargs)
    # get validation dataset
    valid_dataloader = None
    if hvd.rank() == 0:
        logger.info(f'preparing validation data from: {args.task_name}')
    valid_dataset = dataset['validation']
    if args.task_name in tasks_with_duplicates:
        valid_dataset = drop_duplicates_in_input(valid_dataset)
    valid_sampler = DistributedSampler(valid_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=per_worker_batch_size, sampler=valid_sampler,
                                  collate_fn=collate_fn, **kwargs)
                                #   collate_fn=collate_valid, **kwargs)
    if args.valid_interval is None:
        args.valid_interval = args.log_interval

    # define model
    model_cls = get_cls_by_name(args.model_cls)

    if hvd.rank() == 0:
        logger.info(f'Using model class: {model_cls}')
    if not args.from_pretrained:
        model_cfg = AutoConfig.from_pretrained(args.model_cfg)
        if args.model_type == 'encoder' and args.task_name == 'contract_nli':
            model_cfg.num_labels = num_labels
        model = model_cls(config=model_cfg)
    else:
        if hvd.rank() == 0:
            logger.info(f'Loading pretrained model: {args.from_pretrained}')
        model = model_cls.from_pretrained(args.from_pretrained)

    ## add [GEN] token
    model.resize_token_embeddings(len(tokenizer))
    
    ## load cpt of backbone model
    if args.backbone_cpt:
        backbone_cpt = os.path.join(args.backbone_cpt, "model_best.pth")
        cpt = torch.load(backbone_cpt, map_location='cpu')
        model.load_state_dict(cpt['model_state_dict'], strict=False)
        if hvd.rank() == 0:
            logger.info(f'Loaded baseline state dict from: {args.backbone_cpt}')

    # Pass memory settings to pretrained model
    if args.num_mem_tokens is not None:
        memory_cell_cls = get_cls_by_name(args.memory_cell_cls)
        recurrent_wrapper_cls = get_cls_by_name(args.recurrent_wrapper_cls)
        if hvd.rank() == 0:
            logger.info(f'Wrapping in: {memory_cell_cls} and {recurrent_wrapper_cls}')
        
        
        cell = memory_cell_cls(model, args.num_mem_tokens)
        model = recurrent_wrapper_cls(cell, 
                                      segment_size=block_size,
                                      max_n_segments=args.max_n_segments, 
                                      vary_n_segments=args.vary_n_segments,
                                      k2=args.k2,
                                      segment_alignment=args.segment_alignment
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

    # # fix the not-contiguous error with loralib and horovod
    # def make_contiguous(module):
    #     with torch.no_grad():
    #         for param in module.parameters():
    #             param.set_(param.contiguous())
    # make_contiguous(model)
    
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
        if 'generation_outputs' in output:
            data['labels'] = batch['target_text']

            data['generation_outputs'] = output['generation_outputs']
            # if 'labels_mask' in batch:
            #     data['generation_outputs'] = [data['generation_outputs'][i, mask] for i, mask in enumerate(batch['labels_mask'])]
        # if args.model_type == 'encoder':
            
            ##### booydar
            # data['predictions'] = torch.argmax(output['logits'].detach(), dim=-1)
        # data['labels'] = batch['labels']
        for key in batch.keys():
            if 'loss' in key: 
                data[key] = batch[key]
        # else:

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
    scrolls_metric = datasets.load_metric(scrolls_metric_path, args.task_name, keep_in_memory=True)

    def metrics_fn(data):
        # compute metrics based on stored labels, predictions, ...
        metrics = {}
        y, p = None, None
        if 'generation_outputs' in data:
            # replace -100 with pad token in labels
            y = data['labels']
            p = tokenizer.batch_decode(data['generation_outputs'], skip_special_tokens=True)

            metrics['exact_match'] = np.mean([y_ == p_[:len(y_)] for p_, y_ in zip (p, y)])
            # preds = tokenizer.batch_decode(data['generation_outputs'], skip_special_tokens=False)
            # p = [p[:p.index(tokenizer.eos_token)] if tokenizer.eos_token in p else p for p in preds]
            if hvd.rank() == 0 and args.show_valid_examples > 0:
                for i in range(min(args.show_valid_examples, len(y))):
                    logger.info(f'y: {y[i]}')
                    logger.info(f'p: {p[i]}')
                    logger.info(f'p ids: {data["generation_outputs"][i]}')
                    # logger.info('\n'.join([(y_, p_[:len(y_)], y_==p_[:len(y_)]) for p_, y_ in zip (p, y[:30])]))

                    logger.info('-' * 50)
            # todo: do we need to better clean P to remove tokens after eos? not remove special tokens only
        # elif args.model_type == 'encoder':
        #     y, p = data['labels'], data['predictions']

        # if y is not None and p is not None:
            # if args.model_type == 'encoder-decoder':
            # if not isinstance(y[0], list):
                # y = [[_y] for _y in y]
            # result = scrolls_metric.compute(predictions=p, references=y)
            # for metric_name in task_to_metric[args.task_name]:
            #     metrics[metric_name] = result[metric_name]

            # metrics['exact_match'] = np.mean([y_ == p_[:len(y_)] for p_, y_ in zip (p, y)])
            # elif args.model_type == 'encoder' and args.task_name == 'contract_nli':
            #     metrics['exact_match'] = accuracy_score(y, p) * 100
            #     metrics['f1_micro'] = f1_score(y, p, average='micro')
        return metrics

    ### booydar
    batch_metrics_fn = lambda _, y: {key: y[key] for key in y.keys() if (('loss' in key) or ('!log' in key))}
    trainer = Trainer(args, model, optimizer, train_dataloader, valid_dataloader, train_sampler,
                      keep_for_metrics_fn=keep_for_metrics_fn, metrics_fn=metrics_fn,
                      ###booydar
                      batch_metrics_fn=batch_metrics_fn,
                      generate_kwargs={'pad_token_id': tokenizer.pad_token_id})

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
    else:
        # run validation, do not write to tensorboard
        if hvd.rank() == 0:
            logger.info('Running validation on train set:')
        trainer.validate(train_dataloader, split='train', write_tb=True)
        if valid_dataloader is not None:
            if hvd.rank() == 0:
                logger.info('Running validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=True)
        # if test_dataloader is not None:
        #     if hvd.rank() == 0:
        #         logger.info('Runnning validation on test data:')
        #     trainer.validate(test_dataloader, write_tb=True)
