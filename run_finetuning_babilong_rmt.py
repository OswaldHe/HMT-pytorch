import json
import logging
import sys
import os
import shutil
from pathlib import Path

from megatron.data.dataset_utils import get_indexed_dataset_

import horovod.torch as hvd
from dotenv import load_dotenv
import torch
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from sklearn.metrics import f1_score, accuracy_score

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
# parser.add_argument('--task_name', type=str, help='Scrolls task name: "gov_report", "summ_screen_fd", "qmsum", '
#                                                   '"narrative_qa", "qasper", "quality", "contract_nli"')
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

# model args
parser.add_argument('--from_pretrained', type=str, help='model name in HF Model Hub (default: "")')
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: "")')
parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
                    help='model class name to use (default: transformers:BertForPreTraining)')
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
parser.add_argument('--sum_loss', action='store_true', default=False,
                    help='with this flag task loss from all segments is summed')
parser.add_argument('--bptt_depth', type=int, default=-1, help='max number of previous segments in gradient computation.')
parser.add_argument('--segment_ordering', type=str, help='segment order', default='regular',
                    choices=['regular', 'reversed', 'bidirectional', 'repeat_first', 'last_memory_only'])
parser.add_argument('--memory_forward_func', type=str, help='path to memory forward funсtion script', default=None)
parser.add_argument('--memory_layers', type=str, help='memory-augmented layer inds or "all" for all layers', default=None)
parser.add_argument('--share_memory_layers', action='store_true', help='share weights of memory layers', default=False)
parser.add_argument('--reconstruction_loss_coef', type=float, default=None,
                    help='reconstuction loss ratio in total loss')
# parser.add_argument('--segment_ordering', type=str,help='????', default='regular',
#                     choices=['regular', 'reversed', 'bidirectional', 'repeat_first', 'last_memory_only'])


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



names = ['Mary', 'John', 'Daniel', 'Sandra']
actions = ['moved', 'went', 'went back', 'journeyed', 'travelled']
places = ['bathroom', 'hallway', 'garden', 'office', 'bedroom', 'kitchen']
choices_dict = {'names': names, 'actions': actions, 'places': places}

class MemoryDataset(Dataset):
    def __init__(self, choices_dict=choices_dict, num_facts=1, split='train', dataset='quality'):
        self.choices_dict = choices_dict
        self.dataset = load_dataset('tau/scrolls', dataset)[split]
        self.num_facts = num_facts

    def __getitem__(self, ind):
        sample = self.dataset[ind]
        sample['fact'], sample['question'], sample['answer'] = self.generate_qa() 
        return sample
    
    def __len__(self):
        return len(self.dataset)

    def generate_qa(self):
        names, actions, places = self.choices_dict['names'], self.choices_dict['actions'], self.choices_dict['places']

        np.random.shuffle(names)
        facts, questions, answers = [], [], []
        for fact_num, name in zip(range(self.num_facts), names):
            action, place = np.random.choice(actions), np.random.choice(places)

            facts.append(f'{name} {action} to the {place}')
            questions.append(f'Where is {name}?')
            answers.append(place)

        facts = ', '.join(facts) + '.'
        questions = ' '.join(questions)
        answers = ', '.join(answers)
        
        return facts, questions, answers


if __name__ == '__main__':
    args = parser.parse_args()
    # print('\n\n\n\n\n\n\nBs', args.batch_size)
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


    # get datasets
    if hvd.rank() == 0:
        logger.info(f'preparing dataset for babilong')
    
    train_dataset = MemoryDataset(choices_dict, num_facts=1, split='train', dataset='quality')
    valid_dataset = MemoryDataset(choices_dict, num_facts=1, split='validation', dataset='quality')
    
    answers = train_dataset.choices_dict['places']
    labels_map = dict(zip(answers, range(len(answers))))
    num_labels = len(labels_map)
    if args.num_mem_tokens is None:
        input_seg_size = args.input_size
    else:
        input_seg_size = args.input_size - args.num_mem_tokens - tokenizer.num_special_tokens_to_add()
        if 'sep_token' in tokenizer.special_tokens_map:
            input_seg_size -= 1

    if args.model_type == 'encoder-decoder':
        # raise NotImplementedError
        global_attention_first_token = False  # should be True for LED
        encode_plus_kwargs = {'truncation': True, 'padding': 'longest', 'pad_to_multiple_of': 1}
        # generate_kwargs = {'max_length': args.target_seq_len, 'min_length': args.target_seq_len}
        generate_kwargs = {}

        def collate_fn(batch):
            # cut too long strings because they may slow down tokenization
            inputs = [b['fact'] + b['input'][:args.input_seq_len * 10] for b in batch]
            questions = [b['question'] for b in batch]
            labels = [b['answer'][:args.target_seq_len * 10] for b in batch]
            if args.input_prefix:
                inputs = [args.input_prefix + inp for inp in inputs]

            total_input_size = args.max_n_segments * input_seg_size
            features = tokenizer.batch_encode_plus(list(inputs), return_tensors='pt', **encode_plus_kwargs, max_length=total_input_size)
            questions = tokenizer.batch_encode_plus(list(questions), return_tensors='pt', **encode_plus_kwargs)['input_ids']

            with tokenizer.as_target_tokenizer():
                labels = tokenizer.batch_encode_plus(list(labels), max_length=args.target_seq_len, return_tensors='pt',
                                                        **encode_plus_kwargs).input_ids
            labels[labels == tokenizer.pad_token_id] = -100
            features['labels'] = labels
            features['id'] = [b['id'] for b in batch]
            features['target_text'] = [b['answer'] for b in batch]
            if 'global_attention_mask' in features:
                raise RuntimeError('What global attention mask for Longformer and LongformerEncoder-Decoder should be?')
            return features

    elif args.model_type == 'encoder':
        if args.use_generate_on_valid:
            raise RuntimeError('use_generate_on_valid should be set to False for encoder-only models')

        encode_plus_kwargs = {
                            #   'max_length': args.input_seq_len,
                              'truncation': True,
                              'padding': 'longest',
                              'pad_to_multiple_of': 1}
        generate_kwargs = {}

        def collate_fn(batch, input_seg_size=input_seg_size):
            # cut too long strings because they may slow down tokenization
            # inputs = ['[memorize] ' + b['fact'] + ' [/memorize] ' + b['input'][:args.input_seq_len * 10] for b in batch]
            inputs = [b['fact'] + b['input'][:args.input_seq_len * 10] for b in batch]
            questions = [b['question'] for b in batch]
            labels = [b['answer'][:args.target_seq_len * 10] for b in batch]
            if args.input_prefix:
                inputs = [args.input_prefix + inp for inp in inputs]

            total_input_size = args.max_n_segments * input_seg_size
            features = tokenizer.batch_encode_plus(list(inputs), return_tensors='pt', **encode_plus_kwargs, max_length=total_input_size)
            questions = tokenizer.batch_encode_plus(list(questions), return_tensors='pt', **encode_plus_kwargs)['input_ids']
            
            q_len = questions.shape[1] - 1
            features['input_ids'] = torch.cat([features['input_ids'][:, :total_input_size - q_len], questions[:, 1:]], dim=1)
            
            labels = np.array([labels_map[t] for t in labels])
            features['labels'] = torch.from_numpy(labels)
            return features

    else:
        raise NotImplementedError('only encoder and encoder-decoder models are supported')

    
    # shuffle train data each epoch (one loop over train_dataset)
    train_sampler = DistributedSampler(train_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=True,
                                       drop_last=False, seed=args.seed)
    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    global_batch_size = per_worker_batch_size * hvd.size()
    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}
    train_dataloader = DataLoader(train_dataset, batch_size=per_worker_batch_size, sampler=train_sampler,
                                  collate_fn=collate_fn, **kwargs)
    # get validation dataset
    valid_dataloader = None
    if hvd.rank() == 0:
        logger.info(f'preparing validation data from babilong')
    # if args.task_name in tasks_with_duplicates:
    #     valid_dataset = drop_duplicates_in_input(valid_dataset)
    valid_sampler = DistributedSampler(valid_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=per_worker_batch_size, sampler=valid_sampler,
                                  collate_fn=collate_fn, **kwargs)
    if args.valid_interval is None:
        args.valid_interval = args.log_interval

    # define model
    model_cls = get_cls_by_name(args.backbone_cls)
    if hvd.rank() == 0:
        logger.info(f'Using model class: {model_cls}')
    if not args.from_pretrained:
        model_cfg = AutoConfig.from_pretrained(args.model_cfg)
        if args.model_type == 'encoder':
            model_cfg.num_labels = num_labels
        model = model_cls(config=model_cfg)
    else:
        if hvd.rank() == 0:
            logger.info(f'Loading pretrained model: {args.from_pretrained}')
        if args.model_type == 'encoder-decoder':
            model = model_cls.from_pretrained(args.from_pretrained)
        elif args.model_type == 'encoder':
            model = model_cls.from_pretrained(args.from_pretrained, num_labels=num_labels)

    # Aydar # Pass memory settings to pretrained model
    if args.num_mem_tokens is not None:
        if args.memory_forward_func is not None:
            args.memory_forward_func = get_cls_by_name(args.memory_forward_func)
        #     implementation_path = os.path.dirname(args.memory_forward_implementation)
        #     print(f'Loooking for memory_forward in {implementation_path}')
        #     sys.path.append(implementation_path)
        #     from memory_forward import memory_forward
        # else:
        #     memory_forward = None

        rmt_config = {
            'num_mem_tokens': args.num_mem_tokens, 
            'max_n_segments': args.max_n_segments,
            # 'segment_ordering': args.segment_ordering,
            'input_size': args.input_size,
            'bptt_depth': args.bptt_depth, 
            'sum_loss': args.sum_loss,
            'tokenizer': tokenizer,
            'memory_forward_func': args.memory_forward_func,
            'memory_layers': args.memory_layers,
            'share_memory_layers': args.share_memory_layers,
            'reconstruction_loss_coef': args.reconstruction_loss_coef,
        }
        rmt_cls = get_cls_by_name(args.model_cls)
        if hvd.rank() == 0:
            logger.info(f'Wrapping in: {rmt_cls}')
        
        ## load cpt
        if args.model_cpt:
            model_cpt = os.path.join(args.model_cpt, "model_best.pth")
            cpt = torch.load(model_cpt, map_location='cpu')
            # model.load_state_dict(cpt['model_state_dict'])
            drop_keys = { "cls_token", "sep_token", "mem_token_ids", "embeddings.weight"}
            fixed_state_dict = {}
            for key, value in cpt['model_state_dict'].items():
                if 'model' in key:
                    key = key.split('model.')[1]
                if key not in drop_keys:
                    fixed_state_dict[key] = value
            if hvd.rank() == 0:
                logger.info(f'Loaded state dict from: {args.model_cpt}')
        
        model = rmt_cls(model, **rmt_config)
    
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
        if args.model_type == 'encoder':
            
            ##### booydar
            data['labels'] = batch['labels']
            for key in batch.keys():
                if 'loss' in key: 
                    data[key] = batch[key]
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
        y, p = None, None
        if args.model_type == 'encoder-decoder' and 'generation_outputs' in data:
            # replace -100 with pad token in labels
            y = data['labels']
            p = tokenizer.batch_decode(data['generation_outputs'], skip_special_tokens=True)
            if hvd.rank() == 0 and args.show_valid_examples > 0:
                for i in range(min(args.show_valid_examples, len(y))):
                    logger.info(f'y: {y[i]}')
                    logger.info(f'p: {p[i]}')
                    logger.info(f'p ids: {data["generation_outputs"][i]}')
                    logger.info('-' * 50)
            # todo: do we need to better clean P to remove tokens after eos? not remove special tokens only
        elif args.model_type == 'encoder':
            y, p = data['labels'], data['predictions']

        if y is not None and p is not None:
            if args.model_type == 'encoder-decoder':
                metrics['exact_match'] = np.mean([pred == label for pred, label in zip(y, p)]) * 100
                # if not isinstance(y[0], list):
                #     y = [[_y] for _y in y]
                # result = scrolls_metric.compute(predictions=p, references=y)
                # for metric_name in task_to_metric[args.task_name]:
                    # metrics[metric_name] = result[metric_name]
            elif args.model_type == 'encoder':
                metrics['exact_match'] = accuracy_score(y, p) * 100
                metrics['f1_micro'] = f1_score(y, p, average='micro')
        return metrics

    ### booydar
    batch_metrics_fn = lambda _, y: {key: y[key] for key in y.keys() if (('loss' in key) or ('!log' in key))}
    trainer = Trainer(args, model, optimizer, train_dataloader, valid_dataloader, train_sampler,
                      keep_for_metrics_fn=keep_for_metrics_fn, metrics_fn=metrics_fn,
                      ###booydar
                      batch_metrics_fn=batch_metrics_fn,
                      generate_kwargs=generate_kwargs if args.use_generate_on_valid else {})

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
        trainer.validate(train_dataloader, split='train', write_tb=False)
        if valid_dataloader is not None:
            if hvd.rank() == 0:
                logger.info('Running validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=False)
