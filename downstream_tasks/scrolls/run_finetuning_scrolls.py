import logging
import os
import shutil
from pathlib import Path

import horovod.torch as hvd
import datasets
import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, DistributedSampler

from lm_experiments_tools import Trainer, TrainerArgs

load_dotenv()

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger()

# if CUDA_VISIBLE_DEVICES is not set make all gpus visible
if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])

logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
# first call to torch.cuda.device_count() sets visible gpus, following calls will not change the result
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

hvd.init()

# import transformers  # noqa: E402, F401
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser  # noqa: E402

from lm_experiments_tools.utils import prepare_run, get_cls_by_name, get_optimizer  # noqa: E402

torch.set_num_threads(2)
# first call to torch.cuda.device_count() sets visible gpus, following calls will not change the result
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

# model args
parser.add_argument('--from_pretrained', type=str, help='model name in HF Model Hub (default: "")')
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: "")')
parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
                    help='model class name to use (default: transformers:BertForPreTraining)')
parser.add_argument('--model_type', type=str, default='encoder-decoder',
                    help='model type, encoder, encoder-decoder, decoder, affects preprocessing '
                         '(default: encoder-decoder)')

# tokenizer
parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (default: 0.0)')


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


def main():
    args = parser.parse_args()
    # set current working dir
    args.working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(args.working_dir)

    prepare_run(args, logger, logger_fmt)

    if hvd.rank() == 0:
        logger.info(f'hvd size: {hvd.size()}')
        logger.info(f'FP16: {args.fp16}')

    if hvd.rank() == 0 and args.model_path is None:
        logger.warning('model_path is not set: config, logs and checkpoints will not be saved.')

    if not args.from_pretrained:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)

    if args.model_type == 'encoder-decoder':
        # global_attention_first_token = False  # should be True for LED
        encode_plus_kwargs = {'truncation': True, 'padding': 'longest', 'pad_to_multiple_of': 1}
        # generate_kwargs = {'max_length': args.target_seq_len, 'min_length': args.target_seq_len}
        generate_kwargs = {}

        def collate_fn(batch):
            # cut too long strings because they may slow down tokenization
            inputs = [b['input'][:args.input_seq_len * 10] for b in batch]
            if 'outputs' in batch[0]:
                # if we have more than 1 label per example (only in valid) take only one of them
                # to compute loss on valid
                labels = [b['outputs'][0][:args.target_seq_len * 10] for b in batch]
            else:
                labels = [b['output'][:args.target_seq_len * 10] for b in batch]
            if args.input_prefix:
                inputs = [args.input_prefix + inp for inp in inputs]
            features = tokenizer.batch_encode_plus(list(inputs), max_length=args.input_seq_len, return_tensors='pt',
                                                   **encode_plus_kwargs)
            with tokenizer.as_target_tokenizer():
                labels = tokenizer.batch_encode_plus(list(labels), max_length=args.target_seq_len, return_tensors='pt',
                                                     **encode_plus_kwargs).input_ids
            labels[labels == tokenizer.pad_token_id] = -100
            features['labels'] = labels
            features['id'] = [b['id'] for b in batch]
            if 'outputs' in batch[0]:
                features['target_text'] = [b['outputs'] for b in batch]
            else:
                features['target_text'] = [b['output'] for b in batch]
            if 'global_attention_mask' in features:
                raise RuntimeError('What global attention mask for Longformer and LongformerEncoder-Decoder should be?')
            return features

    elif args.model_type == 'encoder' and args.task_name == 'contract_nli':
        if args.use_generate_on_valid:
            raise RuntimeError('use_generate_on_valid should be set to False for encoder-only models')

        encode_plus_kwargs = {'max_length': args.input_seq_len,
                              'truncation': True,
                              'padding': 'longest',
                              'pad_to_multiple_of': 64}
        generate_kwargs = {}
        labels_map = {'Contradiction': 0, 'Entailment': 1, 'Not mentioned': 2}
        num_labels = len(labels_map)

        def collate_fn(batch):
            # cut too long strings because they may slow down tokenization
            inputs = [b['input'][:args.input_seq_len * 10] for b in batch]
            labels = [b['output'][:args.target_seq_len * 10] for b in batch]
            if args.input_prefix:
                inputs = [args.input_prefix + inp for inp in inputs]
            features = tokenizer.batch_encode_plus(list(inputs), return_tensors='pt', **encode_plus_kwargs)
            labels = np.array([labels_map[t] for t in labels])
            features['labels'] = torch.from_numpy(labels)
            return features

    else:
        raise NotImplementedError('only encoder-decoder models are supported for scrolls datasets or '
                                  'encoder models only for contract_nli task')

    # get train dataset
    if hvd.rank() == 0:
        logger.info(f'preparing dataset for: {args.task_name}')
    dataset = datasets.load_dataset('tau/scrolls', args.task_name)
    train_dataset = dataset['train']
    # shuffle train data each epoch (one loop over train_dataset)
    train_sampler = DistributedSampler(train_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=True,
                                       drop_last=False, seed=args.seed)
    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}
    train_dataloader = DataLoader(train_dataset, batch_size=per_worker_batch_size, sampler=train_sampler,
                                  collate_fn=collate_fn, **kwargs)
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
        if args.model_type == 'encoder-decoder':
            model = model_cls.from_pretrained(args.from_pretrained)
        elif args.model_type == 'encoder' and args.task_name == 'contract_nli':
            model = model_cls.from_pretrained(args.from_pretrained, num_labels=num_labels)

    # define optimizer
    optimizer_cls = get_optimizer(args.optimizer)
    if optimizer_cls is None:
        raise RuntimeError(f'{args.optimizer} was not found in optimizers, torch.optim, transformers.optimization')

    if hvd.rank() == 0:
        logger.info(f'Using optimizer class: {optimizer_cls}')

    optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # for encoder only classification
    def keep_for_metrics_fn(batch, output):
        # select data from batch and model output that would be used to compute metrics
        data = {}
        if 'generation_outputs' in output:
            data['labels'] = batch['target_text']
            data['generation_outputs'] = output['generation_outputs']
        if args.model_type == 'encoder':
            data['labels'] = batch['labels']
            data['predictions'] = torch.argmax(output['logits'].detach(), dim=-1)
        return data

    scrolls_metric = datasets.load_metric(scrolls_metric_path, args.task_name, keep_in_memory=True)

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
            # todo: do we need to better clean P to remove tokens after eos? not only remove special tokens
        elif args.model_type == 'encoder':
            y, p = data['labels'], data['predictions']

        if y is not None and p is not None:
            if args.model_type == 'encoder-decoder':
                if not isinstance(y[0], list):
                    y = [[_y] for _y in y]
                result = scrolls_metric.compute(predictions=p, references=y)
                for metric_name in task_to_metric[args.task_name]:
                    metrics[metric_name] = result[metric_name]
            elif args.model_type == 'encoder' and args.task_name == 'contract_nli':
                metrics['exact_match'] = accuracy_score(y, p) * 100
                metrics['f1_micro'] = f1_score(y, p, average='micro')
        return metrics

    trainer = Trainer(args, model, optimizer, train_dataloader, valid_dataloader, train_sampler,
                      keep_for_metrics_fn=keep_for_metrics_fn, metrics_fn=metrics_fn,
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


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(e)
