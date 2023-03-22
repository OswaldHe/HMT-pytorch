import json
import logging
import os
import re
import string
from pathlib import Path

import horovod.torch as hvd
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

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

import transformers  # noqa: E402
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser  # noqa: E402

from lm_experiments_tools.utils import prepare_run, get_cls_by_name, get_optimizer  # noqa: E402
import lm_experiments_tools.optimizers as optimizers  # noqa: E402

torch.set_num_threads(2)
# all gpus set with CUDA_VISIBLE_DEVICES are visible to process, indexing from 0 to ...
torch.cuda.set_device(hvd.local_rank())

parser = HfArgumentParser(TrainerArgs)
parser.add_argument('--data_path', type=str, help='path the training data, could be a folder')
parser.add_argument('--valid_data_path', type=str, help='path the valid data, could be a folder')
parser.add_argument('--test_data_path', type=str, help='path the test data, could be a folder')
parser.add_argument('--validate_only', action='store_true', default=False,
                    help='Skip training and run only validation. (default: False)')
parser.add_argument('--working_dir', type=str, default='.',
                    help='working dir, should be a dir with t5-experiments repo (default: .)')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--show_valid_examples', type=int, default=0,
                    help='how many valid examples to show during training (default: 0)')

parser.add_argument('--input_seq_len', type=int, default=128, help='input sequnce length (default: 128).')
parser.add_argument('--target_seq_len', type=int, default=16, help='input sequnce length (default: 16).')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')

parser.add_argument('--source_prefix', type=str, default='', help='add task prefix to a source string (default: "")')

# model args
parser.add_argument('--from_pretrained', type=str, help='model name in HF Model Hub (default: "")')
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: "")')
parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
                    help='model class name to use (default: transformers:BertForPreTraining)')
parser.add_argument('--model_type', type=str, default='encoder',
                    help='model type, encoder, encoder-decoder, decoder, affects preprocessing (default: encoder)')

# tokenizer
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


class HyperpartisanDataset(Dataset):
    def __init__(self, datafile, x_field='text', label_field='label'):
        if isinstance(datafile, str):
            datafile = Path(datafile)
        self.data = []
        for line in datafile.open('r'):
            self.data += [json.loads(line)]
        self.x_field = x_field
        self.label_field = label_field

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][self.x_field]
        label = self.data[idx][self.label_field]
        return x, label


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def main():
    args = parser.parse_args()
    # set current working dir
    args.working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(args.working_dir)

    prepare_run(args, logger, logger_fmt)

    if hvd.rank() == 0:
        logger.info(f'hvd size: {hvd.size()}')
        logger.info(f'FP16: {args.fp16}')
        logger.info(f'working dir: {args.working_dir}')

    if hvd.rank() == 0 and args.model_path is None:
        logger.warning('model_path is not set: config, logs and checkpoints will not be saved.')

    if not args.from_pretrained:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)

    labels_map = {'false': 0, 'true': 1}
    # collate_fn depends on model type (encoder, encoder-decoder)
    # collate_fn defines how data is prepared for the model
    if args.model_type == 'encoder':
        num_labels = len(labels_map)
        encode_plus_kwargs = {'max_length': args.input_seq_len,
                              'truncation': True,
                              'padding': 'longest',
                              'pad_to_multiple_of': 64}

        def collate_fn(batch):
            inputs, labels = zip(*batch)
            features = tokenizer.batch_encode_plus(list(inputs), return_tensors='pt', **encode_plus_kwargs)
            # map labels to ids
            labels = np.array([labels_map[t] for t in labels])
            labels = {'labels': torch.from_numpy(labels)}
            return {**features, **labels}
    elif args.model_type == 'encoder-decoder':
        # global_attention_first_token = False  # should be True for LED
        num_labels = 0
        encode_plus_kwargs = {'truncation': True,
                              'padding': 'longest',
                              'pad_to_multiple_of': 1}
        # generate predictions to fixed length
        generate_kwargs = {'max_length': args.target_seq_len, 'min_length': args.target_seq_len}
        # generate predictions to max targets length in batch
        generate_kwargs = {}

        def collate_fn(batch):
            inputs, labels = zip(*batch)
            if args.source_prefix:
                inputs = [args.source_prefix + inp for inp in inputs]
            features = tokenizer.batch_encode_plus(list(inputs), max_length=args.input_seq_len,
                                                   return_tensors='pt', **encode_plus_kwargs)
            with tokenizer.as_target_tokenizer():
                labels = tokenizer.batch_encode_plus(list(labels), max_length=args.target_seq_len,
                                                     return_tensors='pt', **encode_plus_kwargs).input_ids
            labels[labels == tokenizer.pad_token_id] = -100
            features['labels'] = labels
            if 'global_attention_mask' in features:
                raise RuntimeError('What global attention mask for Longformer and LongformerEncoder-Decoder should be?')
            return features
    else:
        raise NotImplementedError('only encoder & encoder-decoder type of model is supported')

    # get train dataset
    if hvd.rank() == 0:
        logger.info(f'preparing training data from: {args.data_path}')
    data_path = Path(args.data_path).expanduser().absolute()
    train_dataset = HyperpartisanDataset(data_path)
    # shuffle train data each epoch (one loop over train_dataset)
    train_sampler = DistributedSampler(train_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=True,
                                       drop_last=False, seed=args.seed)
    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}
    train_dataloader = DataLoader(train_dataset, batch_size=per_worker_batch_size, sampler=train_sampler,
                                  collate_fn=collate_fn, **kwargs)
    # get validation dataset
    if args.valid_data_path:
        if hvd.rank() == 0:
            logger.info(f'preparing validation data from: {args.valid_data_path}')
        valid_data_path = Path(args.valid_data_path).expanduser().absolute()
        valid_dataset = HyperpartisanDataset(valid_data_path)
        valid_sampler = DistributedSampler(valid_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=False)
        valid_dataloader = DataLoader(valid_dataset, batch_size=per_worker_batch_size, sampler=valid_sampler,
                                      collate_fn=collate_fn, **kwargs)
        if args.valid_interval is None:
            args.valid_interval = args.log_interval
    else:
        valid_dataloader = None
        if hvd.rank() == 0:
            logger.info('No validation data is used.')
    # get test dataset
    if args.test_data_path:
        if hvd.rank() == 0:
            logger.info(f'preparing test data from: {args.test_data_path}')
        test_data_path = Path(args.test_data_path).expanduser().absolute()
        test_dataset = HyperpartisanDataset(test_data_path)
        test_sampler = DistributedSampler(test_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=per_worker_batch_size, sampler=test_sampler,
                                     collate_fn=collate_fn, **kwargs)

    # define model
    model_cls = get_cls_by_name(args.model_cls)
    if hvd.rank() == 0:
        logger.info(f'Using model class: {model_cls}')
    if not args.from_pretrained:
        model_cfg = AutoConfig.from_pretrained(args.model_cfg)
        model_cfg.num_labels = num_labels
        model = model_cls(config=model_cfg)
    else:
        if hvd.rank() == 0:
            logger.info(f'Loading pretrained model: {args.from_pretrained}')
        model = model_cls.from_pretrained(args.from_pretrained, num_labels=num_labels)

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
        if args.model_type == 'encoder':
            data['labels'] = batch['labels']
            data['predictions'] = torch.argmax(output['logits'].detach(), dim=-1)
        elif args.model_type == 'encoder-decoder' and 'generation_outputs' in output:
            # logger.info(f'{output["generation_outputs"].shape}')
            data['labels'] = batch['labels']
            data['generation_outputs'] = output['generation_outputs']
        return data

    def metrics_fn(data):
        # compute metrics based on stored labels, predictions, ...
        metrics = {}
        y, p = None, None
        if args.model_type == 'encoder':
            y, p = data['labels'], data['predictions']
        elif args.model_type == 'encoder-decoder' and 'generation_outputs' in data:
            y = tokenizer.batch_decode(data['labels'], skip_special_tokens=True)
            p = tokenizer.batch_decode(data['generation_outputs'], skip_special_tokens=True)
            if hvd.rank() == 0 and args.show_valid_examples > 0:
                for i in range(min(args.show_valid_examples, len(y))):
                    logger.info(f'y: {y[i]}')
                    logger.info(f'p: {p[i]}')
                    logger.info(f'p token_ids: {data["generation_outputs"][i]}')
                    logger.info('-' * 50)
            # map to labels
            y = [labels_map.get(normalize_answer(_y), 0) for _y in y]
            p = [labels_map.get(normalize_answer(_p), 0) for _p in p]
        if y is not None and p is not None:
            # accuracy, f1, precision, recall
            metrics['accuracy'] = accuracy_score(y, p)
            metrics['f1'] = f1_score(y, p)
            metrics['precision'] = precision_score(y, p)
            metrics['recall'] = recall_score(y, p)
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
        if args.valid_data_path:
            if hvd.rank() == 0:
                logger.info('Runnning validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=False)
        if args.test_data_path:
            if hvd.rank() == 0:
                logger.info('Runnning validation on test data:')
            trainer.validate(test_dataloader, split='test', write_tb=True)
    else:
        # run validation, do not write to tensorboard
        if hvd.rank() == 0:
            logger.info('Running validation on train set:')
        trainer.validate(train_dataloader, write_tb=False)
        if args.valid_data_path:
            if hvd.rank() == 0:
                logger.info('Running validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=False)
        if args.test_data_path:
            if hvd.rank() == 0:
                logger.info('Running validation on test data:')
            trainer.validate(test_dataloader, split='test', write_tb=False)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(e)
