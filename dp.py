import os
from typing import List, Tuple, Optional, Union, Dict
from pathlib import Path
import json

from overrides import overrides

import t5
from t5.data.tasks import TaskRegistry  # noqa: F401 TaskRegistry should be imported before any usage of tasks
from t5.data.mixtures import MixtureRegistry  # noqa: F401 the same with Mixtures
from t5.evaluation.metrics import f1_score_with_invalid as t5_f1_score_with_invalid
from t5.evaluation.metrics import bleu as t5_bleu

import tensorflow.compat.v1 as tf
import torch
import horovod.torch as hvd

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.models.component import Component
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.models.torch_model import TorchModel
from deeppavlov.core.common.metrics_registry import register_metric

from lm_experiments_tools.utils import get_optimizer, get_cls_by_name, get_git_hash_commit

import logging

# works with:
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
# do not work with
# from deeppavlov.core.common.log import init_logger
# init_logger()
# log = logging.getLogger('deeppavlov')
log = logging.getLogger(__name__)

hvd.init()
if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[hvd.local_rank()]

torch.set_num_threads(4)
tf.config.set_visible_devices([], 'GPU')

import transformers  # noqa: E402
# # list of official t5 models
from transformers.models.t5 import T5_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: E402
from transformers.data.processors.utils import InputFeatures  # noqa: E402
from transformers import AutoTokenizer, T5Tokenizer, T5Config  # noqa: E402


class T5DatasetReader(DatasetReader):

    def read(self, data_path: str, name: Optional[str] = None, train: str = 'train', valid: Optional[str] = None,
             test: Optional[str] = None, train_task: Optional[str] = None, shuffle=False, seed=None, **kwargs):
        split_mapping = {'train': train, 'valid': valid, 'test': test}
        # filter unused splits
        split_mapping = {el: split_mapping[el] for el in split_mapping if split_mapping[el]}
        t5task = t5.data.get_mixture_or_task(name)
        data = {}

        def _get_dataset(task, split, tfds_split):
            shuffle_split = shuffle if split == 'train' else False
            shard_info = None
            if hvd.size() > 1:
                # do we really need sharding if train set is shuffled? - only if we care about real epochs
                # not all samples might be used in case of multi-gpu validation (max batch_size samples might get lost)
                log.info(f'Using sharded {split} set with hvd.rank: {hvd.rank()} and hvd.size: {hvd.size()}')
                shard_info = t5.seqio.ShardInfo(index=hvd.rank(), num_shards=hvd.size())
            if 'copy_pretokenized' in t5task.get_dataset.__code__.co_varnames:
                return task.get_dataset(split=tfds_split, sequence_length=None, copy_pretokenized=True,
                                        shuffle=shuffle_split, seed=seed, shard_info=shard_info)
            return task.get_dataset(split=tfds_split, sequence_length=None, shuffle=shuffle_split, seed=seed,
                                    shard_info=shard_info)

        if train_task is not None:
            t5_train_task = t5.data.get_mixture_or_task(train_task)
            data['train'] = _get_dataset(t5_train_task, 'train', split_mapping['train'])
            del split_mapping['train']

        data = dict(**data, **{k: _get_dataset(t5task, k, v) for k, v in split_mapping.items()})
        return data


class T5DatasetIterator(DataLearningIterator):
    def preprocess(self, data, *args, **kwargs):
        return [(x['inputs_pretokenized'].numpy().decode('utf8'), x['targets_pretokenized'].numpy().decode('utf8')) for x in data]


class T5TFDatasetIterator(DataLearningIterator):

    def __init__(self, data, **kwargs):
        self.data = data

    def _preprocess(self, x):
        return x['inputs_pretokenized'].numpy().decode('utf8'), x['targets_pretokenized'].numpy().decode('utf8')

    def gen_batches(self, batch_size: int, data_type: str = 'train', **kwargs):
        i = 0
        batch_x, batch_y = (), ()
        # hm, islice is too slow for getting batches from tf.Dataset
        for sample in self.data[data_type]:
            x, y = self._preprocess(sample)
            batch_x += (x,)
            batch_y += (y,)
            i += 1
            if i == batch_size:
                yield batch_x, batch_y
                i = 0
                batch_x, batch_y = (), ()

        # yield remainders
        if len(batch_x) != 0:
            yield batch_x, batch_y

    def get_instances(self, data_type: str = 'train'):
        raise NotImplementedError


class TorchTransformersPreprocessor(Component):
    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = True,
                 max_seq_length: int = 512,
                 return_tokens: bool = False,
                 reduce_pad: bool = False,
                 truncation: str = 'longest_first',
                 **kwargs) -> None:
        self.max_seq_length = max_seq_length
        self.return_tokens = return_tokens
        self.reduce_pad = reduce_pad
        self.padding_strategy = 'longest' if reduce_pad else 'max_length'
        self.truncation = truncation

        if Path(vocab_file).is_file():
            vocab_file = str(expand_path(vocab_file))
            self.tokenizer = AutoTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)

    def __call__(self, texts_a: List[str], texts_b: Optional[List[str]] = None) -> Union[
            List[InputFeatures], Tuple[List[InputFeatures], List[List[str]]]]:

        if texts_b is None:
            batch_text = texts_a
        else:
            batch_text = zip(texts_a, texts_b)
        batch_text = list(batch_text)

        # use batch encode plus and reduce paddings
        batch = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=batch_text, add_special_tokens=True,
                                                 max_length=self.max_seq_length, padding=self.padding_strategy,
                                                 truncation=self.truncation,
                                                 return_attention_mask=True, return_tensors='pt')

        if 'token_type_ids' not in batch:
            batch['token_type_ids'] = torch.zeros_like(batch['input_ids'])

        input_features = []
        tokens = []
        for encoded_dict in [dict(zip(batch.keys(), el)) for el in zip(*[torch.unbind(v, dim=0) for v in batch.values()])]:
            curr_features = InputFeatures(input_ids=encoded_dict['input_ids'],
                                          attention_mask=encoded_dict['attention_mask'],
                                          token_type_ids=encoded_dict['token_type_ids'],
                                          label=None)
            input_features.append(curr_features)
            if self.return_tokens:
                tokens.append(self.tokenizer.convert_ids_to_tokens(encoded_dict['input_ids'][0]))

        if self.return_tokens:
            return input_features, tokens
        else:
            return input_features


class T5Text2TextModel(TorchModel):
    def __init__(self, pretrained_model,
                 optimizer: str = 'AdamW',
                 optimizer_parameters: dict = {"lr": 1e-3, "weight_decay": 0.01, "betas": (0.9, 0.999), "eps": 1e-6},
                 t5_configs_path: Optional[str] = None,
                 checkpoint: Optional[str] = None,
                 clip_norm: Optional[float] = None,
                 check_commit: bool = True,
                 max_generation_len: int = 128,
                 beam_size: int = 0,
                 length_penalty: float = 0.4,
                 sub_batch_size: Optional[int] = None,
                 **kwargs):
        self.pretrained_model = pretrained_model
        self.t5_configs_path = t5_configs_path
        self.checkpoint = checkpoint
        self.check_commit = check_commit
        self.max_generation_len = max_generation_len
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.clip_norm = clip_norm
        self.sub_batch_size = sub_batch_size

        # super().__init__ calls self.load()
        super().__init__(optimizer=optimizer,
                         optimizer_parameters=optimizer_parameters,
                         **kwargs)

        if self.lr_scheduler_name:
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_name)(
                self.optimizer, **self.lr_scheduler_parameters)

        if hvd.size() > 1:
            # todo: mb remove if hvd.size() > 1 conds
            log.info('hvd: broadcasting parameters')
            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
            log.info('hvd: broadcasting optimizer parameters')
            hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
            if self.sub_batch_size is not None:
                raise RuntimeError('hvd and self.sub_batch_size != None are not supported')

    def load(self, fname=None):
        if fname is not None:
            self.load_path = fname

        # need to support:
        # * loading of default model from huggingface
        # * custom model from experiments (with custom configs and model implementations)
        if self.pretrained_model in T5_PRETRAINED_MODEL_ARCHIVE_LIST:
            # load default models from HF
            from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
            self.model = T5ForConditionalGeneration.from_pretrained(self.pretrained_model)
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        elif Path(self.pretrained_model).is_file():
            # load model from HF Transformers configuration file, with random weights
            raise NotImplementedError
        elif Path(self.pretrained_model).is_dir():
            # model from experiments - one folder one model with configuration file and possible multiple checkpoints
            self.model, self.tokenizer = load_experiment(self.pretrained_model, t5_configs_path=self.t5_configs_path,
                                                         checkpoint=self.checkpoint, check_commit=self.check_commit)
        else:
            raise RuntimeError("Could not get model to be loaded.")

        self.model.to(self.device)

        optimizer_cls = get_optimizer(self.optimizer_name)
        if optimizer_cls is None:
            raise RuntimeError(f'Optimizer {self.optimizer_name} was not found')
        log.info(f'Using optimizer {optimizer_cls}')
        self.optimizer = optimizer_cls(self.model.parameters(), **self.optimizer_parameters)

        if self.lr_scheduler_name is not None:
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_name)(
                self.optimizer, **self.lr_scheduler_parameters)

        if self.load_path:
            log.info(f"Load path {self.load_path} is given.")
            weights_path = Path(self.load_path.resolve())
            weights_path = weights_path.with_suffix(".pth.tar")
            if weights_path.exists():
                log.info(f"Load path {weights_path} exists.")
                log.info(f"Initializing `{self.__class__.__name__}` from saved.")

                # now load the weights, optimizer from saved
                log.info(f"Loading weights from {weights_path}.")
                checkpoint = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.epochs_done = checkpoint.get("epochs_done", 0)
            else:
                log.info(f"Initilized with specified pretrained_model. Load path {weights_path} does not exist.")

        if hvd.size() > 1:
            # all workers load model parameters and optimizer state from disk, no broadcasting needed
            log.info('hvd: creating DistributedOptimizer in load')
            self.optimizer = hvd.DistributedOptimizer(self.optimizer,
                                                      named_parameters=self.model.named_parameters(),
                                                      op=hvd.Average,
                                                      gradient_predivide_factor=1.0,
                                                      backward_passes_per_step=1
                                                      )

    def _build_input(self, features: List[InputFeatures]):
        _input = {}
        for elem in ['input_ids', 'attention_mask']:
            _input[elem] = [getattr(f, elem) for f in features]
            _input[elem] = torch.stack(_input[elem], dim=0).to(self.device)
        return _input

    def _get_learning_rates(self):
        learning_rates = {}
        for p in ['lr', 'scaled_lr']:
            for j, param_group in enumerate(self.optimizer.param_groups):
                if p in param_group and param_group[p] is not None:
                    learning_rates[f'{p}/param_group_{j}'] = param_group[p]
        return learning_rates

    def train_on_batch(self, features: List[InputFeatures], labels: List[InputFeatures]) -> Dict:
        input_x = self._build_input(features)
        input_y = self._build_input(labels)
        batch_size = len(input_x['input_ids'])

        input_y['input_ids'] -= (1 - input_y['attention_mask']) * 100

        self.optimizer.zero_grad()
        # todo: refactor sub-batches are used in __call__ and train_on_batch
        # todo: full batch goes to gpu, mb only sub-batch?
        sub_batch_size = self.sub_batch_size
        if sub_batch_size is None:
            sub_batch_size = batch_size

        batch_loss = 0
        n_gradient_acc_steps = max(1, batch_size // sub_batch_size)
        for i in range(0, batch_size, sub_batch_size):
            outputs = self.model(input_ids=input_x['input_ids'][i: i + sub_batch_size],
                                 attention_mask=input_x['attention_mask'][i: i + sub_batch_size],
                                 labels=input_y['input_ids'][i: i + sub_batch_size],
                                 decoder_attention_mask=input_y['attention_mask'][i: i + sub_batch_size])
            loss = outputs.loss / n_gradient_acc_steps
            batch_loss += loss.detach().item()
            loss.backward()

        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {**{'loss': batch_loss}, **self._get_learning_rates()}

    def __call__(self, features: List[InputFeatures]) -> List[str]:
        _input = self._build_input(features)
        batch_size = len(_input['input_ids'])

        sub_batch_size = self.sub_batch_size
        if sub_batch_size is None:
            sub_batch_size = batch_size

        predicted_tokens = []
        with torch.no_grad():
            for i in range(0, batch_size, sub_batch_size):
                batch_input = {k: _input[k][i: i + sub_batch_size] for k in _input}
                if self.beam_size == 0:
                    p_batch_tokens = self.model.generate(**batch_input, max_length=self.max_generation_len)
                else:
                    p_batch_tokens = self.model.generate(**batch_input, max_length=self.max_generation_len,
                                                         num_beams=self.beam_size, length_penalty=self.length_penalty)
                p_batch_tokens = p_batch_tokens.cpu().numpy().tolist()
                predicted_tokens += p_batch_tokens

        # warning: conversion from indices to tokens should be done we the same vocabulary as in pipeline
        # (currently we use only HFT tokenizer)
        # but we might use post-processor from t5tasks in next pipeline step
        # or just self.tokenizer.decode(tokens, skip_special_tokens=True)?
        predictions = [self.tokenizer.decode(tokens).replace('<pad>', '').replace('</s>', '').strip()
                       for tokens in predicted_tokens]

        return predictions

    @overrides
    def process_event(self, event_name: str, data: dict) -> None:
        """Process event. After epoch, increase `self.epochs_done`. After validation, decrease learning rate in
            `self.learning_rate_drop_div` times (not lower than `self.min_learning_rate`)
            if given `self.learning_rate_drop_patience`.

        Args:
            event_name: whether event is send after epoch or batch.
                    Set of values: ``"after_epoch", "after_batch"``
            data: event data (dictionary)
        Returns:
            None
        """
        if event_name == "after_epoch":
            self.epochs_done += 1

        if event_name == "after_validation" and 'impatience' in data and self.learning_rate_drop_patience:
            if data['impatience'] == self.learning_rate_drop_patience:
                log.info(f"----------Current LR is decreased in {self.learning_rate_drop_div} times----------")
                if self.load_before_drop:
                    self.load(self.save_path)
                    self.model.eval()
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] / self.learning_rate_drop_div, self.min_learning_rate)

    @overrides
    def save(self, fname: Optional[Union[str, Path]] = None, *args, **kwargs) -> None:
        """Save torch model to `fname` (if `fname` is not given, use `self.save_path`). Checkpoint includes
            `model_state_dict`, `optimizer_state_dict`, and `epochs_done` (number of training epochs).

        Args:
            fname:
            *args:
            **kwargs:

        Returns:

        """
        # todo: hvd.rank() == 0 move to trainer logic?
        # model.save() is called for save_every_n_batches
        if hvd.rank() == 0:
            if fname is None:
                fname = self.save_path

            fname = Path(fname)
            if not fname.parent.is_dir():
                raise ConfigError("Provided save path is incorrect!")

            weights_path = fname.with_suffix(".pth.tar")
            log.info(f"Saving model to {weights_path}.")
            # move the model to `cpu` before saving to provide consistency
            torch.save({
                "model_state_dict": self.model.cpu().state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epochs_done": self.epochs_done
            }, weights_path)
            # return it back to device (necessary if it was on `cuda`)
            self.model.to(self.device)


class T5Text2TextPostprocessor(Component):
    def __init__(self, task: str, **kwargs):
        self.postprocess_fn = t5.data.get_mixture_or_task(task).postprocess_fn

    def __call__(self, predictions: List[str]):
        return [self.postprocess_fn(p) for p in predictions]


@register_metric('f1_score_with_invalid')
def f1_score_with_invalid(y_true, y_predicted) -> float:
    # used by qqp, mrpc
    return t5_f1_score_with_invalid(y_true, y_predicted)['f1'] / 100.0


@register_metric('t5_bleu')
def bleu(y_true, y_predicted) -> float:
    return t5_bleu(y_true, y_predicted)['bleu']


def load_experiment(path, t5_configs_path, checkpoint=None, check_commit=True):
    path = Path(path)
    cfg = json.load((path / 'config.json').open('r'))
    model_cfg = Path(t5_configs_path) / cfg['model_cfg'] if cfg['model_cfg'] is not None else None
    model_cls = get_cls_by_name(cfg['model_cls'])
    if check_commit:
        assert cfg['COMMIT'] == get_git_hash_commit(), f"expected commit {cfg['COMMIT']}, " \
                                                       f"but current is {get_git_hash_commit()}"
    # take latest checkpoint
    if checkpoint is None:
        checkpoint = list(sorted(path.glob('*.pth'), key=lambda x: x.stat().st_ctime))[-1]

    if model_cfg is None:
        t5config = T5Config.from_pretrained(cfg['base_model'])
    else:
        t5config = T5Config.from_json_file(model_cfg)

    t5tokenizer = T5Tokenizer.from_pretrained(cfg['base_model'])

    model = model_cls(config=t5config)

    state_dict = torch.load(str(checkpoint), map_location='cpu')
    model.load_state_dict(state_dict["model_state_dict"])
    print(f'Model was loaded from: {checkpoint}')
    model.eval()
    return model, t5tokenizer
