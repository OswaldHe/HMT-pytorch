from typing import List, Tuple, Optional, Union, Dict
from pathlib import Path


import t5
from t5.data.tasks import TaskRegistry  # noqa: F401 TaskRegistry should be imported before any usage of tasks
from t5.data.mixtures import MixtureRegistry  # noqa: F401 the same with Mixtures
from t5.evaluation.metrics import f1_score_with_invalid as t5_f1_score_with_invalid

import tensorflow.compat.v1 as tf
import torch

from transformers.data.processors.utils import InputFeatures
from transformers import AutoTokenizer
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.models.component import Component
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.models.torch_model import TorchModel
from deeppavlov.core.common.metrics_registry import register_metric

# list of official t5 models
from transformers.models.t5 import T5_PRETRAINED_MODEL_ARCHIVE_LIST

import logging

# works with:
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
# do not work with
# from deeppavlov.core.common.log import init_logger
# init_logger()

# log = logging.getLogger('deeppavlov')
log = logging.getLogger(__name__)

tf.config.set_visible_devices([], 'GPU')


class T5DatasetReader(DatasetReader):

    def read(self, data_path: str, name: Optional[str] = None, train: str = 'train', valid: Optional[str] = None,
             test: Optional[str] = None, train_task: Optional[str] = None, shuffle=False, **kwargs):
        split_mapping = {'train': train, 'valid': valid, 'test': test}
        # filter unused splits
        split_mapping = {el: split_mapping[el] for el in split_mapping if split_mapping[el]}
        t5task = t5.data.get_mixture_or_task(name)
        data = {}

        def _get_dataset(task, split):
            if 'copy_pretokenized' in t5task.get_dataset.__code__.co_varnames:
                return task.get_dataset(split=split, sequence_length=None, shuffle=shuffle, copy_pretokenized=True)
            return task.get_dataset(split=split, sequence_length=None, shuffle=shuffle)

        if train_task is not None:
            t5_train_task = t5.data.get_mixture_or_task(train_task)
            data['train'] = _get_dataset(t5_train_task, split_mapping['train'])
            del split_mapping['train']

        data = dict(**data, **{k: _get_dataset(t5task, v) for k, v in split_mapping.items()})
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
                 sub_batch_size: Optional[int] = None,
                 **kwargs):
        self.pretrained_model = pretrained_model
        self.t5_configs_path = t5_configs_path
        self.checkpoint = checkpoint
        self.check_commit = check_commit
        self.max_generation_len = max_generation_len
        self.clip_norm = clip_norm
        self.sub_batch_size = sub_batch_size
        # super().__init__ calls self.load()
        super().__init__(optimizer=optimizer,
                         optimizer_parameters=optimizer_parameters,
                         **kwargs)

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
            from utils import load_experiment
            self.model, self.tokenizer = load_experiment(self.pretrained_model, t5_configs_path=self.t5_configs_path,
                                                         checkpoint=self.checkpoint, check_commit=self.check_commit)
        else:
            raise RuntimeError("Could not get model to be loaded.")

        self.model.to(self.device)

        self.optimizer = getattr(torch.optim, self.optimizer_name)(
            self.model.parameters(), **self.optimizer_parameters)
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

    def _build_input(self, features: List[InputFeatures]):
        _input = {}
        for elem in ['input_ids', 'attention_mask']:
            _input[elem] = [getattr(f, elem) for f in features]
            _input[elem] = torch.stack(_input[elem], dim=0).to(self.device)
        return _input

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

        return {'loss': batch_loss}

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
                p_batch_tokens = self.model.generate(**batch_input, max_length=self.max_generation_len)
                p_batch_tokens = p_batch_tokens.cpu().numpy().tolist()
                predicted_tokens += p_batch_tokens

        # warning: conversion from indices to tokens should be done we the same vocabulary as in pipeline
        # (currently we use only HFT tokenizer)
        # but we might use post-processor from t5tasks in next pipeline step
        # or just self.tokenizer.decode(tokens, skip_special_tokens=True)?
        predictions = [self.tokenizer.decode(tokens).replace('<pad>', '').replace('</s>', '').strip()
                       for tokens in predicted_tokens]

        return predictions


class T5Text2TextPostprocessor(Component):
    def __init__(self, task: str, **kwargs):
        self.postprocess_fn = t5.data.get_mixture_or_task(task).postprocess_fn

    def __call__(self, predictions: List[str]):
        return [self.postprocess_fn(p) for p in predictions]


@register_metric('f1_score_with_invalid')
def f1_score_with_invalid(y_true, y_predicted) -> float:
    # used by qqp, mrpc
    return t5_f1_score_with_invalid(y_true, y_predicted)['f1'] / 100.0
