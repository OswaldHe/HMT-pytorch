import datasets
from datasets import load_dataset, Dataset
from typing import Union, List
import os

from .tokenize import filter_by_length, tokenize_dataset, tokenize_column
from .grouping import group_dataset
from .io import read_jsonl
from .prep_funcs import prepare_qmsum_train

HF_HOME = os.getenv('HF_HOME', None)
HMT_PYTORCH_PATH = os.getenv('HMT_PYTORCH_PATH', None)



def load_qmsum_train_dataset(path, **kwargs):
    """Load the QMSum train dataset. 

    :param path: Path to train.jsonl that is generated after running the processing script in the QMSum repository.
    :type path: str
    """
    data = read_jsonl(path)
    return data


def load_qmsum_train(max_token_num, block_size, tokenizer, path=f"{HMT_PYTORCH_PATH}/data/qmsum/train.jsonl", source=None, **kwargs):
    if source == 'huggingface':
        ds = load_dataset('ioeddk/qmsum', split='train', cache_dir=HF_HOME, streaming=kwargs.get('streaming', False))
    else:
        ds = load_qmsum_train_dataset(path=path)
        ds = prepare_qmsum_train(ds)

    # Tokenize the dataset
    ds = tokenize_dataset(ds, tokenizer=tokenizer, is_qa_task=True, text_column_name='text', **kwargs)

    # Tokenize the answer dataset, determine the token amount in each answer, and set the mask size. 
    answer_ids = tokenize_column(ds, tokenizer, column_name='answer')
    ds = ds.add_column("mask_size", [len(answer_id) for answer_id in answer_ids])

    # Remove unnecessary columns
    column_to_remove = [name for name in ds.column_names if name not in {'input_ids', 'attention_mask', 'labels', 'mask_size'}]
    ds = ds.remove_columns(column_to_remove)

    ds = filter_by_length(ds, max_token_num=max_token_num, tokens_column_name='input_ids')
    ds = group_dataset(ds, split='train', history_size=None, block_size=block_size, is_qa_task=True)
    return ds

