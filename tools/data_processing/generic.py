from typing import Callable
import datasets
from datasets import Dataset

from .tokenize import tokenize_dataset, tokenize_column, filter_by_length
from .grouping import group_dataset

def prepare_test(ds: Dataset, prep_func: Callable, max_token_num, test_length, block_size, tokenizer, with_answer=False, **kwargs):
    assert prep_func is not None, "Please provide a preparation function for the dataset."

    ds = prep_func(ds)

    # Tokenize the text column
    ds = tokenize_dataset(ds, tokenizer=tokenizer, text_column_name='text', **kwargs)  # Tokenize the text column

    # Tokenize the answer dataset, determine the token amount in each answer, and set the mask size. 
    answer_ids = tokenize_column(ds, tokenizer, column_name='answer')
    ds = ds.add_column("mask_size", [len(answer_id) for answer_id in answer_ids])

    keep_columns = {'input_ids', 'attention_mask', 'labels', 'mask_size'}
    if with_answer: keep_columns.add('answer')

    # Remove unnecessary columns
    column_to_remove = [name for name in ds.column_names if name not in keep_columns]
    ds = ds.remove_columns(column_to_remove)

    # Filter the dataset by lenght
    ds = filter_by_length(ds, max_token_num=max_token_num, tokens_column_name='input_ids')
    ds = group_dataset(ds, split='test', history_size=test_length, block_size=block_size, is_qa_task=True, with_answer=with_answer)

    return ds
