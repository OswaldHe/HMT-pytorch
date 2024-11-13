from datasets import load_dataset
from .tokenize import tokenize_dataset, filter_by_length, tokenize_column
from .grouping import group_dataset
from .prep_funcs import prepare_musique_train
import os

HF_HOME = os.getenv('HF_HOME', None)

def load_musique_train(max_token_num, block_size, tokenizer, **kwargs):
    ds = load_dataset("dgslibisey/MuSiQue", split=kwargs.get('split', 'train'), streaming=kwargs.get('streaming', False), cache_dir=HF_HOME)
    ds = prepare_musique_train(ds)
    ds = tokenize_dataset(ds, tokenizer=tokenizer, is_qa_task=True, text_column_name='text', **kwargs)

    answer_ids = tokenize_column(ds, tokenizer, column_name='answer')
    ds = ds.add_column("mask_size", [len(answer_id) for answer_id in answer_ids])

    column_to_remove = [name for name in ds.column_names if name not in {'input_ids', 'attention_mask', 'labels', 'mask_size'}]
    ds = ds.remove_columns(column_to_remove)

    ds = filter_by_length(ds, max_token_num=max_token_num, tokens_column_name='input_ids')
    ds = group_dataset(ds, split='train', history_size=None, block_size=block_size, is_qa_task=True)
    return ds
