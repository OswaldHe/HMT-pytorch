import datasets
from .tokenize import tokenize_dataset
from .grouping import group_dataset

def load_redpajama(tokenizer, history_size, block_size, split='train[:75%]', **kwargs):
    assert type(split) == str, "split should be a string, the format of the list of splits is not supported"
    ds = datasets.load_dataset(path='togethercomputer/RedPajama-Data-V2', name=kwargs.get('name', 'sample'), split=split, streaming=kwargs.get('streaming', False), trust_remote_code=True)
    ds = tokenize_dataset(ds, tokenizer=tokenizer, is_qa_task=False, text_column_name='raw_content', **kwargs)
    ds = group_dataset(ds, split=split, history_size=history_size, block_size=block_size, is_qa_task=False)
    return ds
