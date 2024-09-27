from typing import List
import datasets

def filter_by_length(dataset: datasets.Dataset, max_token_num: int, tokens_column_name: str='text'):
    dataset = dataset.filter(lambda x: len(x[tokens_column_name]) < max_token_num)
    assert len(dataset) != 0, f"The number of samples with length < {max_token_num} is 0!"
    return dataset

def tokenization(dataset, tokenizer, text_column_name):
    return tokenizer(dataset[text_column_name])

def tokenize_column(dataset, tokenizer, column_name):
    return tokenizer(dataset[column_name])['input_ids']

def tokenize_dataset(dataset, tokenizer, text_column_name='text', **kwargs):
    tokenized_dataset = dataset.map(
        lambda x: tokenization(x, tokenizer, text_column_name=text_column_name),
        batched=True,
        batch_size=kwargs.get('batch_size', 4),
        desc=f"Running tokenizer",
        num_proc=kwargs.get('num_proc', 8)
    )
    return tokenized_dataset
