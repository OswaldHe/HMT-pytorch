import datasets
from datasets import load_dataset
from typing import List, Union
import os
from .tokenize import tokenize_dataset, tokenize_column, filter_by_length
from .grouping import group_dataset

HF_HOME = os.environ.get("HF_HOME", None)

def prepare_narrativeqa_dataset(dataset: Union[datasets.Dataset, tuple[datasets.Dataset]]):
    prompt = "Prompt: Answer the question based on the provided document."
    question_delimiter = "Question: "
    context_delimiter = "Document: "
    answer_delimiter = "Answer: "
    prepared_data = []

    if isinstance(dataset, datasets.Dataset):
        dataset = (dataset)

    def nqa_entry_to_text(dataset):
        """
        Convert a NarrativeQA dataset entry to a text entry. consists of text and labels. 
        The text is a concat of prompt, question delimiter, question, context delimiter, context and label is the answer.
        """
        return {
            "text": [f"{prompt}\n{question_delimiter}{entry[0]['text']}\n{context_delimiter}{entry[1]['text']}\n{answer_delimiter}{entry[2][0]['text']}" for entry in zip(dataset['question'], dataset['document'], dataset['answers'])],
            "answer": [entry[0]['text'] for entry in dataset['answers']]
        }
    
    prepared_data = ()
    for split in dataset:
        prepared_data = prepared_data + (split.map(nqa_entry_to_text, batched=True, 
                                                    desc=f"Concatenating QA Input Questions and Contexts",
                                                    num_proc=8).remove_columns(['question', 'document', 'answers']),)
    return prepared_data

def load_narrativeqa_train_valid(max_token_num, block_size, tokenizer, **kwargs):
    ds = load_dataset(path="deepmind/narrativeqa", split=['train', 'validation'], cache_dir=HF_HOME, streaming=kwargs.get('streaming', False))
    ds = prepare_narrativeqa_dataset(ds)
    ds_out = ()
    for split in ds:
        ds_out_split = tokenize_dataset(split, tokenizer=tokenizer, is_qa_task=True, text_column_name='text', **kwargs)
        answer_ids = tokenize_column(ds_out_split, tokenizer, column_name='answer')
        ds_out_split = ds_out_split.add_column("mask_size", [len(answer_id) for answer_id in answer_ids])
        column_to_remove = [name for name in ds_out_split.column_names if name not in {'input_ids', 'attention_mask', 'labels', 'mask_size'}]
        ds_out_split = ds_out_split.remove_columns(column_to_remove)
        ds_out_split = filter_by_length(ds_out_split, max_token_num=max_token_num, tokens_column_name='input_ids')
        ds_out_split = group_dataset(ds_out_split, split='train', history_size=None, block_size=block_size, is_qa_task=True)
        ds_out = ds_out + (ds_out_split,)
    return ds_out

def load_narrativeqa_test(max_token_num, test_length, block_size, tokenizer, with_answer=False, **kwargs):
    ds = load_dataset(path="deepmind/narrativeqa", split=kwargs.get('split', 'test'), cache_dir=HF_HOME, streaming=kwargs.get('streaming', False))
    ds = prepare_narrativeqa_dataset(ds)
    ds = tokenize_dataset(ds, tokenizer=tokenizer, is_qa_task=True, text_column_name='text', **kwargs)
    answer_ids = tokenize_column(ds, tokenizer, column_name='answer')
    ds = ds.add_column("mask_size", [len(answer_id) for answer_id in answer_ids])

    keep_columns = {'input_ids', 'attention_mask', 'labels', 'mask_size'}
    if with_answer: keep_columns.add('answer')

    column_to_remove = [name for name in ds.column_names if name not in keep_columns]
    ds = ds.remove_columns(column_to_remove)

    ds = filter_by_length(ds, max_token_num=max_token_num, tokens_column_name='input_ids')
    ds = group_dataset(ds, split='test', history_size=test_length, block_size=block_size, is_qa_task=True)
    return ds
