import datasets
from datasets import load_dataset, Dataset
from typing import Union, List
import os

from .tokenize import filter_by_length, tokenize_dataset, tokenize_column
from .grouping import group_dataset
from .io import read_jsonl

HF_HOME = os.getenv('HF_HOME', None)

"""
Load the QMSum dataset from Hugging Face.
"""
def load_qmsum_test_dataset(split: Union[List[str], str]="test", **kwargs):
    """Load QMSum test dataset from LongBench. 

    :param split: The split of the dataset to load. Defaults to "test".
    :type split: Union[List[str], str], optional
    :return: The QMSum dataset.
    :rtype: datasets.Dataset
    """
    ds = load_dataset(path="THUDM/LongBench", name="qmsum", split=split, cache_dir=HF_HOME, streaming=kwargs.get('streaming', False))
    return ds

def prepare_qmsum_test_ppl(dataset: datasets.Dataset):
    prompt = "Prompt: Answer the question based on the provided document."
    question_delimiter = "Question: "
    context_delimiter = "Document: "
    answer_delimiter = "Answer: "

    def qmsum_entry_to_text_longbench(dataset):
        """
        Convert a QMSum dataset entry to a text entry. 
        The text is a concat of prompt, question delimiter, question, context delimiter, context and label is the answer.
        """
        return {
            "text": [f"{prompt}\n{question_delimiter}{entry[0]}\n{context_delimiter}{entry[1]}\n{answer_delimiter}{entry[2][0]}" for entry in zip(dataset['input'], dataset['context'], dataset['answers'])],
            "answer": [entry[0] for entry in dataset['answers']],
        }

    dataset = dataset.map(qmsum_entry_to_text_longbench, batched=True, 
                                                    desc=f"Concatenating QA Input Questions and Contexts",
                                                    num_proc=8).remove_columns(['input', 'context', 'answers', 'length', 
                                                                                'dataset', 'language', 'all_classes', '_id'])
    return dataset


def load_qmsum_train_dataset(path, **kwargs):
    """Load the QMSum train dataset. 

    :param path: Path to train.jsonl that is generated after running the processing script in the QMSum repository.
    :type path: str
    """
    data = read_jsonl(path)
    return data

def prepare_qmsum_train(dataset: List):
    import re

    def extract_and_remove_s_tags(text):
        # Extract content within <s></s> tags
        extracted = re.findall(r'<s>(.*?)</s>', text, re.DOTALL)
        
        # Remove <s></s> tags and their content from the original text
        cleaned_text = re.sub(r'<s>.*?</s>', '', text, flags=re.DOTALL)
        
        return extracted, cleaned_text.strip()

    prompt = "Prompt: Complete the task based on the provided meeting records."

    # Process each entry in the dataset
    task_col = []
    meeting_notes_col = []
    answer_col = []
    for entry in dataset:
        extracted, cleaned_text = extract_and_remove_s_tags(entry['src'])
        
        # Combine extracted parts into a single string (if needed)
        extracted_text = ' '.join(extracted)

        task_col.append(extracted_text)
        meeting_notes_col.append(cleaned_text)
        answer_col.append(entry['tgt'])

    raw_datas = {
        'task': task_col,
        'meeting_notes': meeting_notes_col,
        'answers': answer_col
    }

    dataset = Dataset.from_dict(raw_datas)

    prompt = "Prompt: Complete the task based on the provided meeting notes."
    question_delimiter = "Task: "
    context_delimiter = "Meeting Notes: "
    answer_delimiter = "Answer: "

    def qmsum_entry_to_text(dataset):
        """
        Convert a QMSum dataset entry to a text entry. consists of text and labels. 
        The text is a concat of prompt, question delimiter, question, context delimiter, context and label is the answer.
        """
        return {
            "text": [f"{prompt}\n{question_delimiter}{entry[0]}\n{context_delimiter}{entry[1]}\n{answer_delimiter}{entry[2]}" for entry in zip(dataset['task'], dataset['meeting_notes'], dataset['answers'])],
            "answer": dataset['answers']
        }

    dataset = dataset.map(qmsum_entry_to_text, batched=True,
                                                    desc=f"Concatenating QMSum Input Questions and Contexts",
                                                    num_proc=8).remove_columns(['task', 'meeting_notes', 'answers'])

    return dataset



def load_qmsum_train(max_token_num, block_size, tokenizer, path="/home/yingqi/repo/QMSum/data/train.jsonl", **kwargs):
    ds = load_qmsum_train_dataset(path=path)
    ds = prepare_qmsum_train(ds)
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


def load_qmsum_test(max_token_num, test_length, block_size, tokenizer, split='test', with_answer=False, **kwargs):
    ds = load_qmsum_test_dataset(split=split, **kwargs)  # Load the dataset from Hugging Face
    ds = prepare_qmsum_test_ppl(ds)  # Prepare the dataset in PPL Testing format. 
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
    ds = group_dataset(ds, split='test', history_size=test_length, block_size=block_size, is_qa_task=True)
    return ds
