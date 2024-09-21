import datasets
from datasets import load_dataset
import os
from typing import List, Union

HF_HOME = os.environ["HF_HOME"]

def load_narrativeqa_dataset(source: str="deepmind", split: Union[List[str], str]="test", **kwargs):
    """
    Load the NarrativeQA dataset from Hugging Face.
    source could be deepmind or longbench.
    """

    if isinstance(split, str):
        split = [split]

    if source == "deepmind":
        ds = load_dataset(path="deepmind/narrativeqa", split=split, cache_dir=HF_HOME, streaming=kwargs.get('streaming', False))
    elif source == "longbench":
        ds = load_dataset(path="THUDM/LongBench", name="narrativeqa", split=split, cache_dir=HF_HOME, streaming=kwargs.get('streaming', False))
    else:
        raise ValueError(f"Invalid source: {source}, please specify deepmind or longbench.")
    return ds

def prepare_narrativeqa_dataset(dataset: Union[datasets.Dataset, tuple[datasets.Dataset]], source: str="deepmind"):
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
            "answer_length": [len(entry[0]['text']) for entry in dataset['answers']]
        }
    
    prepared_data = ()
    if source == "deepmind":
        for split in dataset:
            prepared_data = prepared_data + (split.map(nqa_entry_to_text, batched=True, 
                                                       desc=f"Concatenating QA Input Questions and Contexts",
                                                       num_proc=8).remove_columns(['question', 'document', 'answers']),)
    else:
        raise NotImplementedError(f"process dataset for source {source} not implemented.")
    return prepared_data

def filter_by_length(dataset: tuple[datasets.Dataset], max_length: int=100000):
    out = ()

    for split in dataset:
        split_ = split.filter(lambda x: len(x['document']['text']) < max_length)
        assert len(split_) != 0, f"The number of samples with length < {max_length} is 0!"
        out = out + (split_,)
    return out

def load_qa_dataset(name, max_context_length: int=None, **kwargs):
    """
    Load a question answering dataset from Hugging Face.

    :param name: The name of the dataset to load.
    :type name: str
    :raises NotImplementedError: If the dataset is not implemented.
    :return: The HF dataset.
    :rtype: datasets.Dataset
    """
    if name == 'deepmind/narrativeqa':
        ds = load_narrativeqa_dataset(split=kwargs.get('split', 'test'), source='deepmind', streaming=kwargs.get('streaming', False))
        if max_context_length is not None:
            ds = filter_by_length(ds, max_context_length)
        ds = prepare_narrativeqa_dataset(ds, source='deepmind')
        if isinstance(ds, tuple) and len(ds) == 1:
            ds = ds[0]
    else:
        raise NotImplementedError(f"Dataset {name} not implemented.")
    return ds
