import datasets
from datasets import load_dataset
import os
from typing import List, Union

HF_HOME = os.environ["HF_HOME"]

"""
Load the NarrativeQA dataset from Hugging Face.
source could be deepmind or longbench.
"""
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


"""
Load the Musique dataset from LongBench from Hugging Face.
"""
def load_musique_dataset(split: Union[List[str], str]="test", **kwargs):
    assert split == "test", "Musique from LongBench dataset only has test split. Also since it's relative small, we don't support further slicing. "
    ds = load_dataset(path="THUDM/LongBench", name="musique", split=split, cache_dir=HF_HOME, streaming=kwargs.get('streaming', False))
    return ds

def prepare_musique_dataset(dataset: datasets.Dataset):
    prompt = "Prompt: Answer the question based on the provided document."
    question_delimiter = "Question: "
    context_delimiter = "Document: "
    answer_delimiter = "Answer: "
    prepared_data = []

    if isinstance(dataset, datasets.Dataset):
        dataset = (dataset, )

    def musique_entry_to_text(dataset):
        """
        Convert a Musique dataset entry to a text entry. 
        The text is a concat of prompt, question delimiter, question, context delimiter, context and label is the answer.
        """
        return {
            "text": [f"{prompt}\n{question_delimiter}{entry[0]}\n{context_delimiter}{entry[1]}\n{answer_delimiter}{entry[2][0]}" for entry in zip(dataset['input'], dataset['context'], dataset['answers'])],
            "answer_length": [len(entry[0]) for entry in dataset['answers']]
        }
    
    prepared_data = ()
    for split in dataset:
        prepared_data = prepared_data + (split.map(musique_entry_to_text, batched=True, 
                                                    desc=f"Concatenating QA Input Questions and Contexts",
                                                    num_proc=8).remove_columns(['input', 'context', 'answers', 'length', 
                                                                                'dataset', 'language', 'all_classes', '_id']),)
    return prepared_data


"""
Load a question answering dataset from Hugging Face.
"""
def load_qa_dataset(name, subset:str=None, max_context_length: int=None, **kwargs):
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
        ds = prepare_narrativeqa_dataset(ds, source='deepmind')
        if isinstance(ds, tuple) and len(ds) == 1:
            ds = ds[0]
    elif name == 'THUDM/LongBench':
        if subset == 'musique':
            ds = load_musique_dataset(split=kwargs.get('split', 'test'), streaming=kwargs.get('streaming', False))
            if isinstance(ds, datasets.Dataset):
                ds = (ds,)
            ds = prepare_musique_dataset(ds)
            if isinstance(ds, tuple) and len(ds) == 1:
                ds = ds[0]
        elif subset == 'qmsum':
            qmsum_data_base = kwargs.get('qmsum_data_base', None)
            assert qmsum_data_base is not None, "qmsum_data_base is not specified. Please specify the path to the qmsum dataset."
        else:
            raise NotImplementedError(f"Dataset subset {subset} not implemented for {name}.")
    else:
        raise NotImplementedError(f"Dataset {name} not implemented.")
    return ds

