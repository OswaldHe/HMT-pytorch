from datasets import load_dataset
from .tokenize import tokenize_dataset, filter_by_length, tokenize_column
from .grouping import group_dataset
import os

HF_HOME = os.getenv('HF_HOME', None)



def prepare_musique_test_ppl(dataset, **kwargs):
    prompt = "<Prompt>: Answer the question based on the following passages. "
    context_delimiter = "\n<Passages>: "
    question_delimiter = "\n<Question>: "
    answer_delimiter = "\n<Answer>: "

    columns = dataset.column_names

    def musique_entry_to_text(dataset):
        dataset_ = {'text': [prompt + context_delimiter + passages + question_delimiter + question + answer_delimiter + answer[0] for question, passages, answer in zip(dataset['input'], dataset['context'], dataset['answers'])],
                    'answer': [answer[0] for answer in dataset['answers']]}
        return dataset_
    
    dataset = dataset.map(musique_entry_to_text, batched=True, 
                                                    desc=f"Concatenating QA Input Questions, Answers, and Passagesf for MuSiQue",
                                                    num_proc=kwargs.get('num_proc', 8)).remove_columns(columns)
    return dataset


def prepare_musique_train(dataset):
    prompt = "<Prompt>: Answer the question based on the following passages. "
    context_delimiter = "\n<Passages>: "
    question_delimiter = "\n<Question>: "
    answer_delimiter = "\n<Answer>: "

    dataset = dataset.filter(lambda x: x['answerable'] == True)

    remove_columns = dataset.column_names
    remove_columns.remove('answer')

    concatenated_paragraphs = []
    for paragraphs in dataset['paragraphs']:
        concatenated_paragraph = ""
        for i, paragraph_dict in enumerate(paragraphs):
            concatenated_paragraph += f"Paragraph {i}: " + " (title): " + paragraph_dict['title'] + " (paragraph content): " + paragraph_dict['paragraph_text']
        concatenated_paragraphs.append(concatenated_paragraph)

    def musique_entry_to_text(dataset):
        dataset_ = {'text': [prompt + context_delimiter + passages + question_delimiter + question + answer_delimiter + answer for question, passages, answer in zip(dataset['question'], concatenated_paragraphs, dataset['answer'])],
                    'answer': [answer[0] for answer in dataset['answer']]}
        return dataset_


    dataset = dataset.map(musique_entry_to_text, batched=True, 
                                                    desc=f"Concatenating QA Input Questions, Answers, and Passagesf for MuSiQue",
                                                    num_proc=8).remove_columns(remove_columns)

    return dataset

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


def load_musique_test(test_length, block_size, tokenizer, with_answer=False, **kwargs):
    ds = load_dataset("THUDM/LongBench", "musique", split="test", streaming=kwargs.get('streaming', False), cache_dir=HF_HOME)
    ds = prepare_musique_test_ppl(ds)  # Prepare the dataset in PPL Testing format. 
    ds = tokenize_dataset(ds, tokenizer=tokenizer, is_qa_task=True, text_column_name='text', **kwargs)
    
    answer_ids = tokenize_column(ds, tokenizer, column_name='answer')
    ds = ds.add_column("mask_size", [len(answer_id) for answer_id in answer_ids])

    keep_columns = {'input_ids', 'attention_mask', 'labels', 'mask_size'}
    if with_answer: keep_columns.add('answer')

    # Remove unnecessary columns
    column_to_remove = [name for name in ds.column_names if name not in keep_columns]
    ds = ds.remove_columns(column_to_remove)
    
    ds = group_dataset(ds, split='test', history_size=test_length, block_size=block_size, is_qa_task=True, with_answer=with_answer)
    return ds

