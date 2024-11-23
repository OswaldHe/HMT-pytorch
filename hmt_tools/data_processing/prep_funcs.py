import datasets
from datasets import Dataset
from typing import List

def prepare_narrativeqa(dataset: datasets.Dataset):
    prompt = "Prompt: Answer the question based on the provided document."
    question_delimiter = "Question: "
    context_delimiter = "Document: "
    answer_delimiter = "Answer: "

    def nqa_entry_to_text(dataset):
        """
        Convert a NarrativeQA dataset entry to a text entry. consists of text and labels. 
        The text is a concat of prompt, question delimiter, question, context delimiter, context and label is the answer.
        """
        return {
            "text": [f"{prompt}\n{question_delimiter}{entry[0]}\n{context_delimiter}{entry[1]}\n{answer_delimiter}{entry[2][0]}" for entry in zip(dataset['input'], dataset['context'], dataset['answers'])],
            "answer": [entry[0] for entry in dataset['answers']]
        }
    
    dataset = dataset.map(nqa_entry_to_text, batched=True, 
                                                desc=f"Concatenating QA Input Questions and Contexts",
                                                num_proc=8).remove_columns(['input', 'context', 'answers', 'length', 
                                                                            'dataset', 'language', 'all_classes', '_id'])
    return dataset


def prepare_qasper(dataset: datasets.Dataset):
    prompt = "Prompt: Answer the question based on the provided scientific context."
    question_delimiter = "Question: "
    context_delimiter = "Scientific Context: "
    answer_delimiter = "Answer: "

    def qasper_entry_to_text_longbench(dataset):
        """
        Convert a QMSum dataset entry to a text entry. 
        The text is a concat of prompt, question delimiter, question, context delimiter, context and label is the answer.
        """
        return {
            "text": [f"{prompt}\n{question_delimiter}{entry[0]}\n{context_delimiter}{entry[1]}\n{answer_delimiter}{entry[2][0]}" for entry in zip(dataset['input'], dataset['context'], dataset['answers'])],
            "answer": [entry[0] for entry in dataset['answers']],
        }

    dataset = dataset.map(qasper_entry_to_text_longbench, batched=True, 
                                                    desc=f"Concatenating QA Input Questions and Contexts",
                                                    num_proc=8).remove_columns(['input', 'context', 'answers', 'length', 
                                                                                'dataset', 'language', 'all_classes', '_id'])
    return dataset

def prepare_qmsum_test_ppl(dataset: datasets.Dataset):
    # pre_prompt = "[Prompt]: Remember the following context."
    prompt = "[Prompt]: Remember the following context and answer the question based on the provided context."
    question_delimiter = "[Question]: "
    context_delimiter = "[Context]: "
    answer_delimiter = "[Answer]: "

    def qmsum_entry_to_text_longbench(dataset):
        """
        Convert a QMSum dataset entry to a text entry. 
        The text is a concat of prompt, question delimiter, question, context delimiter, context and label is the answer.
        """
        return {
            "text": [f"{prompt}\n\n{context_delimiter}{entry[1]}\n\n{prompt}\n\n{question_delimiter}{entry[0]}\n\n{answer_delimiter}" for entry in zip(dataset['input'], dataset['context'])],
            "answer": [entry[0] for entry in dataset['answers']],
        }

    dataset = dataset.map(qmsum_entry_to_text_longbench, batched=True, 
                                                    desc=f"Concatenating QA Input Questions and Contexts",
                                                    num_proc=8).remove_columns(['input', 'context', 'answers', 'length', 
                                                                                'dataset', 'language', 'all_classes', '_id'])
    return dataset

def prepare_qmsum_test_it(dataset: datasets.Dataset, tokenizer=None):
    """Instruction tunning version of the QMSum test set. 

    :param dataset: A QMSum test dataset
    :type dataset: datasets.Dataset
    :param tokenizer: tokenizer to use for encoding the text
    :type tokenizer: _type_
    """
    from nltk import word_tokenize
    # tokneize a sent
    def tokenize(sent):
        tokens = ' '.join(word_tokenize(sent.lower()))
        return tokens

    def clean_data(text):
        text = text.replace('{ vocalsound } ', '')
        text = text.replace('{ disfmarker } ', '')
        text = text.replace('a_m_i_', 'ami')
        text = text.replace('l_c_d_', 'lcd')
        text = text.replace('p_m_s', 'pms')
        text = text.replace('t_v_', 'tv')
        text = text.replace('{ pause } ', '')
        text = text.replace('{ nonvocalsound } ', '')
        text = text.replace('{ gap } ', '')
        text = text.replace('{ gap }', '')
        return text

    def qmsum_entry_to_text_longbench(dataset):
        """
        Convert a QMSum dataset entry to a text entry. 
        The text is a concat of prompt, question delimiter, question, context delimiter, context and label is the answer.
        """
        return {
            # "text": [tokenizer.apply_chat_template([{"role": "system", "content": "you are chatbot than can answer questions with given long document"},
            #                                         {"role": "user", "content": "[Document]: \n" +entry[1] + "\n\n" + "[Question]: \n" + entry[0]}], tokenize=False, add_generation_prompt=True) for entry in zip(dataset['input'], dataset['context'])],
            "text": [tokenizer.apply_chat_template([{"role": "system", "content": "you are chatbot than can answer questions with given long document"},
                                                    {"role": "user", "content": clean_data(tokenize(entry[1])) + "\n\n" + tokenize(entry[0])}], tokenize=False, add_generation_prompt=True) for entry in zip(dataset['input'], dataset['context'])],
            "answer": [entry[0] for entry in dataset['answers']],
        }

    dataset = dataset.map(qmsum_entry_to_text_longbench, batched=True, 
                                                    desc=f"Concatenating QA Input Questions and Contexts",
                                                    num_proc=8).remove_columns(['input', 'context', 'answers', 'length', 
                                                                                'dataset', 'language', 'all_classes', '_id'])
    return dataset

def prepare_qmsum_train(dataset: List):
    import re

    def extract_and_remove_s_tags(text):
        # Extract content within <s></s> tags
        extracted = re.findall(r'<s>(.*?)</s>', text, re.DOTALL)
        
        # Remove <s></s> tags and their content from the original text
        cleaned_text = re.sub(r'<s>.*?</s>', '', text, flags=re.DOTALL)
        
        return extracted, cleaned_text.strip()


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

    pre_prompt = "[Prompt]: Remember the following context."
    context_delimiter = "[Context]: "
    prompt = "[Prompt]: Answer the question based on the provided context."
    question_delimiter = "[Question]: "
    answer_delimiter = "[Answer]: "

    def qmsum_entry_to_text(dataset):
        """
        Convert a QMSum dataset entry to a text entry. consists of text and labels. 
        The text is a concat of prompt, question delimiter, question, context delimiter, context and label is the answer.
        """
        return {
            "text": [f"{pre_prompt}\n \
                     {context_delimiter}{entry[1]}\n \
                     {prompt}\n{question_delimiter}{entry[0]}\n \
                     {answer_delimiter}{entry[2]}" for entry in zip(dataset['task'], dataset['meeting_notes'], dataset['answers'])],
            "answer": dataset['answers']
        }

    dataset = dataset.map(qmsum_entry_to_text, batched=True,
                                                    desc=f"Concatenating QMSum Input Questions and Contexts",
                                                    num_proc=8).remove_columns(['task', 'meeting_notes', 'answers'])

    return dataset



def prepare_musique_test_ppl(dataset, **kwargs):
    prompt = "<Prompt>: Answer the question based on the following passages. "
    context_delimiter = "<Passages>: "
    question_delimiter = "<Question>: "
    answer_delimiter = "<Answer>: "

    columns = dataset.column_names

    def musique_entry_to_text(dataset):
        dataset_ = {'text': [prompt + context_delimiter + passages + question_delimiter + question + answer_delimiter for question, passages in zip(dataset['input'], dataset['context'])],
                    'answer': [answer[0] for answer in dataset['answers']]}
        return dataset_
    
    dataset = dataset.map(musique_entry_to_text, batched=True, 
                                                    desc=f"Concatenating QA Input Questions, Answers, and Passagesf for MuSiQue",
                                                    num_proc=kwargs.get('num_proc', 8)).remove_columns(columns)
    return dataset


def prepare_musique_train(dataset):
    prompt = "<Prompt>: Answer the question based on the following passages. "
    context_delimiter = "<Passages>: "
    question_delimiter = "<Question>: "
    answer_delimiter = "<Answer>: "

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


def prepare_qmsum_train_it(dataset: datasets.Dataset, tokenizer=None):
    """Instruction tunning version of the QMSum test set. 

    :param dataset: A QMSum test dataset
    :type dataset: datasets.Dataset
    :param tokenizer: tokenizer to use for encoding the text
    :type tokenizer: _type_
    """
    def qmsum_entry_to_text_template(dataset):
        """
        Convert a QMSum dataset entry to a text entry. 
        The text is a concat of prompt, question delimiter, question, context delimiter, context and label is the answer.
        """
        return {
            # "text": [tokenizer.apply_chat_template([{"role": "system", "content": "you are chatbot than can answer questions with given long document"},
            #                                         {"role": "user", "content": "[Document]: \n" +entry[1] + "\n\n" + "[Question]: \n" + entry[0]}], tokenize=False, add_generation_prompt=True) for entry in zip(dataset['input'], dataset['context'])],
            "text": [tokenizer.apply_chat_template([{"role": "system", "content": "you are chatbot than can answer questions with given long document"},
                                                    {"role": "user", "content": entry[1] + "\n\n" + entry[0] + "\n\n"},
                                                    {"role": "assistant", "content": entry[2]}], tokenize=False, add_generation_prompt=True) for entry in zip(dataset['task'], dataset['meeting_notes'], dataset['answers'])],
            "answer": dataset['answers']
        }

    import re

    def extract_and_remove_s_tags(text):
        # Extract content within <s></s> tags
        extracted = re.findall(r'<s>(.*?)</s>', text, re.DOTALL)
        
        # Remove <s></s> tags and their content from the original text
        cleaned_text = re.sub(r'<s>.*?</s>', '', text, flags=re.DOTALL)
        
        return extracted, cleaned_text.strip()


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

    dataset = dataset.map(qmsum_entry_to_text_template, batched=True,
                                                    desc=f"Concatenating QMSum Input Questions and Contexts",
                                                    num_proc=8).remove_columns(['task', 'meeting_notes', 'answers'])
    return dataset


def prepare_dolly_sum_train(dataset, tokenizer):

    def dolly_entry_to_text_template(dataset):
        """
        Convert a Dolly dataset entry to a text entry. 
        The text is a concat of prompt, question delimiter, question, context delimiter, context and label is the answer.
        """
        return {
            # "text": [tokenizer.apply_chat_template([{"role": "system", "content": "you are chatbot than can answer questions with given long document"},
            #                                         {"role": "user", "content": "[Document]: \n" +entry[1] + "\n\n" + "[Question]: \n" + entry[0]}], tokenize=False, add_generation_prompt=True) for entry in zip(dataset['input'], dataset['context'])],
            "text": [tokenizer.apply_chat_template([{"role": "system", "content": "you are chatbot than can answer questions with given long document"},
                                                    {"role": "user", "content": entry[1].replace('\n\n', '\n') + "\n\n" + entry[0].replace('\n\n', '\n') + "\n\n"},
                                                    {"role": "assistant", "content": entry[2].replace('\n\n', '\n')}], tokenize=False, add_generation_prompt=True) for entry in zip(dataset['instruction'], dataset['context'], dataset['response'])],
            "answer": [d.replace('\n\n', '\n') for d in dataset['response']]
        }

    dataset = dataset.filter(lambda x: x['category'] == 'summarization')

    dataset = dataset.map(dolly_entry_to_text_template, batched=True,
                                                        desc=f"Concatenating QA Input Questions and Contexts",
                                                        num_proc=8).remove_columns(['instruction', 'context', 'response', 'category'])
    return dataset

def prepare_nihs_train(dataset, tokenizer, task, use_chat_template=True, use_instruction=True, use_examples=True, use_post_prompt=True):
    # print(dataset)
    from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input
    prompt_cfg = {
        'instruction': DEFAULT_PROMPTS[task]['instruction'] if use_instruction else '',
        'examples': DEFAULT_PROMPTS[task]['examples'] if use_examples else '',
        'post_prompt': DEFAULT_PROMPTS[task]['post_prompt'] if use_post_prompt else '',
        'template': DEFAULT_TEMPLATE,
        'chat_template': use_chat_template,
    }

    def nihs_entry_to_text_template(dataset):
        """
        Convert a Dolly dataset entry to a text entry. 
        The text is a concat of prompt, question delimiter, question, context delimiter, context and label is the answer.
        """
        return {
            # "text": [tokenizer.apply_chat_template([{'role': 'user', 'content': get_formatted_input(context, 
            #                                                                                         question, prompt_cfg['examples'], 
            #                                                                                         prompt_cfg['instruction'], prompt_cfg['post_prompt'], 
            #                                                                                         template=prompt_cfg['template'])},
            "text": [get_formatted_input(context, 
                                        question, prompt_cfg['examples'], 
                                        prompt_cfg['instruction'], prompt_cfg['post_prompt'], 
                                        template=prompt_cfg['template']) + "\n\n" + target for context, question, target in zip(dataset['input'], dataset['question'], dataset['target'])],
            "answer": dataset['target']
        }

    dataset = dataset.map(nihs_entry_to_text_template, batched=True,
                                                        desc=f"Concatenating Babilong Input Questions and Contexts",
                                                        num_proc=8).remove_columns(['input', 'question', 'target'])
    return dataset

def prepare_nihs_test(dataset, tokenizer, task, use_chat_template=True, use_instruction=True, use_examples=True, use_post_prompt=True):
    from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input
    prompt_cfg = {
        'instruction': DEFAULT_PROMPTS[task]['instruction'] if use_instruction else '',
        'examples': DEFAULT_PROMPTS[task]['examples'] if use_examples else '',
        'post_prompt': DEFAULT_PROMPTS[task]['post_prompt'] if use_post_prompt else '',
        'template': DEFAULT_TEMPLATE,
        'chat_template': use_chat_template,
    }

    def nihs_entry_to_text_template(dataset):
        """
        Convert a Dolly dataset entry to a text entry. 
        The text is a concat of prompt, question delimiter, question, context delimiter, context and label is the answer.
        """
        return {
            # "text": [tokenizer.apply_chat_template([{'role': 'user', 'content': get_formatted_input(context, 
            #                                                                                         question, prompt_cfg['examples'], 
            #                                                                                         prompt_cfg['instruction'], prompt_cfg['post_prompt'], 
            #                                                                                         template=prompt_cfg['template'])}], tokenize=False, add_generation_prompt=True) for context, question, target in zip(dataset['input'], dataset['question'], dataset['target'])],
            "text": [get_formatted_input(context, 
                                                                                                    question, prompt_cfg['examples'], 
                                                                                                    prompt_cfg['instruction'], prompt_cfg['post_prompt'], 
                                                                                                    template=prompt_cfg['template']) for context, question, target in zip(dataset['input'], dataset['question'], dataset['target'])],
            "answer": dataset['target']
        }

    dataset = dataset.map(nihs_entry_to_text_template, batched=True,
                                                        desc=f"Concatenating Babilong Input Questions and Contexts",
                                                        num_proc=8).remove_columns(['input', 'question', 'target'])
    return dataset

def prepare_longbench_test(dataset, subset_name):
    # How longbench set the delimiter? 
    # prompt = "<Prompt>: Answer the question based on the following passages. "
    # context_delimiter = "<Passages>: "
    # question_delimiter = "<Question>: "
    # answer_delimiter = "<Answer>: "

    import json
    with open("/home/yingqi/repo/HMT-pytorch/configs/dataset2prompt.json", "r") as f:
        prompt_cfg = json.load(f)
    prompt = prompt_cfg[subset_name]

    columns = dataset.column_names

    def apply_prompt(prompt, context, question):
        return prompt.replace("{context}", context).replace("{input}", question)

    def musique_entry_to_text(dataset):
        dataset_ = {'text': [apply_prompt(prompt, cxt, question) for question, cxt in zip(dataset['input'], dataset['context'])],
                    'answer': [answer[0] for answer in dataset['answers']]}
        return dataset_
    
    dataset = dataset.map(musique_entry_to_text, batched=True, 
                                                    desc=f"Concatenating QA Input Questions, Answers, and Passagesf for MuSiQue").remove_columns(columns)
    return dataset