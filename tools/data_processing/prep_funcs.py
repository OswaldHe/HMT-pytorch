import datasets

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

