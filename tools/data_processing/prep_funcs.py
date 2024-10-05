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
