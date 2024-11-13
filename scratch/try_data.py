from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM-135M-Instruct', cache_dir='/home/yingqi/scratch/hf_home')
ds = load_dataset("databricks/databricks-dolly-15k", split='train')


def dolly_entry_to_text_template(dataset):
    """
    Convert a QMSum dataset entry to a text entry. 
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

ds = ds.filter(lambda x: x['category'] == 'summarization')

ds = ds.map(dolly_entry_to_text_template, batched=True,
                                                    desc=f"Concatenating QA Input Questions and Contexts",
                                                    num_proc=8).remove_columns(['instruction', 'context', 'response', 'category'])

print(ds.column_names)
for i in range(10):
    print(i)
    print(ds[i]['text'])
    print(ds[i]['answer'])
    print('-'*100)