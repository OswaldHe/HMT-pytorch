import numpy as np
import os
import sys
import tqdm
import torch
import datasets
import logging
import math
import accelerate
import datetime
import json

from itertools import chain
from functools import partial
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

def OpenROAD(
    tokenizer,
    ds_file="RAG-EDA/training_dataset/generator_dataset/QA_finetuning_v1v2amend1.jsonl",
    corpus_file="RAG-EDA/benchmark/openroad_documentation.json",
    mode='simple',
    neg_sample=5,
    max_len=4096,
    batch_size=1
):
    dataset = datasets.load_dataset('json', data_files=ds_file)
    dataset = dataset['train']
    split_ds = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds = split_ds['train']
    valid_ds = split_ds['test']
    corpus_knowledge = []

    with open(corpus_file, 'r') as f:
        corpus_dict = json.load(f)
        for entry in corpus_dict:
            corpus_knowledge.extend(entry['knowledge'])

    def tokenize_function(examples, tok_mode='train'):
        prompt_text = {
            'text': [],
            'answer_len': []
        }
        for chat in examples['conversation']:
            chat = chat[0]
            question = chat['question']
            reference = chat['reference_content']
            reference_id = chat['reference']
            answer = chat['answer']
            if mode == 'simple':
                ref_tok = tokenizer.encode(reference)
                if len(ref_tok) > max_len - 80 and tok_mode == 'train' :
                    ref_tok = ref_tok[:(max_len-80)]
                reference = tokenizer.decode(ref_tok)
                messages = [
                    {"role": "system", "content": "You are an expert with EDA tool usage. Answer the question based on the following reference information."},
                    {"role": "system", "content": reference},
                    {"role": "user", "content": question}
                ]
                message_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                message_str += answer
                prompt_text['text'].append(message_str)
                answer_len = len(tokenizer.encode(answer))
                prompt_text['answer_len'].append(answer_len)
            elif mode == 'hard':
                # traverse all knowledge
                candidate_neg_knowledge = []
                for kn in corpus_knowledge:
                    if kn['id'] not in reference_id:
                        candidate_neg_knowledge.append(kn['content'])
                
                neg_knowledge_samples = np.random.choice(candidate_neg_knowledge, neg_sample, replace=False)
                insert_index = np.random.randint(0, neg_sample)
                neg_knowledge_samples = list(neg_knowledge_samples)
                neg_knowledge_samples.insert(insert_index, reference)
                mix_reference = " ".join(neg_knowledge_samples)
                mix_ref_tok = tokenizer.encode(mix_reference)
                if len(mix_ref_tok) > max_len - 80 and tok_mode == 'train':
                    ref_tok = tokenizer.encode(reference)
                    if len(ref_tok) > max_len - 80:
                        ref_tok = ref_tok[:max_len-80]
                    mix_ref_tok = ref_tok
                mix_reference = tokenizer.decode(mix_ref_tok)
                messages = [
                    {"role": "system", "content": "You are an expert with EDA tool usage. Answer the question based on the following reference information."},
                    {"role": "system", "content": mix_reference},
                    {"role": "user", "content": question}
                ]
                message_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                message_str += answer
                prompt_text['text'].append(message_str)
                answer_len = len(tokenizer.encode(answer))
                prompt_text['answer_len'].append(answer_len)
        
        sample = tokenizer(prompt_text['text'])
        sample['labels'] = sample['input_ids'].copy()
        sample['answer_len'] = prompt_text['answer_len']
        return sample
    
    def tokenize_function_train(examples):
        return tokenize_function(examples, tok_mode='train')
    
    def tokenize_function_valid(examples):
        return tokenize_function(examples, tok_mode='valid')

    id_pad_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    def collate_fn(batch):
        input_ids = [torch.tensor(b['input_ids'][::-1]) for b in batch]
        labels = [torch.tensor(b['labels'][::-1]) for b in batch]
        attention_mask = [torch.tensor(b['attention_mask'][::-1]) for b in batch]
        answer_len = [b['answer_len'] for b in batch]
        input_ids = pad_sequence(input_ids, padding_value=id_pad_value).T.flip(1)
        labels = pad_sequence(labels, padding_value=-100).T.flip(1)
        attention_mask = pad_sequence(attention_mask, padding_value=0).T.flip(1)

        collated = {'input_ids': input_ids,
                    'labels': labels,
                    'attention_mask': attention_mask,
                    'answer_len': answer_len}

        return collated
    
    column_names = valid_ds.column_names

    train_tok = train_ds.map(
        tokenize_function_train,
        batched=True,
        remove_columns=column_names,
        desc='tokenize OpenROAD QA training dataset',
        num_proc=8
    )

    valid_tok = valid_ds.map(
        tokenize_function_valid,
        batched=True,
        remove_columns=column_names,
        desc='tokenize OpenROAD QA valid dataset',
        num_proc=8
    )

    generator = torch.Generator()
    generator.manual_seed(42)
    train_dataloader = DataLoader(train_tok, batch_size=batch_size, collate_fn=collate_fn,
                                    shuffle=True, drop_last=False, generator=generator, pin_memory=True)

    valid_dataloader = DataLoader(valid_tok, batch_size=batch_size, collate_fn=collate_fn,
                                    shuffle=False, drop_last=False, pin_memory=True)
    
    return train_dataloader, valid_dataloader

def OpenROAD_test(
    tokenizer,
    ds_file="/home/jovyan/workspace/RAG-EDA/benchmark/ORD-QA.jsonl",
    corpus_file="/home/jovyan/workspace/RAG-EDA/benchmark/openroad_documentation.json",
    mode='simple',
    neg_sample=5,
    batch_size=1
):
    dataset = datasets.load_dataset('json', data_files=ds_file)
    dataset = dataset['train']
    corpus_knowledge = []

    with open(corpus_file, 'r') as f:
        corpus_dict = json.load(f)
        for entry in corpus_dict:
            corpus_knowledge.extend(entry['knowledge'])

    def tokenize_function(examples):
        prompt_text = {
            'text': [],
            'answer': [],
            'answer_len': []
        }
        for i in range(len(examples['id'])):
            question = examples['question'][i]
            reference = " ".join(examples['reference_content'][i])
            reference_id = examples['reference'][i]
            answer = examples['answer'][i]
            if mode == 'simple':
                messages = [
                    {"role": "system", "content": "You are an expert with EDA tool usage. Answer the question based on the following reference information."},
                    {"role": "system", "content": reference},
                    {"role": "user", "content": question}
                ]
                message_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_text['text'].append(message_str)
                prompt_text['answer'].append(answer)
                answer_len = len(tokenizer.encode(answer))
                prompt_text['answer_len'].append(answer_len)
            elif mode == 'hard':
                # traverse all knowledge
                candidate_neg_knowledge = []
                for kn in corpus_knowledge:
                    if kn['id'] not in reference_id:
                        candidate_neg_knowledge.append(kn['content'])
                
                neg_knowledge_samples = np.random.choice(candidate_neg_knowledge, neg_sample, replace=False)
                insert_index = np.random.randint(0, neg_sample)
                neg_knowledge_samples = list(neg_knowledge_samples)
                neg_knowledge_samples.insert(insert_index, reference)
                messages = [
                    {"role": "system", "content": "You are an expert with EDA tool usage. Answer the question based on the reference information."},
                    {"role": "system", "content": " ".join(neg_knowledge_samples)},
                    {"role": "user", "content": question}
                ]
                message_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_text['text'].append(message_str)
                prompt_text['answer'].append(answer)
                answer_len = len(tokenizer.encode(answer))
                prompt_text['answer_len'].append(answer_len)
        
        sample = tokenizer(prompt_text['text'])
        sample['answer'] = prompt_text['answer']
        sample['labels'] = sample['input_ids'].copy()
        sample['answer_len'] = prompt_text['answer_len']
        return sample

    id_pad_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    def collate_fn(batch):
        input_ids = [torch.tensor(b['input_ids'][::-1]) for b in batch]
        labels = [torch.tensor(b['labels'][::-1]) for b in batch]
        attention_mask = [torch.tensor(b['attention_mask'][::-1]) for b in batch]
        answer_len = [b['answer_len'] for b in batch]
        answer = [b['answer'] for b in batch]
        input_ids = pad_sequence(input_ids, padding_value=id_pad_value).T.flip(1)
        labels = pad_sequence(labels, padding_value=-100).T.flip(1)
        attention_mask = pad_sequence(attention_mask, padding_value=0).T.flip(1)

        collated = {'input_ids': input_ids,
                    'labels': labels,
                    'attention_mask': attention_mask,
                    'answer': answer,
                    'answer_len': answer_len}

        return collated
    
    column_names = dataset.column_names

    test_tok = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc='tokenize OpenROAD QA training dataset',
        num_proc=8
    )

    test_dataloader = DataLoader(test_tok, batch_size=batch_size, collate_fn=collate_fn,
                                    shuffle=False, drop_last=False, pin_memory=True)
    
    return test_dataloader

def main():
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    # Call the OpenROAD function
    train_dataloader, valid_dataloader = OpenROAD(tokenizer=tokenizer, mode='hard', neg_sample=6)

    # Print the size of the dataloaders
    print(f"Train dataloader size: {len(train_dataloader)}")
    print(f"Valid dataloader size: {len(valid_dataloader)}")

    # Iterate through the dataloader and print the first batch
    for batch in train_dataloader:
        decoded_inputs = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        print("Decoded inputs:", decoded_inputs)
        print("-------------------------------")

    test_dataloader = OpenROAD_test(tokenizer=tokenizer, mode='simple')
    print(f"Test dataloader size: {len(test_dataloader)}")

    for batch in test_dataloader:
        decoded_inputs = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        print("Decoded inputs:", decoded_inputs)
        break

if __name__ == "__main__":
    main()
