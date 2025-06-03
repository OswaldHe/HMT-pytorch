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
import re

from itertools import chain
from functools import partial
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def LongSFT(dataset, tokenizer, batch_size=1, shuffle=False, clip=False, seed=3704):

    def tokenize_function(examples):
        tok_sample = {
            'text': [],
            'answer_len': []
        }

        for ind in range(len(examples['question'])):
            e = {"question": examples['question'][ind], "answer": examples['answer'][ind]}
            if len(e['question']) < 30:
                continue
            m = re.search(r'\bQuestion:\b', e['question'])
            if m:
                context = e['question'][:m.start()]
                question = e['question'][m.start()+len("Question:"):]
                pos = question.find("Assistant:")
                if pos != -1:
                    question = question[:pos]
                answer = e["answer"]
                # clip last 3000 tokens
                if clip:
                    tok_context = tokenizer.encode(context)
                    if len(tok_context) > 2500:
                        tok_context = tok_context[-2500:]
                    context = tokenizer.decode(tok_context)
                    tok_answer = tokenizer.encode(answer)
                    if len(tok_answer) > 300:
                        tok_answer = tok_answer[:300]
                    answer = tokenizer.decode(tok_answer)
                chat = [
                    {"role": "user", "content": context},
                    {"role": "system", "content": "Repeat and recall the previous passage and answer the following question by user."},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
                message_text = tokenizer.apply_chat_template(chat, tokenize=False)
                tok_sample['text'].append(message_text)
                tok_sample['answer_len'].append(len(tokenizer.encode(answer))+1)
            else:
                question = e['question']
                answer = e["answer"]
                pos = question.find("Assistant")
                if pos != -1:
                    question = question[:pos]
                if clip:
                    tok_context = tokenizer.encode(question)
                    if len(tok_context) > 2500:
                        tok_context = tok_context[-2500:]
                    question = tokenizer.decode(tok_context)
                    tok_answer = tokenizer.encode(answer)
                    if len(tok_answer) > 300:
                        tok_answer = tok_answer[:300]
                    answer = tokenizer.decode(tok_answer)
                chat = [
                    {"role": "user", "content": question},
                    {"role": "system", "content": "Answer user's question."},
                    {"role": "assistant", "content": answer}
                ]
                message_text = tokenizer.apply_chat_template(chat, tokenize=False)
                tok_sample['text'].append(message_text)
                tok_sample['answer_len'].append(len(tokenizer.encode(answer))+1)
        
        sample = tokenizer(tok_sample['text'])
        sample['answer_len'] = tok_sample['answer_len']
        sample['labels'] = sample['input_ids'].copy()
        return sample
    
    id_pad_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    def collate_fn(batch):
        input_ids = [torch.tensor(b['input_ids'][::-1]) for b in batch]
        labels = [torch.tensor(b['labels'][::-1]) for b in batch]
        attention_mask = [torch.tensor(b['attention_mask'][::-1]) for b in batch]
        mask_size = [b['answer_len'] for b in batch]
        input_ids = pad_sequence(input_ids, padding_value=id_pad_value).T.flip(1)
        labels = pad_sequence(labels, padding_value=-100).T.flip(1)
        attention_mask = pad_sequence(attention_mask, padding_value=0).T.flip(1)

        collated = {'input_ids': input_ids,
                    'labels': labels,
                    'attention_mask': attention_mask,
                    'mask_size': mask_size}

        # labels_mask = []
        # for i in range(input_ids.shape[0]):
        #     labels_mask.append(input_ids.shape[1] - mask[i])
        # collated['labels_mask'] = labels_mask

        return collated
    
    column_names = dataset.column_names

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=2048,
        remove_columns=column_names,
        desc='tokenize ChatQA2 Long SFT dataset',
        num_proc=32
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                    shuffle=shuffle, drop_last=False, generator=generator, pin_memory=True)
    
    return dataloader



    
    

