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

from itertools import chain
from functools import partial
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def PubMedQA(dataset, tokenizer, fuse_size=1, batch_size=2, shuffle=False, seed=3704):

    def tokenize_function(examples):
        tok_sample = {
            'text': [],
            'mask': []
        }
        answer_mask = []
        for i in range(0, len(examples['question']), fuse_size):
            long_context = ''
            cap = fuse_size
            if (i*fuse_size + fuse_size) > len(examples['question']):
                cap = len(examples['question']) - i*fuse_size
            for j in range(cap):
                context = ' '.join(examples['context'][i*fuse_size+j]['contexts'])
                long_context = ' '.join([long_context, context])
            
            for j in range(cap):
                question = examples['question'][i*fuse_size+j]
                answer = examples['long_answer'][i*fuse_size+j]
                concat_sample = ' '.join([long_context, question, answer])
                tok_sample['text'].append(concat_sample)
                tok_sample['mask'].append(len(question)+len(long_context)+1)
        
        sample = tokenizer(tok_sample['text'])
        sample['mask'] = tok_sample['mask']
        sample['labels'] = sample['input_ids'].copy()
        return sample
    
    id_pad_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    def collate_fn(batch):
        input_ids = [torch.tensor(b['input_ids'][::-1]) for b in batch]
        labels = [torch.tensor(b['labels'][::-1]) for b in batch]
        attention_mask = [torch.tensor(b['attention_mask'][::-1]) for b in batch]
        mask = [b['mask'] for b in batch]
        input_ids = pad_sequence(input_ids, padding_value=id_pad_value).T.flip(1)
        labels = pad_sequence(labels, padding_value=-100).T.flip(1)
        attention_mask = pad_sequence(attention_mask, padding_value=0).T.flip(1)

        collated = {'input_ids': input_ids,
                    'labels': labels,
                    'attention_mask': attention_mask}

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
        desc='tokenize PubMedQA dataset',
        num_proc=32
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                    shuffle=shuffle, drop_last=False, generator=generator, pin_memory=True)
    
    return dataloader



    
    

