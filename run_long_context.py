import numpy as np
import os
import sys
import tqdm
import torch
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import chain
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from modeling_rmt.language_modeling import MemoryCell, RecurrentWrapper

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

block_size = 1022
mask_size = 2048
history_size = 100000
n_segments = 60

model_name = 'facebook/opt-350m'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

cell = MemoryCell(model, num_mem_tokens=1)
cell.load_state_dict(torch.load("model_ckpt/memory_cell_512_1024_save_32.pt"))
model = RecurrentWrapper(cell,
                        segment_size=block_size,
                        max_n_segments=n_segments)
model.to(device)
model.eval()

"""### Prepare dataset"""

def group_texts(examples, block_size, history_size=None):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    if history_size is None:
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
    else:
        result = {
            k: [t[max({0, i - history_size}) : i + block_size] for i in range(0, total_length, history_size)]
            for k, t in concatenated_examples.items()
        }
    result["labels"] = result["input_ids"].copy()
    return result

id_pad_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
def collate_fn(batch):
    input_ids = [torch.tensor(b['input_ids'][::-1]) for b in batch]
    labels = [torch.tensor(b['labels'][::-1]) for b in batch]
    attention_mask = [torch.tensor(b['attention_mask'][::-1]) for b in batch]
    input_ids = pad_sequence(input_ids, padding_value=id_pad_value).T.flip(1)
    labels = pad_sequence(labels, padding_value=-100).T.flip(1)
    attention_mask = pad_sequence(attention_mask, padding_value=0).T.flip(1)

    collated = {'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask}

    if input_ids.shape[1] != block_size:
        labels_mask = torch.ones_like(input_ids, dtype=bool)
        labels_mask[:, :-mask_size] = False
        collated['labels_mask'] = labels_mask

    return collated

task_name = 'wikitext-103-v1'
raw_datasets = datasets.load_dataset('wikitext', task_name)
column_names = raw_datasets["train"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]

def tokenize_function(examples):
    return tokenizer(examples[text_column_name])

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=column_names,
    desc="Running tokenizer on dataset",
)

valid_dataset = tokenized_datasets["test"].map(lambda x: group_texts(x, history_size, block_size),
                                                        batched=True, desc=f"Grouping valid in chunks of {block_size}")

valid_dataloader = DataLoader(valid_dataset, batch_size=2,
                                        collate_fn=collate_fn, shuffle=False, drop_last=True, pin_memory=True)

valid_gen = iter(valid_dataloader)
eval_steps = 50
valid_losses = []

for step in tqdm.tqdm(range(eval_steps)):
    batch = next(valid_gen)
    for k, v in batch.items():
        batch[k] = v.cpu()
    batch['segment_size'] = block_size
    with torch.no_grad():
        out = model(**batch)
    loss = out.loss

    valid_losses.append(loss.detach().item())

print(f'Loss on 100 validation samples (CrossEntropy): {np.mean(valid_losses)}')
