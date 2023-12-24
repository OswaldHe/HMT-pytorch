import numpy as np
import os
import sys
import tqdm
import torch
import datasets
from matplotlib import pyplot as plt
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import chain
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from modeling_rmt.language_modeling import MemoryCell, RecurrentWrapper
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from accelerate import Accelerator

def main():

    accelerator = Accelerator()
    device = accelerator.device

    block_size = 254
    mask_size = 512
    n_segments = 10
    history_size = (n_segments-1)*block_size
    batch_size = 2

    torch.manual_seed(1358)
    np.random.seed(1358)

    model_name = 'facebook/opt-2.7b'
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    cell = MemoryCell(model, num_mem_tokens=1)
    ori_model = RecurrentWrapper(cell,
                        segment_size=block_size,
                        max_n_segments=n_segments,
                        )
    state_dict = get_fp32_state_dict_from_zero_checkpoint('opt_2.7b_model_ckpt_deepspeed')
    ori_model.load_state_dict(state_dict)
    cell = copy.deepcopy(ori_model.memory_cell)
    model = RecurrentWrapper(cell,
                            copy.deepcopy(model.get_input_embeddings()),
                            segment_size=block_size,
                            max_n_segments=n_segments*2,
                            mask_size=mask_size,
                            n_cell_out=5)

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
            labels_mask[:, :-block_size] = False
            collated['labels_mask'] = labels_mask

        return collated


    task_name = 'wikitext-2-v1'
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

    train_dataset = tokenized_datasets["train"].map(lambda x: group_texts(x, history_size, block_size),
                                                            batched=True, desc=f"Grouping train in chunks of {block_size} and history {history_size}")
    valid_dataset = tokenized_datasets["validation"].map(lambda x: group_texts(x, history_size, block_size),
                                                            batched=True, desc=f"Grouping valid in chunks of {block_size}")

    train_rnd_generator = torch.Generator()
    train_rnd_generator.manual_seed(42)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                    shuffle=True, drop_last=False, generator=train_rnd_generator, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                            collate_fn=collate_fn, shuffle=False, drop_last=True, pin_memory=True)


    from torch.optim import AdamW
    optim = AdamW(params=model.parameters(), lr=8e-6)

    model, optim, train_dataloader = accelerator.prepare(
        model, optim, train_dataloader
    )

    model.to(device)
    model.train()

    train_steps = 200
    eval_steps = 50

    train_gen = iter(train_dataloader)
    valid_gen = iter(valid_dataloader)

    losses = []
    for step in tqdm.tqdm(range(train_steps)):
        optim.zero_grad()

        batch = next(train_gen)
        for k, v in batch.items():
            batch[k] = v.cpu()
        batch['segment_size'] = block_size
        out = model(**batch)
        loss = out.loss

        accelerator.backward(loss)
        optim.step()

        losses.append(loss.detach().item())

    plt.plot(losses)
    plt.xlabel('step')
    plt.ylabel('train loss')
    plt.savefig('loss.png')
    plt.show()

    model.eval()
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

    print(f'Loss on {eval_steps * batch_size} validation samples (CrossEntropy): {np.mean(valid_losses)}')

    test_dataset = tokenized_datasets["validation"].map(lambda x: group_texts(x, 10000, block_size),
                                                            batched=True, batch_size=len(tokenized_datasets["validation"]), desc=f"Grouping valid in chunks of {block_size}")

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                            collate_fn=collate_fn, shuffle=False, drop_last=True, pin_memory=True)


    test_gen = iter(test_dataloader)
    test_losses = []

    for step in tqdm.tqdm(range(eval_steps)):
        batch = next(test_gen)
        for k, v in batch.items():
            batch[k] = v.cpu()
        batch['segment_size'] = block_size
        with torch.no_grad():
            out = model(**batch)
        loss = out.loss

        test_losses.append(loss.detach().item())

    print(f'Loss on {eval_steps * batch_size} test samples (CrossEntropy): {np.mean(test_losses)}')

if __name__ == "__main__":
    main()
