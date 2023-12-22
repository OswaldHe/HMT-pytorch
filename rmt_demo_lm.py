import numpy as np
import os
import sys
import tqdm
import torch
import datasets
from matplotlib import pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import chain
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence

from accelerate import Accelerator

# Experiments:
# TODO(DONE): OPT-2.7B w/o long term mem
# TODO(OOM): OPT-2.7B w/ long mem 5 token attention (maybe too short)
# TODO: change cross-attn arch (add output linear + MLPs ?)
# TODO: only attend first k tokens / transform to lower dim
# TODO: OPT-1.3B w/o long mem (various config and seq length)
# TODO: OPT-1.3B w/ long mem 10 token attn
# TODO: OPT-350M & OPT-1.3B test larger dataset (war and peace + les miserables)
# TODO: multi-level summarization (? optimizer state may explode)

def main():
    accelerator = Accelerator()

    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = accelerator.device
    # sys.path.append('github')
    # sys.path.append('..')
    # device = 'cpu'

    torch.manual_seed(1358)

    """### Load model"""

    model_name = 'facebook/opt-2.7b'
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_size = 256
    memory_size = 1
    n_segments = 2

    batch_size = 2

    block_size = input_size
    block_size -= 2 * memory_size
    history_size = (n_segments - 1) * block_size

    mask_size = block_size

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

    train_dataset = tokenized_datasets["train"].map(lambda x: group_texts(x, block_size, history_size),
                                                            batched=True, desc=f"Grouping train in chunks of {block_size} and history {history_size}")
    valid_dataset = tokenized_datasets["validation"].map(lambda x: group_texts(x, block_size, history_size),
                                                            batched=True, desc=f"Grouping valid in chunks of {block_size}")

    train_rnd_generator = torch.Generator()
    train_rnd_generator.manual_seed(42)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                    shuffle=True, drop_last=False, generator=train_rnd_generator, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                            collate_fn=collate_fn, shuffle=False, drop_last=True, pin_memory=True)
    
    test_dataset = tokenized_datasets["validation"].map(lambda x: group_texts(x, 100000, block_size),
                                                            batched=True, batch_size=len(tokenized_datasets["validation"]), desc=f"Grouping valid in chunks of 510")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                            collate_fn=collate_fn, shuffle=False, drop_last=True, pin_memory=True)

    """### Add RMT"""

    from modeling_rmt.language_modeling import MemoryCell, RecurrentWrapper

    cell = MemoryCell(model, num_mem_tokens=memory_size)
    model = RecurrentWrapper(cell,
                            segment_size=block_size,
                            max_n_segments=n_segments*4,
                            mask_size=mask_size,
                            n_cell_out=4
                            )
    model.to(device)

    try:
        model.eval()
        gen = iter(train_dataloader)
        batch = next(gen)
        batch.pop('labels_mask')
        with torch.no_grad():
            out = model(**batch)
        print('Success!')
    except IndexError:
        print('Error: Input size too large!')

    """### Train the model"""

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
    optim = AdamW(params=model.parameters(), lr=1e-05)
    lr_sched = LambdaLR(optim, lr_lambda=(lambda step: 0.5 if step%2 == 0 else 1.0))

    train_steps = 250
    eval_steps = 100

    model, optim, train_dataloader, valid_dataloader, lr_sched = accelerator.prepare(
        model, optim, train_dataloader, valid_dataloader, lr_sched
    )

    train_gen = iter(train_dataloader)
    valid_gen = iter(valid_dataloader)

    model.to(device)
    model.train()

    block_size_var = [(254, 5e-6), (126, 1e-5)]
    losses = []
    for step in tqdm.tqdm(range(train_steps)):
        optim.zero_grad()

        batch = next(train_gen)
        for k, v in batch.items():
            batch[k] = v.cpu()

        for b, lr in block_size_var:

            batch['segment_size'] = b
            out = model(**batch)
            loss = out.loss

            accelerator.backward(loss)
            optim.step()
            lr_sched.step()

            losses.append(loss.detach().item())

    plt.plot(losses)
    plt.xlabel('step')
    plt.ylabel('train loss')
    plt.savefig('loss.png')
    plt.show()

    valid_losses = []
    model.eval()
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

    accelerator.wait_for_everyone()

    # TODO(DONE): RMT is unstable for variable segment width
    # TODO(DONE): Need a benchmark for very long context (60~100k) and topic switching (try long novels and law cases)
    # TODO(DONE): Baseline testing (sliding window)

    test_losses = []

    test_gen = iter(test_dataloader)

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
