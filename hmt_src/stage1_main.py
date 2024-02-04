import numpy as np
import os
import sys
import tqdm
import torch
import datasets
import copy
import json
import logging
import math
import accelerate

from matplotlib import pyplot as plt
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import chain
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from accelerate import Accelerator
from pathlib import Path
from peft import get_peft_model, LoraConfig, TaskType
from modeling_rmt.language_modeling import MemoryCell, RecurrentWrapper

# set up logging
logging_fmt = "[%(levelname)s] (%(asctime)s): %(message)s"
date_fmt = '%m/%d/%Y %I:%M:%S %p'
logging.basicConfig(format=logging_fmt, datefmt=date_fmt, level=logging.INFO)
setup_logger = logging.getLogger('')

setup_logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

parser = ArgumentParser()

#cli arguments
parser.add_argument('--task_name', type=str, default='wikitext', help='training/validation task name (e.g. wikitext, pg19, samsum, etc.)')
parser.add_argument('--task_subset', type=str, default='wikitext-103-v1', help='subset of dataset (e.g., wikitext-2-v1)')
parser.add_argument('--batch_size', type=int, default=2, help='number of batches per device')
parser.add_argument('--num_seg_save', type=int, default=5, help='max number of segment inference results saved on GPU')
parser.add_argument('--seed', type=int, default=3407, help='random seed for training')
parser.add_argument('--model_name', type=str, default='facebook/opt-2.7b', help='transformer model name for backbone of HMT')
parser.add_argument('--segment_length', type=int, default=256, help='segment length of HMT')
parser.add_argument('--bptt_depth', type=int, default=2, help='number of segments unrolled in bptt')
parser.add_argument('--test_length', type=int, default=2000, help='context length of input to test')
parser.add_argument('--mask_size', type=int, default=256, help='loss computation mask length')
parser.add_argument('--training_step', type=int, default=500, help='number of training steps')
parser.add_argument('--eval_step', type=int, default=100, help='number of evaluation steps')
parser.add_argument('--test_step', type=int, default=100, help='number of testing steps')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='training learning rate')
parser.add_argument('--lr_decay', action='store_true', default=False, help='whether having learning rate decay or not')
parser.add_argument('--use_lora', action='store_true', default=False, help='whether use PEFT LoRA to speed up training')
parser.add_argument('--lr_decay_gamma', type=float, default=0.8, help='rate of lr decay')


torch.manual_seed(3407)

def main():
    global torch

    args = parser.parse_args()
    accelerator = Accelerator()
    device = accelerator.device
    from accelerate.logging import get_logger
    logger = get_logger('')

    """### Load model"""
    model = AutoModelForCausalLM.from_pretrained(args.model_name, token='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')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token='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')

    if args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=8, 
            lora_alpha=16, 
            lora_dropout=0.1
            )
        model = get_peft_model(model, peft_config)
        logger.info(f'Added LoRA, trainable parameters with LoRA only:')
        model.print_trainable_parameters()

    input_size = args.segment_length
    memory_size = 1
    n_segments = args.bptt_depth

    batch_size = args.batch_size

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

    task_name = args.task_subset
    train_ds, valid_ds, test_ds = datasets.load_dataset(args.task_name, task_name, split=['train', 'validation', 'test'])
    column_names = train_ds.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    train_ds_tok = train_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on training dataset",
        num_proc=32
    )

    valid_ds_tok = valid_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on valid dataset",
        num_proc=32
    )

    test_ds_tok = test_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on test dataset",
        num_proc=32
    )

    train_dataset = train_ds_tok.map(lambda x: group_texts(x, history_size, block_size),
                                                            batched=True, desc=f"Grouping train in chunks of {block_size} and history {history_size}")
    valid_dataset = valid_ds_tok.map(lambda x: group_texts(x, history_size, block_size),
                                                            batched=True, desc=f"Grouping valid in chunks of {block_size}")

    train_rnd_generator = torch.Generator()
    train_rnd_generator.manual_seed(args.seed)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                    shuffle=True, drop_last=False, generator=train_rnd_generator, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                            collate_fn=collate_fn, shuffle=False, drop_last=True, pin_memory=True)
    
    test_dataset = test_ds_tok.map(lambda x: group_texts(x, args.test_length, block_size),
                                                            batched=True, desc=f"Grouping valid in chunks of {block_size}")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                            collate_fn=collate_fn, shuffle=False, drop_last=True, pin_memory=True)


    cell = MemoryCell(model, num_mem_tokens=memory_size)
    model = RecurrentWrapper(cell,
                            copy.deepcopy(model.get_input_embeddings()),
                            segment_size=block_size,
                            max_n_segments=n_segments,
                            mask_size=mask_size,
                            n_cell_out=args.num_seg_save
                            )

    from torch.optim import AdamW
    optim = AdamW(params=model.parameters(), lr=args.learning_rate)
    from torch.optim.lr_scheduler import StepLR
    if args.lr_decay:
        scheduler = StepLR(optim, step_size=100, gamma=args.lr_decay_gamma)
    else:
        scheduler = StepLR(optim, step_size=100, gamma=1.0)

    train_steps = args.training_step
    eval_steps = args.eval_step

    model, optim, train_dataloader, valid_dataloader, scheduler = accelerator.prepare(
        model, optim, train_dataloader, valid_dataloader, scheduler
    )

    train_gen = iter(train_dataloader)
    valid_gen = iter(valid_dataloader)

    model.to(device)
    model.train()

    losses = []
    for step in tqdm.tqdm(range(train_steps)):
        optim.zero_grad()

        batch = next(train_gen)
        for k, v in batch.items():
            batch[k] = v.cpu()

        batch['segment_size'] = block_size
        out, _ = model(**batch)
        loss = out.loss

        accelerator.backward(loss)
        optim.step()
        if args.lr_decay:
            scheduler.step()
        logger.info(f'loss: {loss.item()}')
        logger.info(f'ppl: {out.ppl.item()}')
        losses.append(loss.detach().item())

    accelerator.wait_for_everyone()

    plt.plot(losses)
    plt.xlabel('step')
    plt.ylabel('train loss')
    plt.savefig('loss.png')
    plt.show()

    valid_losses = []
    valid_ppl = []
    model.eval()
    for step in tqdm.tqdm(range(eval_steps)):
        batch = next(valid_gen)
        for k, v in batch.items():
            batch[k] = v.cpu()
        batch['segment_size'] = block_size
        with torch.no_grad():
            out, _ = model(**batch)
        loss = out.loss
        ppl = out.ppl

        valid_losses.append(loss.detach().item())
        valid_ppl.append(ppl.detach().item())

    print(f'Loss on {eval_steps * batch_size} validation samples (CrossEntropy): {np.mean(valid_losses)}')
    print(f'PPL on {eval_steps * batch_size} validation samples: {np.mean(valid_ppl)}')

    test_losses = []
    test_ppl = []

    test_gen = iter(test_dataloader)

    for step in tqdm.tqdm(range(args.test_step)):
        batch = next(test_gen)
        for k, v in batch.items():
            batch[k] = v.cpu()
        batch['segment_size'] = block_size
        with torch.no_grad():
            out, _ = model(**batch)
        loss = out.loss
        ppl = out.ppl
        test_losses.append(loss.detach().item())
        test_ppl.append(ppl.detach().item())

    print(f'PPL on {args.test_step * batch_size} test samples: {np.mean(test_ppl)}')

if __name__ == "__main__":
    main()
