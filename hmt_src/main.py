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
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTConfig
from itertools import chain
from functools import partial
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from accelerate import Accelerator, DistributedDataParallelKwargs
from pathlib import Path
from peft import get_peft_model, LoraConfig, TaskType
from modeling_rmt.language_modeling import MemoryCell, RecurrentWrapper
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from hmt_src.pubmedqa_ds_preprocess import PubMedQA
from hmt_src.openroad_qa_preprocess import OpenROAD, OpenROAD_test
from modeling_rmt.compression import inject_eae
from accelerate.utils import DummyOptim, DummyScheduler


# set up logging
logging_fmt = "[%(levelname)s] (%(asctime)s): %(message)s"
date_fmt = '%m/%d/%Y %I:%M:%S %p'
logging.basicConfig(format=logging_fmt, datefmt=date_fmt, level=logging.INFO)
setup_logger = logging.getLogger('')

setup_logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

# Create a new logger for the program process
logger = logging.getLogger('program_process')
logger.setLevel(logging.INFO)

# Create a file handler
file_handler = logging.FileHandler('program_process.log')
file_handler.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set the formatter for both handlers
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


parser = ArgumentParser()

#cli arguments
parser.add_argument('--task_name', type=str, default='wikitext', help='training/validation task name (e.g. wikitext, pg19, samsum, etc.)')
parser.add_argument('--task_subset', type=str, default=None, help='subset of dataset (e.g., wikitext-2-v1)')
parser.add_argument('--batch_size', type=int, default=2, help='number of batches per device')
parser.add_argument('--num_seg_save', type=int, default=5, help='max number of segment inference results saved on GPU')
parser.add_argument('--seed', type=int, default=3407, help='random seed for training')
parser.add_argument('--model_name', type=str, default='facebook/opt-2.7b', help='transformer model name for backbone of HMT')
parser.add_argument('--segment_length', type=int, default=256, help='segment length of HMT')
parser.add_argument('--bptt_depth', type=int, default=2, help='number of segments unrolled in bptt')
parser.add_argument('--sum_fraction', type=float, default=0.5, help='fraction of the segment that will be used for representation extraction')
parser.add_argument('--test_length', type=int, default=2000, help='context length of input to test')
parser.add_argument('--training_step', type=int, default=500, help='number of training steps')
parser.add_argument('--eval_step', type=int, default=100, help='number of evaluation steps')
parser.add_argument('--test_step', type=int, default=100, help='number of testing steps')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='training learning rate')
parser.add_argument('--lr_decay', action='store_true', default=False, help='whether having learning rate decay or not')
parser.add_argument('--use_lora', action='store_true', default=False, help='whether use PEFT LoRA to speed up training')
parser.add_argument('--lr_decay_gamma', type=float, default=0.8, help='rate of lr decay')
parser.add_argument('--num_sensory', type=int, default=0, help='number of preserved tokens for sensory memory')
parser.add_argument('--mem_recall_hidden_dim', type=int, default=4096, help='hidden dimension of cross attention in memory recall mech.')
parser.add_argument('--rmt_only', action='store_true', default=False, help='train and evaluate with only rmt')
parser.add_argument('--baseline_only', action='store_true', default=False, help='train and evaluate only the backbone model')
parser.add_argument('--segment_alignment', type=str, default=None, help='alignment of segments in evaluation.')
parser.add_argument('--hmt_stage_1', action='store_true', default=False, help='stage 1 of HMT to find memory param')
parser.add_argument('--hmt_stage_2', action='store_true', default=False, help='stage 2 of HMT to find memory param')
parser.add_argument('--save_ckpt', type=str, default=None, help='store the model checkpoint to the specified directory, only used for HMT')
parser.add_argument('--load_from_ckpt', type=str, default=None, help='load the checkpoint for HMT stage 2')
parser.add_argument('--mem_recall_context', type=int, default=100, help='number of memory embeddings cached in memory recall mech.')
parser.add_argument('--token_file', type=str, default=None, help='path to the file with Huggingface token. Used for gated model such as Llama2.')
parser.add_argument('--train_set_split', type=str, default=None, 
        help='slice upper bound of training set to reduce time for tokenization. use percentage notation (e.g., 2%), or integer')
parser.add_argument('--interleave_dataset', action='store_true', default=False, help='whether mix every two samples in the dataset to create context switching.')
parser.add_argument('--interleave_len', type=int, default=100, help='the interleaving length of dataset (first sample pick some tokens, then the second).')
parser.add_argument('--plot_hist', action='store_true', default=False, help='show memory recall context histogram.')
parser.add_argument('--fuse_size', type=int, default=2, help='the number of questions and context to fuse for PubMedQA dataset')
parser.add_argument('--timing', action='store_true', default=False, help='profile the timing of inference.')
parser.add_argument('--inference_only', action='store_true', default=False, help='perform inference of the model only.')
parser.add_argument('--dynamic', action='store_true', default=False, help='whether dynamically change reading speed based on memory.')
parser.add_argument('--dilate_dataset', action='store_true', default=False, help='dilate the sample by inserting padding tokens.')
parser.add_argument('--dilate_len', type=int, default=888, help='number of padding tokens inserted to dilate the sample.')
parser.add_argument('--dilate_str', type=str, default='$', help='the token you want to insert to dilate the sample.')
parser.add_argument('--train_memory_map', action='store_true', default=False, help='train memory projection for dynamic reading speed.')
parser.add_argument('--inject_autoencoder', action='store_true', default=False, help='use autoencoder to compress/decompress the intermediate embeddings.')
parser.add_argument('--generate', type=str, default=None, help='generate for harry potter book.')
parser.add_argument('--streaming', action='store_true', default=False, help='generate text in streaming mode')
parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle the dataset')
parser.add_argument('--shuffle_train', action='store_true', default=True, help='shuffle the training dataset')
parser.add_argument('--cache_dir', type=str, default='.', help='cache directory, default to the current directory')
parser.add_argument('--wandb_project', type=str, default=None, help='Name for the WanDB Project')
parser.add_argument('--wandb_run', type=str, default=None, help='Name for the WanDB run')
parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity (username or team name)')
parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs to train the model')

torch.manual_seed(3407)

def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds

def main():
    global torch

    args = parser.parse_args()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], log_with="wandb")
    device = accelerator.device
    from accelerate.logging import get_logger
    logger = get_logger('')

    # Initialize WanDB Tracker
    accelerator.init_trackers(
        project_name=args.wandb_project, 
        config={"dropout": 0.1, 
                "learning_rate": args.learning_rate, 
                "model_name": args.model_name,
                "task_name": args.task_name, 
                "test_length": args.test_length},
        init_kwargs={"wandb": {"entity": args.wandb_entity, "name": f'{args.wandb_run}'}}
    )

    token=None
    if args.token_file is not None:
        with open(args.token_file, 'r') as f:
            token = f.read()

    """### Clearning CUDA Cache"""
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    """### Load model"""
    cache_dir = os.environ.get('HF_HOME', args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, token=token, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=token, cache_dir=cache_dir)
    
    if isinstance(model.config, OPTConfig):
        word_emb_dim = model.config.word_embed_proj_dim
    else:
        word_emb_dim = model.config.hidden_size

    if args.inject_autoencoder:
        model = inject_eae(model, word_emb_dim, 16, 2)

    if args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False, 
            # target_modules=['embed_tokens', 'gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj'],
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.1
            )
        model = get_peft_model(model, peft_config)
        logger.info(f'Added LoRA, trainable parameters with LoRA only:')
        model.print_trainable_parameters()

    input_size = args.segment_length
    memory_size = 1
    n_segments = args.bptt_depth

    if args.baseline_only:
        logger.warning('training and evaluating only the backbone. remember to align the segment rightward')
        memory_size = 0
        n_segments = 2

    batch_size = args.batch_size

    block_size = input_size
    block_size -= 2 * memory_size
    block_size -= args.num_sensory
    history_size = (n_segments - 1) * block_size

    mask_size = block_size

    block_size_2 = input_size - (2*memory_size) - args.num_sensory//2

    """### Prepare dataset"""

    def interleaving_sample(examples, context_len):
        interleave = {}
        for k in examples.keys():
            interleave[k] = []
            for i in range(0, len(examples[k]), 2):
                first = examples[k][i]
                if i+1 >= len(examples[k]):
                    interleave[k].append(first)
                    break
                second = examples[k][i+1]

                res = []
                j = 0
                while j < len(first) and j < len(second):
                    res.extend(first[j:j+context_len]) 
                    res.extend(second[j:j+context_len])
                    j+=context_len
                if j < len(first):
                    res.extend(first[j:])
                if j < len(second):
                    res.extend(second[j:])
                interleave[k].append(res)

        return interleave

    def dilated_sample(examples, insert_len, period, insert_str):
        res = {}
        tok = tokenizer(insert_str)['input_ids'][1]
        attn_mask = tokenizer(insert_str)['attention_mask'][1]
        for k in examples.keys():
            res[k] = []
            for sample in examples[k]:
                ans = []
                i = 0
                while i < len(sample):
                    ans.extend(sample[i:i+period])
                    if k == 'input_ids':
                        ans.extend(insert_len * [tok]) #padding token [double space]
                    else:
                        ans.extend(insert_len * [attn_mask])
                    i+=period
                res[k].append(ans)

        return res


    def group_texts(examples, block_size, history_size=None):
        if args.interleave_dataset:
            examples = interleaving_sample(examples, args.interleave_len)
        elif args.dilate_dataset:
            examples = dilated_sample(examples, args.dilate_len, args.dilate_len, args.dilate_str)
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
    # Log the step
    logger.info("Preparing datasets and dataloaders")
    task_name = args.task_subset
    if args.train_set_split is not None:
        train_ds = datasets.load_dataset(args.task_name, task_name, split='train', streaming=args.streaming, trust_remote_code=True)
        valid_ds = datasets.load_dataset(args.task_name, task_name, split='validation', streaming=args.streaming)
        test_ds = datasets.load_dataset(args.task_name, task_name, split='test', streaming=args.streaming)
        if args.shuffle_train:
            train_ds = train_ds.shuffle(seed=args.seed).take(int(args.train_set_split))
        else:
            train_ds = train_ds.take(int(args.train_set_split))
        valid_ds = valid_ds.take(int(args.train_set_split))
        test_ds = test_ds.take(int(args.train_set_split))
        train_ds = datasets.Dataset.from_generator(partial(gen_from_iterable_dataset, train_ds), features=train_ds.features)
        valid_ds = datasets.Dataset.from_generator(partial(gen_from_iterable_dataset, valid_ds), features=valid_ds.features)
        test_ds = datasets.Dataset.from_generator(partial(gen_from_iterable_dataset, test_ds), features=test_ds.features)
    else:
        if args.task_name == 'pubmed_qa':
            # split train set into three subsets
            train_ds, valid_ds, test_ds = datasets.load_dataset(args.task_name, task_name, split=['train[:75%]', 'train[75%:90%]', 'train[90%:]'])
        elif args.task_name == 'suolyer/pile_arxiv':
            valid_ds, test_ds = datasets.load_dataset(args.task_name, task_name, split=['validation', 'test'])
        elif args.task_name == 'eda_corpus':
            full_ds = datasets.load_dataset("json", data_files="/home/jovyan/workspace/RAG-EDA/training_dataset/generator_dataset/eda_corpus_pretrain.jsonl")
            full_ds = full_ds["train"]
            split_ds = full_ds.train_test_split(test_size=0.05, seed=42)
            train_ds = split_ds['train']
            valid_ds = split_ds['test']
            test_ds = valid_ds
        elif args.task_name == 'eda_qa':
            pass
        else:
            train_ds, valid_ds, test_ds = datasets.load_dataset(args.task_name, task_name, split=['train', 'validation', 'test'], trust_remote_code=True)

    if args.task_name == 'pubmed_qa':
        logger.info("Preprocessing PubMedQA dataset")
        # preprocess qa database
        train_dataloader = PubMedQA(train_ds, tokenizer, fuse_size=2, batch_size=batch_size, shuffle=args.shuffle, seed=args.seed)
        valid_dataloader = PubMedQA(valid_ds, tokenizer, fuse_size=2, batch_size=batch_size)
        test_dataloader = PubMedQA(test_ds, tokenizer, fuse_size=args.fuse_size, batch_size=batch_size)
    elif args.task_name == 'eda_qa':
        logger.info("Preprocessing OpenROAD QA dataset")
        train_dataloader, valid_dataloader = OpenROAD(tokenizer, batch_size=batch_size, max_len=args.bptt_depth*block_size, mode='hard', neg_sample=12)
        test_dataloader = valid_dataloader
    else:
        logger.info("Preprocessing other datasets")
        column_names = valid_ds.column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])

        if args.task_name != 'suolyer/pile_arxiv':
            train_ds_tok = train_ds.map(
                tokenize_function,
                batched=True,
                batch_size=4,
                remove_columns=column_names,
                desc="Running tokenizer on training dataset",
                num_proc=8
            )

        valid_ds_tok = valid_ds.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on valid dataset",
            num_proc=2
        )

        test_ds_tok = test_ds.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on test dataset",
            num_proc=2
        )

        if args.task_name != 'suolyer/pile_arxiv':
            train_dataset = train_ds_tok.map(lambda x: group_texts(x, history_size, block_size),
                                                                    batched=True, desc=f"Grouping train in chunks of {block_size} and history {history_size}")
        valid_dataset = valid_ds_tok.map(lambda x: group_texts(x, history_size, block_size),
                                                                batched=True, desc=f"Grouping valid in chunks of {block_size}")
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                                collate_fn=collate_fn, shuffle=False, drop_last=True, pin_memory=True)
        if args.task_name != 'suolyer/pile_arxiv':
            train_rnd_generator = torch.Generator()
            train_rnd_generator.manual_seed(args.seed)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                            shuffle=args.shuffle, drop_last=False, generator=train_rnd_generator, pin_memory=True)
        else:
            train_dataloader = valid_dataloader
        
        if args.task_name != 'suolyer/pile_arxiv':
            test_dataset = test_ds_tok.map(lambda x: group_texts(x, args.test_length, block_size),
                                                                    batched=True, desc=f"Grouping test in chunks of {block_size}")
        else:
            test_dataset = test_ds_tok
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                                collate_fn=collate_fn, shuffle=False, drop_last=True, pin_memory=True)

    logger.info("Preparing memory cell")
    if args.rmt_only or args.baseline_only:
        cell = MemoryCell(model, 
                    num_mem_tokens=memory_size,
                    num_prepend=0)

        model = RecurrentWrapper(cell,
                                segment_size=block_size,
                                max_n_segments=n_segments,
                                mask_size=mask_size,
                                n_cell_out=args.num_seg_save,
                                segment_alignment=args.segment_alignment
                                )
    else:
        cell = MemoryCell(model, 
                        num_mem_tokens=memory_size,
                        num_prepend=args.num_sensory)

        if args.hmt_stage_1:
            model = RecurrentWrapper(cell,
                                segment_size=block_size,
                                max_n_segments=n_segments,
                                mask_size=mask_size,
                                n_cell_out=args.num_seg_save,
                                segment_alignment=args.segment_alignment
                                )
        else:
            if args.load_from_ckpt is not None and args.hmt_stage_2:
                ori_model = RecurrentWrapper(cell,
                                segment_size=block_size,
                                max_n_segments=n_segments,
                                mask_size=mask_size,
                                n_cell_out=args.num_seg_save,
                                segment_alignment=args.segment_alignment
                                )
                state_dict = get_fp32_state_dict_from_zero_checkpoint(args.load_from_ckpt)
                ori_model.load_state_dict(state_dict)
                cell = copy.deepcopy(ori_model.memory_cell)

            model = RecurrentWrapper(cell,
                                emb=copy.deepcopy(model.get_input_embeddings()),
                                word_emb_dim=word_emb_dim,
                                hidden_dim=args.mem_recall_hidden_dim,
                                ltm_context=args.mem_recall_context,
                                segment_size=block_size,
                                max_n_segments=n_segments,
                                mask_size=mask_size,
                                n_cell_out=args.num_seg_save,
                                segment_alignment=args.segment_alignment
                                )
            
    if args.load_from_ckpt is not None and not args.hmt_stage_2:
        state_dict = get_fp32_state_dict_from_zero_checkpoint(args.load_from_ckpt)
        model.load_state_dict(state_dict)

    logger.info("Preparing optimizer")
    from torch.optim import AdamW
    optimizer_cls = (
        AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )

    optim = optimizer_cls(model.parameters(), lr=args.learning_rate)
    from torch.optim.lr_scheduler import StepLR, LambdaLR

    if (
     accelerator.state.deepspeed_plugin is None
     or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        if args.lr_decay:
            if not args.dynamic:
                scheduler = StepLR(optim, step_size=100, gamma=args.lr_decay_gamma)
            else:
                lambda_f = lambda epoch: (args.lr_decay_gamma ** (epoch // 100)) * 0.1 if epoch%2==0 else (args.lr_decay_gamma ** (epoch // 100))
                scheduler = LambdaLR(optim, lr_lambda=lambda_f)
        else:
            scheduler = StepLR(optim, step_size=100, gamma=1.0)
    else:
        scheduler = DummyScheduler(
            optim, total_num_steps=args.training_step, num_training_steps=100
        )

    train_steps = args.training_step
    eval_steps = args.eval_step


    logger.info("Preparing accelerator")
    # wrap with accelerate
    model, optim, train_dataloader, valid_dataloader, scheduler = accelerator.prepare(
        model, optim, train_dataloader, valid_dataloader, scheduler
    )

    logger.info("Preparing generators")
    train_gen = iter(train_dataloader)
    valid_gen = iter(valid_dataloader)

    logger.info("Moving model to device")
    model.to(device)

    logger.info("Setting model to train mode")
    model.train()

    block_size_list = [block_size, block_size_2]

    if args.train_memory_map:
        # freeze all params
        for n, p in model.named_parameters():
            if 'mem_map' not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True


    if not args.inference_only:
        logger.info("Starting training")
        losses = []
        for epoch in range(args.num_epochs):
            train_gen = iter(train_dataloader)
            total_len = min(train_steps, len(train_dataloader))
            for step in tqdm.tqdm(range(min(train_steps, len(train_dataloader)))):
                optim.zero_grad()

                batch = next(train_gen)
                if args.dynamic:
                    batch['segment_size'] = block_size
                    batch['extra_size'] = args.num_sensory//2
                    batch['mode'] = 'test'
                    out, _ = model(**batch)
                    loss = out.loss

                    accelerator.backward(loss)
                    optim.step()
                    if args.lr_decay:
                        scheduler.step()
                    # logger.debug(f'loss: {loss.item()}')
                    # logger.debug(f'ppl: {out.ppl.item()}')
                    losses.append(loss.detach().item())

                elif args.train_memory_map:
                    for i in range(args.bptt_depth-1):
                        optim.zero_grad()
                        batch['segment_size'] = block_size
                        batch['extra_size'] = args.num_sensory//2
                        batch['switch_at'] = i
                        out, _ = model(**batch)
                        loss = out.loss
                        # need to use this for loss
                        accelerator.backward(loss)
                        optim.step()
                        if args.lr_decay:
                            scheduler.step()
                        # logger.debug(f'loss: {loss.item()}')
                        # logger.debug(f'ppl: {out.ppl.item()}')
                        losses.append(loss.detach().item())
                
                else:
                    batch['segment_size'] = block_size
                    batch['sum_fraction'] = args.sum_fraction
                    if args.task_name == 'eda_qa':
                        batch['mask_size'] = batch['answer_len'][0]
                    out, _ = model(**batch)
                    loss = out.loss
                    accelerator.backward(loss)
                    optim.step()
                    if args.lr_decay:
                        scheduler.step()
                    losses.append(loss.detach().item())
                    accelerator.log({"train loss": loss.detach().item(), "train ppl": out.ppl.detach().item()}, step=step+total_len*epoch)
                
                if step % 50 == 0:
                    # evaluate
                    model.eval()
                    sub_valid_gen = iter(valid_dataloader)
                    eval_losses = []
                    eval_ppl = []
                    for eval_step in range(10):
                        eval_batch = next(sub_valid_gen)
                        if args.task_name == 'eda_qa':
                            batch['mask_size'] = batch['answer_len'][0]
                        with torch.no_grad():
                            out, _ = model(**batch)
                        eval_losses.append(out.loss.detach().item())
                        eval_ppl.append(out.ppl.detach().item())
                    accelerator.log({"eval loss": np.mean(eval_losses), "eval ppl": np.mean(eval_ppl)}, step=step+total_len*epoch)


        accelerator.wait_for_everyone()
        if args.save_ckpt is not None:
            model.save_checkpoint(args.save_ckpt)

    valid_losses = []
    valid_ppl = []
    model.eval()
    for step in tqdm.tqdm(range(eval_steps)):
        batch = next(valid_gen)
        batch['segment_size'] = block_size
        if args.task_name == 'eda_qa':
            batch['mask_size'] = batch['answer_len'][0]
        # if args.timing:
        #     batch['prof'] = True
        with torch.no_grad():
            out, _ = model(**batch)
        loss = out.loss
        ppl = out.ppl
        logger.debug(f'loss: {loss.item()}')
        logger.debug(f'ppl: {ppl.item()}')
        valid_losses.append(loss.detach().item())
        valid_ppl.append(ppl.detach().item())

    print(f'Loss on {eval_steps * batch_size} validation samples (CrossEntropy): {np.mean(valid_losses)}')
    print(f'PPL on {eval_steps * batch_size} validation samples: {np.mean(valid_ppl)}')

    test_losses = []
    test_ppl = []
    total_hist = []

    test_gen = iter(test_dataloader)

    for step in tqdm.tqdm(range(args.test_step)):
        batch = next(test_gen)
        batch['segment_size'] = block_size
        if args.task_name == 'eda_qa':
            batch['mask_size'] = batch['answer_len'][0]
        if args.dynamic:
            batch['extra_size'] = args.num_sensory//2
            batch['mode'] = 'test'
        if args.timing:
            batch['prof'] = True
        with torch.no_grad():
            out, hist = model(**batch)
        loss = out.loss
        ppl = out.ppl
        test_losses.append(loss.detach().item())
        test_ppl.append(ppl.detach().item())
        logger.info(f'loss: {loss.item()}')
        if hist is not None:
            total_hist.extend(hist)
    
    if (args.baseline_only == False) and (args.rmt_only == False) and (args.hmt_stage_1 == False) and args.plot_hist:
        max_d = np.max(total_hist)
        plt.hist(total_hist, weights=np.ones(len(total_hist))/len(total_hist), bins=50)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel("Context Distance")
        plt.ylabel("Probability")
        plt.savefig('artifact/heatmap_' + date_str + '.png')
        plt.show()

    print(f'PPL on {args.test_step * batch_size} test samples: {np.mean(test_ppl)}')

    if args.generate is not None and device == torch.device('cuda:0'):
        with open(args.generate, 'r') as f:
            prompt_text = f.read()

        encoded_prompt = tokenizer(prompt_text, return_tensors="pt")
        output_seq = model.generate(
            input_ids = encoded_prompt.input_ids.cpu(),
            attention_mask = encoded_prompt.attention_mask.cpu(),
            segment_size = block_size,
            max_new_tokens = 100,
            temperature = 0.6
        )
        print(tokenizer.batch_decode(output_seq, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
    
    
    if args.task_name == 'eda_qa':
        import evaluate
        rouge = evaluate.load('rouge')

        with open('RAG-EDA/benchmark/openroad_documentation.json', 'r') as f:
            corpus_dict = json.load(f)
        
        content = []
        for topic in corpus_dict:
            for knowledge in topic['knowledge']:
                content.append(knowledge['content'])
        
        content_str = " ".join(content)

        ORD_QA_sample = []
        with open('RAG-EDA/benchmark/ORD-QA.jsonl') as file:
            for line in file:
                if line.strip():  # Skip any empty lines
                    ORD_QA_sample.append(json.loads(line))

        rougeL_full = []
        for step in tqdm.tqdm(range(len(ORD_QA_sample))):
            entry = ORD_QA_sample[step]
            question_str = entry['question']
            answer_str = entry['answer']
            messages = [
                {"role": "system", "content": "You are an expert with EDA tool usage. Answer the question based on the following reference information."},
                {"role": "system", "content": content_str},
                {"role": "user", "content": question_str}
            ]
            message_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tok_message = tokenizer(message_str, return_tensors='pt')
            tok_answer = tokenizer.encode(answer_str)
            with torch.no_grad():
                output_seq = model.generate(
                    input_ids = tok_message['input_ids'],
                    attention_mask = tok_message['attention_mask'],
                    segment_size = block_size,
                    max_new_tokens = len(tok_answer)
                )
            predictions = tokenizer.batch_decode(output_seq, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            references = [answer_str]
            results = rouge.compute(predictions=predictions, references=references)
            rougeL_full.append(results['rougeL'])
        print(f'ROUGE-L on {len(ORD_QA_sample)} test samples using whole database: {np.mean(rougeL_full)}')


        test_dataloader = OpenROAD_test(tokenizer, batch_size=batch_size)
        test_gen = iter(test_dataloader)

        rougeL = []

        for step in tqdm.tqdm(range(len(test_dataloader))):
            batch = next(test_gen)
            batch['segment_size'] = block_size
            output_seq = model.generate(
                input_ids = batch['input_ids'],
                attention_mask = batch['attention_mask'],
                segment_size = block_size,
                max_new_tokens = batch['answer_len'][0]
            )
            predictions = tokenizer.batch_decode(output_seq, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            references = batch['answer']
            results = rouge.compute(predictions=predictions, references=references)
            rougeL.append(results['rougeL'])
        
        print(f'ROUGE-L on {len(test_dataloader)} test samples with only correct reference: {np.mean(rougeL)}')
            

if __name__ == "__main__":
    main()
