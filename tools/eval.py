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
from modeling_rmt.compression import inject_eae
from typing import List
import logging, shutil
from accelerate.logging import get_logger

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
parser.add_argument('--save_interval', type=int, default=0, help='Save checkpoint every N steps. 0 means no intermediate saving.')
parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
parser.add_argument('--validation_interval', type=int, default=100, help='Perform validation every N steps')
parser.add_argument('--validation_steps', type=int, default=10, help='Number of validation steps to perform at each validation interval')
parser.add_argument('--curriculum', action='store_true', default=False, help='use curriculum learning')
parser.add_argument('--curriculum_segs', type=str, default=None, help='Comma-separated list of curriculum levels (number of segments for each level)')
parser.add_argument('--wandb_project', type=str, default='redpajama_curriculum', help='Name for the WanDB Project')
parser.add_argument('--wandb_run', type=str, default=None, help='Name for the WanDB run')
parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity (username or team name)')
parser.add_argument('--recache_splits', type=str, default=None, help='Provide a list of dataset splits that to rebuild the tokenization and grouping cache')

torch.manual_seed(3407)

def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


def main():
    global torch

    args = parser.parse_args()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], log_with='wandb')
    device = accelerator.device

    # Initialize WanDB Tracker
    accelerator.init_trackers(
        project_name=args.wandb_project, 
        config={"dropout": 0.1, "learning_rate": 1e-5},
        init_kwargs={"wandb": {"entity": args.wandb_entity, "name": args.wandb_run}}
    )

    recache_splits = args.recache_splits.split(',') if args.recache_splits else None
    print("recache splits: ", recache_splits)

    token=None
    if args.token_file is not None:
        with open(args.token_file, 'r') as f:
            token = f.read()

    """### Load model"""
    cache_dir = os.environ.get('HF_HOME', '.')
    model = AutoModelForCausalLM.from_pretrained(args.model_name, token=token, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=token, cache_dir=cache_dir)

    if isinstance(model.config, OPTConfig):
        word_emb_dim = model.config.word_embed_proj_dim
    else:
        word_emb_dim = model.config.hidden_size


    curriculum = args.curriculum
    if curriculum and args.curriculum_segs is None:
        raise ValueError("curriculum_segs must be provided if curriculum is True")

    levels = [int(level) for level in args.curriculum_segs.split(',')]

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

    # batch_size, block_size, history_size, mask_size, block_size_2, n_segments
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
    logger.info("Loading datasets")
    task_name = args.task_subset
    if args.train_set_split is not None:
        # remember to change the sizes
        # train_ds, valid_ds, test_ds = datasets.load_dataset(args.task_name, task_name, split=['train[:75%]', 'train[75%:90%]', 'train[90%:]'], streaming=args.streaming, trust_remote_code=True)
        train_ds, valid_ds, test_ds = datasets.load_dataset(args.task_name, task_name, split=['train[:75%]', 'train[75%:90%]', 'train[90%:94%]'], streaming=args.streaming, trust_remote_code=True)
    
        # train_ds = train_ds.shuffle(seed=args.seed, buffer_size=20000).take(int(args.train_set_split))
        # valid_ds = valid_ds.take(int(args.train_set_split))
        # test_ds = test_ds.take(int(args.train_set_split))
        train_ds = datasets.Dataset.from_generator(partial(gen_from_iterable_dataset, train_ds), features=train_ds.features)
        valid_ds = datasets.Dataset.from_generator(partial(gen_from_iterable_dataset, valid_ds), features=valid_ds.features)
        test_ds = datasets.Dataset.from_generator(partial(gen_from_iterable_dataset, test_ds), features=test_ds.features)

        # Print dataset sizes
        print(f"Train dataset size: {len(train_ds)}")
        print(f"Validation dataset size: {len(valid_ds)}")
        print(f"Test dataset size: {len(test_ds)}")
    else:
        train_ds, valid_ds, test_ds = datasets.load_dataset(args.task_name, task_name, split=['train', 'validation', 'test'], trust_remote_code=True)

    logger.info(f"Tokenizing datasets")
    column_names = valid_ds.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    # Define the cache directory for tokenized datasets
    tokenized_datasets_dir = os.path.join(cache_dir, "tokenized")
    if not os.path.exists(tokenized_datasets_dir):
        os.makedirs(tokenized_datasets_dir)

    # Function to load or create tokenized dataset
    def load_or_create_tokenized_dataset(dataset, split, recache_splits: List[str]=None):
        cache_path = os.path.join(tokenized_datasets_dir, f"{args.task_name}_{split}.hf")
        print(recache_splits)
        if recache_splits is not None and split in recache_splits:
            if os.path.exists(cache_path):
                logger.info(f"Deleting existing tokenized {split} dataset cache to recache")
                shutil.rmtree(cache_path)

        if os.path.exists(cache_path):
            logger.info(f"Loading tokenized {split} dataset from cache")
            return datasets.load_from_disk(cache_path)
        else:
            logger.info(f"Tokenizing {split} dataset")
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                batch_size=4,
                remove_columns=column_names,
                desc=f"Running tokenizer on {split} dataset",
                num_proc=8
            )
            logger.info(f"Saving tokenized {split} dataset to cache")
            tokenized_dataset.save_to_disk(cache_path)
            return tokenized_dataset

    # Load or create tokenized datasets
    train_ds_tok = load_or_create_tokenized_dataset(train_ds, "train")
    valid_ds_tok = load_or_create_tokenized_dataset(valid_ds, "valid")
    test_ds_tok = load_or_create_tokenized_dataset(test_ds, "test", recache_splits)

    logger.info(f"Creating dataloaders")

    # Define the cache directory for grouped datasets
    grouped_datasets_dir = os.path.join(os.environ.get('HF_HOME', ''), "grouped")
    if not os.path.exists(grouped_datasets_dir):
        os.makedirs(grouped_datasets_dir)

    # Function to load or create grouped dataset
    def load_or_create_grouped_dataset(dataset, split, history_size, block_size, levels: List[int]=None, recache_splits: List[str]=None):
        if levels is not None:
            grouped_datasets = []
            for i,n_segs in enumerate(levels):
                cache_path = os.path.join(grouped_datasets_dir, f"{args.task_name}_{split}_grouped_seg{n_segs}.hf")
                if recache_splits is not None and split in recache_splits:
                    if os.path.exists(cache_path):
                        logger.info(f"Deleting existing grouped {split} dataset cache to recache")
                        shutil.rmtree(cache_path)
                if os.path.exists(cache_path):
                    grouped_datasets.append(datasets.load_from_disk(cache_path))
                else:
                    num_data_per_level = len(dataset) // len(levels)
                    data_subset = dataset.select(range(i * num_data_per_level, (i + 1) * num_data_per_level))
                    curr_n_segments = n_segs
                    curr_history_size = (curr_n_segments - 1) * block_size

                    grouped_dataset = data_subset.map(
                        lambda x: group_texts(x, curr_history_size, block_size),
                        batched=True,
                        desc=f"Grouping {split} in chunks of {block_size}, {n_segs} segments, " + (f" and history {history_size}" if split == 'train' else ""),
                        num_proc=8
                    )
                    grouped_dataset.save_to_disk(cache_path)
                    grouped_datasets.append(grouped_dataset)
            return grouped_datasets
        else: 
            cache_path = os.path.join(grouped_datasets_dir, f"{args.task_name}_{split}_grouped.hf")
            if recache_splits is not None and split in recache_splits:
                if os.path.exists(cache_path):
                    logger.info(f"Deleting existing grouped {split} dataset cache to recache")
                    shutil.rmtree(cache_path)
            if os.path.exists(cache_path):
                logger.info(f"Loading grouped {split} dataset from cache")
                return datasets.load_from_disk(cache_path)
            else:
                logger.info(f"Grouping {split} dataset")
                grouped_dataset = dataset.map(
                    lambda x: group_texts(x, history_size, block_size),
                    batched=True,
                    desc=f"Grouping {split} in chunks of {block_size}" + (f" and history {history_size}" if split == 'train' else ""),
                    num_proc=8
                )
                logger.info(f"Saving grouped {split} dataset to cache")
                grouped_dataset.save_to_disk(cache_path)
                return grouped_dataset

    # Load or create grouped datasets
    if curriculum:
        valid_datasets = load_or_create_grouped_dataset(valid_ds_tok, "valid", history_size, block_size, levels=levels)
        train_datasets = load_or_create_grouped_dataset(train_ds_tok, "train", history_size, block_size, levels=levels)
        # FIXME: Can try test case of different lengths, 3k, 10k, 60k, 100k
        test_dataset = load_or_create_grouped_dataset(test_ds_tok, "test", args.test_length, block_size, recache_splits=recache_splits)
    else:
        valid_dataset = load_or_create_grouped_dataset(valid_ds_tok, "valid", history_size, block_size)
        train_dataset = load_or_create_grouped_dataset(train_ds_tok, "train", history_size, block_size)
        test_dataset = load_or_create_grouped_dataset(test_ds_tok, "test", args.test_length, block_size, recache_splits=recache_splits)

    # Create dataloaders
    if curriculum:
        valid_dataloaders = [DataLoader(valid_dataset, batch_size=batch_size,
                                    collate_fn=collate_fn, shuffle=args.shuffle, drop_last=True, pin_memory=True) for valid_dataset in valid_datasets]

        train_rnd_generator = torch.Generator()
        train_rnd_generator.manual_seed(args.seed)
        train_dataloaders = [DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                    shuffle=args.shuffle, drop_last=False, generator=train_rnd_generator, pin_memory=True) for train_dataset in train_datasets]

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                    collate_fn=collate_fn, shuffle=args.shuffle, drop_last=True, pin_memory=True)
    else:
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                    collate_fn=collate_fn, shuffle=args.shuffle, drop_last=True, pin_memory=True)

        train_rnd_generator = torch.Generator()
        train_rnd_generator.manual_seed(args.seed)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                    shuffle=args.shuffle, drop_last=False, generator=train_rnd_generator, pin_memory=True)

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                    collate_fn=collate_fn, shuffle=args.shuffle, drop_last=True, pin_memory=True)

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
                                max_n_segments=max(levels) if curriculum else n_segments,
                                mask_size=mask_size,
                                n_cell_out=args.num_seg_save,
                                segment_alignment=args.segment_alignment
                                )
        else:
            if args.load_from_ckpt is not None and args.hmt_stage_2:
                ori_model = RecurrentWrapper(cell,
                                segment_size=block_size,
                                max_n_segments=max(levels) if curriculum else n_segments,
                                mask_size=mask_size,
                                n_cell_out=args.num_seg_save,
                                segment_alignment=args.segment_alignment
                                )
                # state_dict = get_fp32_state_dict_from_zero_checkpoint(args.load_from_ckpt)
                # ori_model.load_state_dict(state_dict)
                state_dict = torch.load(args.load_from_ckpt,map_location='cuda:0')
                # Remove the 'module.' prefix from all state dict keys
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                state_dict = new_state_dict
                ori_model.load_state_dict(state_dict)
                cell = copy.deepcopy(ori_model.memory_cell)

            logger.info(f'Creating model with max_n_segments: {max(levels)}')
            model = RecurrentWrapper(cell,
                                emb=copy.deepcopy(model.get_input_embeddings()),
                                word_emb_dim=word_emb_dim,
                                hidden_dim=args.mem_recall_hidden_dim,
                                ltm_context=args.mem_recall_context,
                                segment_size=block_size,
                                max_n_segments=max(levels) if curriculum else n_segments,
                                mask_size=mask_size,
                                n_cell_out=args.num_seg_save,
                                segment_alignment=args.segment_alignment
                                )
            
            if args.load_from_ckpt is not None and not args.hmt_stage_2:
                # checkpoint_dir = os.path.dirname(args.load_from_ckpt)
                state_dict = torch.load(args.load_from_ckpt,map_location='cuda:0')
                # Remove the 'module.' prefix from all state dict keys
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  # FIXME: while saving the state dicts while the model is wrapped, all the weights have a model. prefix and needed to be removed when loading. We should fix this by unwrapping before saving. 
                state_dict = new_state_dict
                model.load_state_dict(state_dict)
                # tag = os.path.basename(args.load_from_ckpt)
                # state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, 'opt-350m')
                # model.load_state_dict(state_dict)

    logger.info("Preparing optimizer")
    from torch.optim import AdamW
    optim = AdamW(params=model.parameters(), lr=args.learning_rate)
    from torch.optim.lr_scheduler import StepLR, LambdaLR
    if args.lr_decay:
        if args.dynamic:
            scheduler = StepLR(optim, step_size=100, gamma=args.lr_decay_gamma)
        else:
            lambda_f = lambda epoch: (args.lr_decay_gamma ** (epoch // 100)) * 0.1 if epoch%2==0 else (args.lr_decay_gamma ** (epoch // 100))
            scheduler = LambdaLR(optim, lr_lambda=lambda_f)
    else:
        scheduler = StepLR(optim, step_size=100, gamma=1.0)

    train_steps = args.training_step
    eval_steps = args.eval_step

    logger.info("Preparing accelerator")
    # wrap with accelerate
    if not curriculum:
        model, optim, train_dataloader, valid_dataloader, scheduler = accelerator.prepare(
            model, optim, train_dataloader, valid_dataloader, scheduler
        )
    else:
        all_dataloaders = train_dataloaders + valid_dataloaders
        model, optim, *all_dataloaders, scheduler = accelerator.prepare(
            model, optim, *all_dataloaders, scheduler
        )
        train_dataloaders = all_dataloaders[:len(train_dataloaders)]
        valid_dataloaders = all_dataloaders[len(train_dataloaders):]

    logger.info("Preparing generators")
    if curriculum:
        train_gens = [iter(train_dataloader) for train_dataloader in train_dataloaders]
        valid_gens = [iter(valid_dataloader) for valid_dataloader in valid_dataloaders]
    else:
        train_gen = iter(train_dataloader)
        valid_gen = iter(valid_dataloader)

    logger.info("Moving model to device")
    model.to(device)

    if args.train_memory_map:
        # freeze all params
        for n, p in model.named_parameters():
            if 'mem_map' not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

    # valid_losses = []
    # valid_ppl = []
    model.eval()
    # for step in tqdm.tqdm(range(eval_steps)):
    #     batch = next(valid_gen)
    #     for k, v in batch.items():
    #         batch[k] = v.cpu()
    #     batch['segment_size'] = block_size
    #     # if args.timing:
    #     #     batch['prof'] = True
    #     with torch.no_grad():
    #         out, _ = model(**batch)
    #     loss = out.loss
    #     ppl = out.ppl
    #     logger.debug(f'loss: {loss.item()}')
    #     logger.debug(f'ppl: {ppl.item()}')
    #     valid_losses.append(loss.detach().item())
    #     valid_ppl.append(ppl.detach().item())

    # print(f'Loss on {eval_steps * batch_size} validation samples (CrossEntropy): {np.mean(valid_losses)}')
    # print(f'PPL on {eval_steps * batch_size} validation samples: {np.mean(valid_ppl)}')

    test_losses = []
    test_ppl = []
    total_hist = []

    test_gen = iter(test_dataloader)

    for step in tqdm.tqdm(range(args.test_step)):
        batch = next(test_gen)
        for k, v in batch.items():
            batch[k] = v.cpu()
        batch['segment_size'] = block_size
        if args.dynamic:
            batch['extra_size'] = args.num_sensory//2
            batch['mode'] = 'test'
        if args.timing:
            batch['prof'] = True
        with torch.no_grad():
            out, hist = model(**batch)
        loss = out.loss
        ppl = out.ppl
        accelerator.log({"Test CrossEntropy Loss": loss.item(), "Test PPL": ppl.item(), }, step=step)
        test_losses.append(loss.detach().item())
        test_ppl.append(ppl.detach().item())
        # logger.info(f'loss: {loss.item()}')
        if hist is not None:
            total_hist.extend(hist)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
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
    
    accelerator.end_training()

if __name__ == "__main__":
    main()
