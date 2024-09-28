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

from tools.data_processing.hmt_qa_datasets import load_qa_dataset
from tools.collate import hmt_collate_fn

parser = ArgumentParser()

qa_tasks = {'deepmind/narrativeqa'}

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
parser.add_argument('--validation_step', type=int, default=10, help='Number of validation steps to perform at each validation interval')
parser.add_argument('--curriculum', action='store_true', default=False, help='use curriculum learning')
parser.add_argument('--curriculum_segs', type=str, default=None, help='Comma-separated list of curriculum levels (number of segments for each level)')
parser.add_argument('--wandb_project', type=str, default='redpajama_curriculum', help='Name for the WanDB Project')
parser.add_argument('--wandb_run', type=str, default=None, help='Name for the WanDB run')
parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity (username or team name)')
parser.add_argument('--max_context_length', type=int, default=None, help='Maximum context length for the dataset. If None, no limit is applied.')
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
        config={"dropout": 0.1, "learning_rate": 1e-5, "task_name": args.task_name, "model_name": args.model_name, "test_length": args.test_length, "checkpoint": os.path.basename(args.load_from_ckpt)},
        init_kwargs={"wandb": {"entity": args.wandb_entity, "name": f'{args.wandb_run}'}}
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

    id_pad_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    collate_fn = partial(hmt_collate_fn, id_pad_value=id_pad_value, is_qa_task=args.task_name in qa_tasks, block_size=block_size, batch_size=batch_size)
    
    # Log the step
    logger.info("Loading datasets")
    if args.task_name == 'deepmind/narrativeqa':
        test_ds = load_qa_dataset('deepmind/narrativeqa', split='test[:' + str(args.test_step) + ']', streaming=args.streaming, trust_remote_code=True)
    elif args.task_name == 'togethercomputer/RedPajama-Data-V2':
        from tools.data_processing.red_pajamav2 import load_redpajama
        test_ds = load_redpajama(tokenizer=tokenizer, split='train[90%:]', history_size=args.test_length, block_size=block_size, streaming=args.streaming, trust_remote_code=True)
    elif args.task_name == 'qmsum':
        from tools.data_processing.qmsum import load_qmsum_test
        test_ds = load_qmsum_test(max_token_num=args.max_context_length, test_length=args.test_length, block_size=block_size, tokenizer=tokenizer, split='test')
    else:
        raise NotImplementedError(f"Task {args.task_name} is not supported")

    # Print dataset sizes
    print(f"Test dataset size: {len(test_ds)}")

    logger.info(f"Creating dataloaders")
    # Create dataloaders
    test_dataloader = DataLoader(test_ds, batch_size=batch_size,
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
                                segment_alignment=args.segment_alignment)
                
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
                                segment_alignment=args.segment_alignment,
                                is_qa_task=args.task_name in qa_tasks
                                )
            
            if args.load_from_ckpt is not None and not args.hmt_stage_2:
                # checkpoint_dir = os.path.dirname(args.load_from_ckpt)
                state_dict = torch.load(args.load_from_ckpt,map_location='cuda:0')
                # Remove the 'module.' prefix from all state dict keys
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  # FIXME: while saving the state dicts while the model is wrapped, all the weights have a model. prefix and needed to be removed when loading. We should fix this by unwrapping before saving. 
                # if args.model_name == 'openlm-research/open_llama_3b_v2':
                #     new_state_dict = {k.replace('base_model.model.', ''): v for k, v in new_state_dict.items()}
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


    logger.info("Preparing accelerator")
    model, optim, test_dataloader, scheduler = accelerator.prepare(
        model, optim, test_dataloader, scheduler
    )

    logger.info("Moving model to device")
    model.to(device)

    if args.train_memory_map:
        # freeze all params
        for n, p in model.named_parameters():
            if 'mem_map' not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

    model.eval()

    test_losses = []
    test_ppl = []
    total_hist = []

    test_gen = iter(test_dataloader)

    # Start Testing
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
        f1 = out.f1['f1']
        accelerator.log({"Test CrossEntropy Loss": loss.item(), "Test PPL": ppl.item(), "Test F1": f1.item()}, step=step)
        test_losses.append(loss.detach().item())
        test_ppl.append(ppl.detach().item())
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
