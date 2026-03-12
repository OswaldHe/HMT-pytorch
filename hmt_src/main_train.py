import numpy as np
import os
import sys
import tqdm
import torch
import json
import logging
import math
import accelerate
import datetime
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTConfig
from accelerate import Accelerator, DistributedDataParallelKwargs
from pathlib import Path
from peft import get_peft_model, LoraConfig, TaskType
from hmt_src.data import generate_dataloaders
from hmt_src.data.openroad_qa_preprocess import OpenROAD_test
from hmt_src.model import generate_model
from accelerate.utils import DummyOptim, DummyScheduler

from hmt_src.utils import apply_chat_template_with_fallback

print("train model.")
 
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

# cli arguments
# random seed
parser.add_argument('--seed', type=int, default=3407, help='random seed for training')

# dataset settings
parser.add_argument('--task_name', type=str, default='wikitext', help='training/validation task name (e.g. wikitext, pg19, samsum, etc.)')
parser.add_argument('--task_subset', type=str, default=None, help='subset of dataset (e.g., wikitext-2-v1)')
parser.add_argument('--token_file', type=str, default=None, help='path to the file with Huggingface token. Used for gated model such as Llama2.')
parser.add_argument('--streaming', action='store_true', default=False, help='generate text in streaming mode')
parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle the dataset')
parser.add_argument('--shuffle_train', action='store_true', default=True, help='shuffle the training dataset')
parser.add_argument('--cache_dir', type=str, default='.', help='cache directory, default to the current directory')
parser.add_argument('--fuse_size', type=int, default=2, help='the number of questions and context to fuse for PubMedQA dataset')

# training and evaluation settings
parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs to train the model')
parser.add_argument('--batch_size', type=int, default=2, help='number of batches per device')
parser.add_argument('--training_step', type=int, default=500, help='number of training steps')
parser.add_argument('--eval_step', type=int, default=100, help='number of evaluation steps')
parser.add_argument('--test_step', type=int, default=100, help='number of testing steps')
parser.add_argument('--train_set_split', type=str, default=None, 
        help='slice upper bound of training set to reduce time for tokenization. use percentage notation (e.g., 2%), or integer')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='training learning rate')
parser.add_argument('--lr_decay', action='store_true', default=False, help='whether having learning rate decay or not')
parser.add_argument('--lr_decay_gamma', type=float, default=0.8, help='rate of lr decay')
parser.add_argument('--use_lora', action='store_true', default=False, help='whether use PEFT LoRA to speed up training')

# HMT model settings
parser.add_argument('--save_ckpt', type=str, default=None, help='store the model checkpoint to the specified directory, only used for HMT')
parser.add_argument('--load_from_ckpt', type=str, default=None, help='load the checkpoint for HMT')

parser.add_argument('--recurrent_type', type=str, default='summary_memory', choices=['summary_memory', 'memory_only'], help='choose recurrent wrapper type')
parser.add_argument('--model_name', type=str, default='facebook/opt-2.7b', help='transformer model name for backbone of HMT')
parser.add_argument('--segment_length', type=int, default=1024, help='segment length of HMT')
parser.add_argument('--dynamic_seg', action='store_true', default=False, help='use frozen BERT segmentation model for dynamic context segmentation')
parser.add_argument('--dynamic_seg_checkpoint', type=str, default=None, help='path to trained segmentation checkpoint used by --dynamic_seg')
parser.add_argument('--num_seg_save', type=int, default=4, help='max number of segment inference results saved on GPU')
parser.add_argument('--bptt_depth', type=int, default=8, help='number of segments unrolled in bptt')
parser.add_argument('--test_max_context_length', type=int, default=None, help='max context length of input to test')

parser.add_argument('--sum_fraction', type=float, default=0.5, help='fraction of the segment that will be used for representation extraction')
parser.add_argument('--num_sensory', type=int, default=0, help='number of preserved tokens for sensory memory')
parser.add_argument('--mem_hidden_dim', type=int, default=4096, help='hidden dimension of cross attention in memory recall mech.')
parser.add_argument('--mem_recall_size', type=int, default=1, help='number of memory embeddings to be concanated with segment.')
parser.add_argument('--mem_window_size', type=int, default=256, help='number of memory embeddings cached in memory recall mech.')
parser.add_argument('--mem_mlp', action='store_true', default=False, help='use an MLP to process memory_state in Memory-Only wrapper')
parser.add_argument('--mem_mlp_hidden_dim', type=int, default=None, help='hidden dimension of memory_state MLP (default: model hidden size)')

parser.add_argument('--rmt_only', action='store_true', default=False, help='train and evaluate with only rmt')
parser.add_argument('--baseline_only', action='store_true', default=False, help='train and evaluate only the backbone model')

parser.add_argument('--plot_hist', action='store_true', default=False, help='show memory recall context histogram.')
parser.add_argument('--timing', action='store_true', default=False, help='profile the timing of inference.')

# wandb settings
parser.add_argument('--wandb_project', type=str, default=None, help='Name for the WanDB Project')
parser.add_argument('--wandb_run', type=str, default=None, help='Name for the WanDB run')
parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity (username or team name)')

torch.manual_seed(3407)

def main():
    global torch

    args = parser.parse_args()
    if args.dynamic_seg and not args.dynamic_seg_checkpoint:
        raise ValueError("--dynamic_seg requires --dynamic_seg_checkpoint")
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], log_with="wandb")
    device = accelerator.device
    from accelerate.logging import get_logger
    logger = get_logger('')

    def mean_metric(value):
        if value is None:
            return None
        if not torch.is_tensor(value):
            value = torch.tensor(value, device=accelerator.device)
        else:
            value = value.detach()
            if value.device != accelerator.device:
                value = value.to(accelerator.device)
            if value.is_sparse:
                value = value.to_dense()
        if value.dim() == 0:
            value = value.unsqueeze(0)
        gathered = accelerator.gather_for_metrics(value)
        return gathered.float().mean().item()

    # Initialize WanDB Tracker
    accelerator.init_trackers(
        project_name=args.wandb_project, 
        config={"dropout": 0.1, 
                "learning_rate": args.learning_rate, 
                "model_name": args.model_name,
                "task_name": args.task_name, 
                "test_max_context_length": args.test_max_context_length},
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
    
    

    if args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False, 
            target_modules=['embed_tokens', 'gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj'],
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.1
            )
        model = get_peft_model(model, peft_config)
        logger.info(f'Added LoRA, trainable parameters with LoRA only:')
        model.print_trainable_parameters()

    batch_size = args.batch_size

    model, block_size, history_size = generate_model(
        args=args, base_model=model, logger=logger, tokenizer=tokenizer
    )

    """### Prepare dataset"""
    logger.info("Preparing datasets and dataloaders")
    include_train = True
    train_dataloader, valid_dataloader, test_dataloader = generate_dataloaders(
        args, tokenizer, batch_size, block_size, history_size, include_train=include_train
    )

    """### Prepare optimizer and scheduler"""
    logger.info("Preparing optimizer")
    from torch.optim import AdamW
    optimizer_cls = (
        AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optim = optimizer_cls(model.parameters(), lr=args.learning_rate)
    
    from torch.optim.lr_scheduler import StepLR
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        if args.lr_decay:
            scheduler = StepLR(optim, step_size=100, gamma=args.lr_decay_gamma)
        else:
            scheduler = StepLR(optim, step_size=100, gamma=1.0)
    else:
        scheduler = DummyScheduler(
            optim, total_num_steps=args.training_step, num_training_steps=100
        )
    
    """### Prepare accelerator"""
    logger.info("Preparing accelerator")
    # wrap with accelerate
    model, optim, train_dataloader, valid_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optim, train_dataloader, valid_dataloader, test_dataloader, scheduler
    )

    logger.info("Moving model to device")
    model.to(device)

    logger.info("Preparing generators")
    train_gen = iter(train_dataloader)

    logger.info("Setting model to train mode")
    model.train()

    logger.info("Starting training")
    train_steps = args.training_step
    losses = []
    for epoch in range(args.num_epochs):
        train_gen = iter(train_dataloader)
        total_len = min(train_steps, len(train_dataloader))
        for step in tqdm.tqdm(range(min(train_steps, len(train_dataloader)))):
            optim.zero_grad()

            batch = next(train_gen)
            batch['segment_size'] = block_size
            batch['sum_fraction'] = args.sum_fraction
            if args.task_name == 'eda_qa':
                batch['mask_size'] = batch['answer_len'][0]
            if args.task_name == 'nvidia/ChatQA2-Long-SFT-data':
                batch['mask_size'] = batch['mask_size'][0]
            out, _, batch_metrics = model(**batch)
            loss = out.loss
            accelerator.backward(loss)
            optim.step()
            if args.lr_decay:
                scheduler.step()
            mean_loss = mean_metric(loss)
            losses.append(mean_loss)
            train_log = {"train loss": mean_loss}
            mean_ppl = mean_metric(batch_metrics.get("ppl"))
            if mean_ppl is not None:
                train_log["train ppl"] = mean_ppl
            mean_f1 = mean_metric(batch_metrics.get("f1"))
            if mean_f1 is not None:
                train_log["train f1"] = mean_f1
            accelerator.log(train_log, step=step+total_len*epoch)
            
            if step % 50 == 0:
                # evaluate
                model.eval()
                sub_valid_gen = iter(valid_dataloader)
                eval_losses = []
                eval_ppl = []
                eval_f1 = []
                for eval_step in range(min(10, len(valid_dataloader))):
                    eval_batch = next(sub_valid_gen)
                    eval_batch['segment_size'] = block_size
                    if args.task_name == 'eda_qa':
                        eval_batch['mask_size'] = eval_batch['answer_len'][0]
                    if args.task_name == 'nvidia/ChatQA2-Long-SFT-data':
                        eval_batch['mask_size'] = eval_batch['mask_size'][0]
                    with torch.no_grad():
                        out, _, eval_metrics = model(**eval_batch)
                    eval_losses.append(mean_metric(out.loss))
                    mean_eval_ppl = mean_metric(eval_metrics.get("ppl"))
                    if mean_eval_ppl is not None:
                        eval_ppl.append(mean_eval_ppl)
                    mean_eval_f1 = mean_metric(eval_metrics.get("f1"))
                    if mean_eval_f1 is not None:
                        eval_f1.append(mean_eval_f1)

                log_payload = {"eval loss": np.mean(eval_losses)}
                if eval_ppl:
                    log_payload["eval ppl"] = np.mean(eval_ppl)
                if eval_f1:
                    log_payload["eval f1"] = np.mean(eval_f1)
                accelerator.log(log_payload, step=step+total_len*epoch)
                model.train()


    accelerator.wait_for_everyone()
    if args.save_ckpt is not None:
        model.save_checkpoint(args.save_ckpt)

    eval_steps = args.eval_step
    valid_losses = []
    valid_ppl = []
    valid_f1 = []
    model.eval()
    valid_gen = iter(valid_dataloader)
    logger.info("Starting evaluation")

    for step in tqdm.tqdm(range(min(eval_steps, len(valid_dataloader)))):
        batch = next(valid_gen)
        batch['segment_size'] = block_size
        if args.task_name == 'eda_qa':
            batch['mask_size'] = batch['answer_len'][0]
        if args.task_name == 'nvidia/ChatQA2-Long-SFT-data':
            batch['mask_size'] = batch['mask_size'][0]
        if args.timing:
            batch['prof'] = True
        
        with torch.no_grad():
            out, _, val_metrics = model(**batch)
        loss = out.loss
        # ppl = out.ppl
        # logger.debug(f'loss: {loss.item()}')
        # logger.debug(f'ppl: {ppl.item()}')
        valid_losses.append(mean_metric(loss))
        mean_valid_ppl = mean_metric(val_metrics.get("ppl"))
        if mean_valid_ppl is not None:
            valid_ppl.append(mean_valid_ppl)
        mean_valid_f1 = mean_metric(val_metrics.get("f1"))
        if mean_valid_f1 is not None:
            valid_f1.append(mean_valid_f1)

    logger.info(f'Loss on {min(eval_steps, len(valid_dataloader)) * batch_size} validation samples (CrossEntropy): {np.mean(valid_losses)}')
    if valid_ppl:
        logger.info(f'PPL on {min(eval_steps, len(valid_dataloader)) * batch_size} validation samples: {np.mean(valid_ppl)}')
    if valid_f1:
        logger.info(f'F1 on {min(eval_steps, len(valid_dataloader)) * batch_size} validation samples: {np.mean(valid_f1)}')

    test_steps = args.test_step
    test_losses = []
    test_ppl = []
    test_f1 = []
    total_hist = []

    test_gen = iter(test_dataloader)
    logger.info("Starting testing")

    for step in tqdm.tqdm(range(min(test_steps, len(test_dataloader)))):
        batch = next(test_gen)
        batch['segment_size'] = block_size
        if args.task_name == 'eda_qa':
            batch['mask_size'] = batch['answer_len'][0]
        if args.task_name == 'nvidia/ChatQA2-Long-SFT-data':
            batch['mask_size'] = batch['mask_size'][0]
        if args.timing:
            batch['prof'] = True
        
        with torch.no_grad():
            out, hist, test_metrics = model(**batch)
        loss = out.loss
        # ppl = out.ppl
        test_losses.append(mean_metric(loss))
        mean_test_ppl = mean_metric(test_metrics.get("ppl"))
        if mean_test_ppl is not None:
            test_ppl.append(mean_test_ppl)
        mean_test_f1 = mean_metric(test_metrics.get("f1"))
        if mean_test_f1 is not None:
            test_f1.append(mean_test_f1)
        # logger.info(f'loss: {loss.item()}')
        if hist is not None:
            total_hist.extend(hist)
    
    if (args.baseline_only == False) and (args.rmt_only == False) and args.plot_hist:
        max_d = np.max(total_hist)
        plt.hist(total_hist, weights=np.ones(len(total_hist))/len(total_hist), bins=50)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel("Context Distance")
        plt.ylabel("Probability")
        plt.savefig('artifact/heatmap_' + date_str + '.png')
        plt.show()

    if test_ppl:
        logger.info(f'PPL on {min(test_steps, len(test_dataloader)) * batch_size} test samples: {np.mean(test_ppl)}')
    if test_f1:
        logger.info(f'F1 on {min(test_steps, len(test_dataloader)) * batch_size} test samples: {np.mean(test_f1)}')
    
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
        # load pre-computed memory embeddings
        mem_seq = None
        if os.path.exists('memory.pt'):
            mem_seq = torch.load('memory.pt')

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
                    mem_seq = mem_seq,
                    max_new_tokens = len(tok_answer),
                    do_sample=False
                )
            predictions = tokenizer.batch_decode(output_seq, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            references = [answer_str]
            results = rouge.compute(predictions=predictions, references=references)
            rougeL_full.append(results['rougeL'])
        logger.info(f'ROUGE-L on {len(ORD_QA_sample)} test samples using whole database: {np.mean(rougeL_full)}')


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
                max_new_tokens = batch['answer_len'][0],
                do_sample=False
            )
            predictions = tokenizer.batch_decode(output_seq, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            references = batch['answer']
            results = rouge.compute(predictions=predictions, references=references)
            rougeL.append(results['rougeL'])
        
        logger.info(f'ROUGE-L on {len(test_dataloader)} test samples with only correct reference: {np.mean(rougeL)}')

if __name__ == "__main__":
    main()
