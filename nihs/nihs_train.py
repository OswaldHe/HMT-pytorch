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
from hmt_tools.collate import hmt_collate_fn
from hmt_tools.models import load_model
from datasets import load_dataset
from hmt_tools.data_processing.generic import prepare_test, prepare_train

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
parser.add_argument('--train_set_split', type=str, default='100%', 
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
parser.add_argument('--save_interval', type=int, default=0, help='Save checkpoint every N steps. 0 means no intermediate saving.')
parser.add_argument('--save_dir', type=str, default=None, help='Directory to save checkpoints')
parser.add_argument('--validation_interval', type=int, default=100, help='Perform validation every N steps')
parser.add_argument('--validation_steps', type=int, default=10, help='Number of validation steps to perform at each validation interval')
# parser.add_argument('--curriculum', action='store_true', default=False, help='use curriculum learning')
# parser.add_argument('--curriculum_segs', type=str, default=None, help='Comma-separated list of curriculum levels (number of segments for each level)')
parser.add_argument('--wandb_project', type=str, default='redpajama_curriculum', help='Name for the WanDB Project')
parser.add_argument('--wandb_run', type=str, default=None, help='Name for the WanDB run')
parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity (username or team name)')
parser.add_argument('--recache_splits', type=str, default=None, help='Provide a list of dataset splits that to rebuild the tokenization and grouping cache')
parser.add_argument('--max_context_length', type=int, default=None, help='Maximum context length for the dataset. If None, no limit is applied.')
parser.add_argument('--is_qa_task', action='store_true', default=False, help='Whether the task is a QA task')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
parser.add_argument('--rouge', action='store_true', default=False, help='Whether to evaluate Rouge-L')
parser.add_argument('--cache_dir', type=str, default='.', help='cache directory, default to the current directory')
parser.add_argument('--it', action='store_true', default=False, help='whether to use instruction tunning version of the QMSum test set')
parser.add_argument('--with_text', action='store_true', default=False, help='whether to return the text in the dataset')
parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of new tokens to generate')
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
        config={"dropout": 0.1, 
                "learning_rate": args.learning_rate, 
                "model_name": args.model_name,
                "task_name": args.task_name, 
                "test_length": args.test_length, 
                "checkpoint": os.path.basename(args.load_from_ckpt),
                "max_context_length": args.max_context_length},
        init_kwargs={"wandb": {"entity": args.wandb_entity, "name": f'{args.wandb_run}'}}
    )

    recache_splits = args.recache_splits.split(',') if args.recache_splits else None
    print("recache splits: ", recache_splits)

    token=None
    if args.token_file is not None:
        with open(args.token_file, 'r') as f:
            token = f.read()

    if args.rouge:
        import evaluate
        rouge_metric = evaluate.load('rouge')

    if args.save_dir is None:
        save_dir = f'./checkpoints/{args.model_name}'
    else:
        save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        logger.info(f"Created directory: {save_dir}")
    else:
        logger.info(f"Directory already exists: {save_dir}")

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

    collate_fn = partial(hmt_collate_fn, id_pad_value=id_pad_value, is_qa_task=args.is_qa_task, block_size=block_size, batch_size=batch_size, with_text=args.with_text)
    
    # Log the step
    logger.info("Loading datasets")
    dataset = datasets.load_dataset(args.task_name, args.task_subset)
    all_train = {}
    all_valid = {}
    from hmt_tools.data_processing.prep_funcs import prepare_nihs_train, prepare_nihs_test
    for split in dataset:
        # print(dataset[split])
        train_ds = prepare_train(dataset[split].select(range(80)), partial(prepare_nihs_train, tokenizer=tokenizer, task=args.task_subset), max_token_num=args.max_context_length, test_length=None, block_size=block_size, tokenizer=tokenizer, with_answer=True, with_text=args.with_text)
        all_train[split] = train_ds
        valid_ds = prepare_test(dataset[split].select(range(80, 100)), partial(prepare_nihs_test, tokenizer=tokenizer, task=args.task_subset), max_token_num=args.max_context_length, test_length=args.test_length, block_size=block_size, tokenizer=tokenizer, with_answer=True, with_text=args.with_text)
        all_valid[split] = valid_ds

    if args.with_text:
        print('train sample --->')
        print(all_train['1k'][0]['text'])
        print('valid sample --->')
        print(all_valid['1k'][0]['text'])
    
    train_rnd_generator = torch.Generator()
    train_rnd_generator.manual_seed(args.seed)

    train_dataloaders = {}
    valid_dataloaders = {}
    for split in all_train:
        train_dataloaders[split] = DataLoader(all_train[split], batch_size=batch_size, collate_fn=collate_fn,
                                    shuffle=args.shuffle, drop_last=False, generator=train_rnd_generator, pin_memory=True)
        # Print the length of train and validation datasets
        logger.info(f"Number of training datapoints for {split}: {len(train_dataloaders[split])}")
    for split in all_valid:
        valid_dataloaders[split] = DataLoader(all_valid[split], batch_size=batch_size, collate_fn=collate_fn,
                                    shuffle=args.shuffle, drop_last=True, pin_memory=True)
        logger.info(f"Number of validation datapoints for {split}: {len(valid_dataloaders[split])}")

    # if args.shuffle_train:
    #     train_ds = train_ds.shuffle(seed=args.seed)
    # else:
    #     train_ds = train_ds.take(int(args.train_set_split))

    # if args.shuffle:
    #     valid_ds = valid_ds.shuffle(seed=args.seed)
    # else:
    #     valid_ds = valid_ds.take(int(args.train_set_split))

    logger.info("Wrapping Model with HMT")
    model = load_model(args, model=model, memory_size=memory_size, block_size=block_size, n_segments=n_segments, mask_size=mask_size, word_emb_dim=word_emb_dim)

    
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
    all_dataloaders = list(train_dataloaders.values()) + list(valid_dataloaders.values())
    model, optim, *all_dataloaders, scheduler = accelerator.prepare(
        model, optim, *all_dataloaders, scheduler
    )
    train_dataloaders = {split: dataloader for split, dataloader in zip(all_train.keys(), all_dataloaders[:len(all_train)])}
    valid_dataloaders = {split: dataloader for split, dataloader in zip(all_valid.keys(), all_dataloaders[len(all_train):])}

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
        global_step = 0
        for epoch in range(args.epochs):
            for split in train_dataloaders:
                train_gen = iter(train_dataloaders[split])
                for step in tqdm.tqdm(range(min(train_steps, len(train_dataloaders[split])))):
                    optim.zero_grad()
                    batch = next(train_gen)
                    # print("================")
                    # # print("<batch>", batch['text'])
                    # print("<input_id shape>", batch['input_ids'].shape)
                    # print("input_ids", batch['input_ids'])
                    # print("attention_mask", batch['attention_mask'])
                    # quit()
                    # print("================")

                    if args.is_qa_task:
                        _ = batch.pop('answer')
                    if args.with_text:
                        _ = batch.pop('text')

                    batch['segment_size'] = block_size
                    batch['sum_fraction'] = args.sum_fraction
                    accelerator.log({ "Input Length": batch['input_ids'].shape[1]}, step=global_step)
                    
                    out, _ = model(**batch)
                    loss = out.loss
                    f1 = out.f1['f1']

                    accelerator.backward(loss)
                    optim.step()
                    if args.lr_decay:
                        scheduler.step()
                    losses.append(loss.detach().item())
                    accelerator.log({"Train Loss": loss.detach().item(), "Train F1": f1, "Epoch": epoch}, step=global_step)
                    
                    if step % args.save_interval == 0:
                        torch.save(model.state_dict(), f'{save_dir}/model_weights_{global_step}.pth')
                    
                    if step % args.validation_interval == 0:
                        valid_losses = []
                        valid_ppl = []
                        valid_f1 = []
                        valid_rouge = []
                        valid_gen = iter(valid_dataloaders[split])
                        model.eval()
                        for val_step in range(min(args.validation_steps, len(valid_dataloaders[split]))):
                            batch = next(valid_gen)
                            if args.is_qa_task:
                                answer = batch.pop('answer')
                            if args.with_text:
                                _ = batch.pop('text')
                            # for k, v in batch.items():
                            #     batch[k] = v.cpu()
                            with torch.no_grad():
                                out, _ = model(**batch)

                                if args.rouge:
                                    text_labels = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], segment_size=block_size, max_new_tokens=args.max_new_tokens, num_beams=1)
                                    text_out = tokenizer.decode(text_labels[0], skip_special_tokens=True)
                                    print("================")
                                    print("<val_step>", val_step)
                                    print("<out>", text_out)
                                    print("<answer>", answer)
                                    print("================")
                                    # all_rouges = accelerator.gather(text_out)
                                    # all_answers = accelerator.gather(answer)
                                    rouge = rouge_metric.compute(predictions=[text_out], references=[answer])
                                    # rouge = rouge_metric.compute(predictions=all_rouges, references=all_answers)
                                    valid_rouge.append(rouge['rouge1'].item())

                            loss = out.loss
                            ppl = out.ppl
                            f1 = out.f1['f1']

                            valid_losses.append(loss.detach().item())
                            valid_ppl.append(ppl.detach().item())
                            valid_f1.append(f1)

                        # Log to wandb by calling `accelerator.log`, `step` is optional
                        accelerator.log({"Validation Loss": np.mean(valid_losses), "Validation PPL": np.mean(valid_ppl), "Validation F1": np.mean(valid_f1), "Epoch": epoch}, step=global_step)
                        if args.rouge:
                            accelerator.log({"Validation Rouge": np.mean(valid_rouge), "Epoch": epoch}, step=global_step)
                        model.train()
                    global_step += 1

            accelerator.wait_for_everyone()
        
        if args.save_ckpt is not None:
            model.save_checkpoint(args.save_ckpt)
        
        plt.plot(losses)
        plt.xlabel('step')
        plt.ylabel('train loss')
        date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        plt.savefig('artifact/loss_' + date_str + '.png')
        plt.show()
        plt.close()

    # valid_losses = []
    # valid_ppl = []
    # model.eval()
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
