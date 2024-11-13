"""
Upload models to Huggingface
"""

import os
import torch
import logging

from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTConfig
from accelerate import Accelerator, DistributedDataParallelKwargs
from pathlib import Path
from peft import get_peft_model, LoraConfig, TaskType
from modeling_rmt.compression import inject_eae
import logging
from hmt_tools.models import load_model_yaml
from accelerate.logging import get_logger
import re

parser = ArgumentParser()

parser.add_argument('--config', type=str, default='config.yaml', help='path to the config file')

#cli arguments
# parser.add_argument('--task_name', type=str, default='wikitext', help='training/validation task name (e.g. wikitext, pg19, samsum, etc.)')
# parser.add_argument('--task_subset', type=str, default=None, help='subset of dataset (e.g., wikitext-2-v1)')
parser.add_argument('--batch_size', type=int, default=2, help='number of batches per device')
# parser.add_argument('--num_seg_save', type=int, default=5, help='max number of segment inference results saved on GPU')
# parser.add_argument('--seed', type=int, default=3407, help='random seed for training')
parser.add_argument('--model_name', type=str, default='facebook/opt-2.7b', help='transformer model name for backbone of HMT')
parser.add_argument('--segment_length', type=int, default=256, help='segment length of HMT')
parser.add_argument('--bptt_depth', type=int, default=2, help='number of segments unrolled in bptt')
# parser.add_argument('--sum_fraction', type=float, default=0.5, help='fraction of the segment that will be used for representation extraction')
# parser.add_argument('--test_length', type=int, default=2000, help='context length of input to test')
# parser.add_argument('--training_step', type=int, default=500, help='number of training steps')
# parser.add_argument('--eval_step', type=int, default=100, help='number of evaluation steps')
# parser.add_argument('--test_step', type=int, default=100, help='number of testing steps')
# parser.add_argument('--learning_rate', type=float, default=1e-5, help='training learning rate')
# parser.add_argument('--lr_decay', action='store_true', default=False, help='whether having learning rate decay or not')
parser.add_argument('--use_lora', action='store_true', default=False, help='whether use PEFT LoRA to speed up training')
# parser.add_argument('--lr_decay_gamma', type=float, default=0.8, help='rate of lr decay')
parser.add_argument('--num_sensory', type=int, default=0, help='number of preserved tokens for sensory memory')
# parser.add_argument('--mem_recall_hidden_dim', type=int, default=4096, help='hidden dimension of cross attention in memory recall mech.')
# parser.add_argument('--rmt_only', action='store_true', default=False, help='train and evaluate with only rmt')
parser.add_argument('--baseline_only', action='store_true', default=False, help='train and evaluate only the backbone model')
# parser.add_argument('--segment_alignment', type=str, default=None, help='alignment of segments in evaluation.')
# parser.add_argument('--hmt_stage_1', action='store_true', default=False, help='stage 1 of HMT to find memory param')
# parser.add_argument('--hmt_stage_2', action='store_true', default=False, help='stage 2 of HMT to find memory param')
# parser.add_argument('--save_ckpt', type=str, default=None, help='store the model checkpoint to the specified directory, only used for HMT')
# parser.add_argument('--load_from_ckpt', type=str, default=None, help='load the checkpoint for HMT stage 2')
# parser.add_argument('--mem_recall_context', type=int, default=100, help='number of memory embeddings cached in memory recall mech.')
parser.add_argument('--token_file', type=str, default=None, help='path to the file with Huggingface token. Used for gated model such as Llama2.')
# parser.add_argument('--train_set_split', type=str, default=None, 
#         help='slice upper bound of training set to reduce time for tokenization. use percentage notation (e.g., 2%), or integer')
# parser.add_argument('--interleave_dataset', action='store_true', default=False, help='whether mix every two samples in the dataset to create context switching.')
# parser.add_argument('--interleave_len', type=int, default=100, help='the interleaving length of dataset (first sample pick some tokens, then the second).')
# parser.add_argument('--plot_hist', action='store_true', default=False, help='show memory recall context histogram.')
# parser.add_argument('--fuse_size', type=int, default=2, help='the number of questions and context to fuse for PubMedQA dataset')
# parser.add_argument('--timing', action='store_true', default=False, help='profile the timing of inference.')
# parser.add_argument('--inference_only', action='store_true', default=False, help='perform inference of the model only.')
# parser.add_argument('--dynamic', action='store_true', default=False, help='whether dynamically change reading speed based on memory.')
# parser.add_argument('--dilate_dataset', action='store_true', default=False, help='dilate the sample by inserting padding tokens.')
# parser.add_argument('--dilate_len', type=int, default=888, help='number of padding tokens inserted to dilate the sample.')
# parser.add_argument('--dilate_str', type=str, default='$', help='the token you want to insert to dilate the sample.')
# parser.add_argument('--train_memory_map', action='store_true', default=False, help='train memory projection for dynamic reading speed.')
parser.add_argument('--inject_autoencoder', action='store_true', default=False, help='use autoencoder to compress/decompress the intermediate embeddings.')
# parser.add_argument('--generate', type=str, default=None, help='generate for harry potter book.')
# parser.add_argument('--streaming', action='store_true', default=False, help='generate text in streaming mode')
# parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle the dataset')
# parser.add_argument('--save_interval', type=int, default=0, help='Save checkpoint every N steps. 0 means no intermediate saving.')
# parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
# parser.add_argument('--validation_interval', type=int, default=100, help='Perform validation every N steps')
# parser.add_argument('--validation_step', type=int, default=10, help='Number of validation steps to perform at each validation interval')
# parser.add_argument('--curriculum', action='store_true', default=False, help='use curriculum learning')
# parser.add_argument('--curriculum_segs', type=str, default=None, help='Comma-separated list of curriculum levels (number of segments for each level)')
# parser.add_argument('--wandb_project', type=str, default='redpajama_curriculum', help='Name for the WanDB Project')
# parser.add_argument('--wandb_run', type=str, default=None, help='Name for the WanDB run')
# parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity (username or team name)')
# parser.add_argument('--max_context_length', type=int, default=10000, help='Maximum context length for the dataset. If None, no limit is applied.')
# parser.add_argument('--recache_splits', type=str, default=None, help='Provide a list of dataset splits that to rebuild the tokenization and grouping cache')
# parser.add_argument('--rouge', action='store_true', default=False, help='compute rouge-l score')
# parser.add_argument('--is_qa_task', action='store_true', default=False, help='whether the task is a qa task')
# parser.add_argument('--temperature', type=float, default=1.0, help='temperature for sampling')
# parser.add_argument('--max_new_tokens', type=int, default=100, help='maximum number of new tokens generated')
parser.add_argument('--cache_dir', type=str, default='.', help='cache directory, default to the current directory')
# parser.add_argument('--save_generated_texts', type=str, default=None, help='filename to save the generated texts')
# parser.add_argument('--do_sample', action='store_true', default=False, help='whether to sample from the model')
# parser.add_argument('--num_beams', type=int, default=5, help='number of beams for beam search')
# parser.add_argument('--it', action='store_true', default=False, help='whether to use instruction tunning version of the QMSum test set')
torch.manual_seed(3407)

def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


def upload_model(**kwargs):
    global torch

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    token=None
    if kwargs['read_token_file'] is not None:
        with open(kwargs['read_token_file'], 'r') as f:
            token = f.read()
    if kwargs['write_token_file'] is not None:
        with open(kwargs['write_token_file'], 'r') as f:
            write_token = f.read()

    """### Load model"""
    cache_dir = os.environ.get('HF_HOME', kwargs.get('cache_dir', '.'))
    model = AutoModelForCausalLM.from_pretrained(kwargs['model_name'], token=token, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(kwargs['model_name'], token=token, cache_dir=cache_dir)

    if isinstance(model.config, OPTConfig):
        word_emb_dim = model.config.word_embed_proj_dim
    else:
        word_emb_dim = model.config.hidden_size


    if kwargs['inject_autoencoder']:
        model = inject_eae(model, word_emb_dim, 16, 2)

    if kwargs['use_lora']:
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
    input_size = kwargs['segment_length']
    memory_size = 1
    n_segments = kwargs['bptt_depth']

    if kwargs['baseline_only']:
        logger.warning('training and evaluating only the backbone. remember to align the segment rightward')
        memory_size = 0
        n_segments = 2

    batch_size = kwargs['batch_size']

    block_size = input_size
    block_size -= 2 * memory_size
    block_size -= kwargs['num_sensory']
    history_size = (n_segments - 1) * block_size

    mask_size = block_size

    block_size_2 = input_size - (2*memory_size) - kwargs['num_sensory']//2

    """### Prepare dataset"""

    id_pad_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    logger.info("Preparing memory cell")
    model = load_model_yaml(model=model, memory_size=memory_size, block_size=block_size, n_segments=n_segments, mask_size=mask_size, word_emb_dim=word_emb_dim, **kwargs)

    model.push_to_hub(re.sub(r'^[^/]+/', 'ioeddk/', kwargs['model_name']+'_hmt'), token=write_token)


if __name__ == "__main__":
    import yaml
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    defaults = {
        'inject_autoencoder': False,
        'use_lora': False,
        'baseline_only': False,
        'rmt_only': False,
        'hmt_stage_1': False,
        'hmt_stage_2': False,
        'mem_recall_hidden_dim': 4096,
        'mem_recall_context': 100,
        'segment_alignment': None,
    }

    for key, value in defaults.items():
        if config.get(key) is None:
            config[key] = value

    upload_model(**config)
