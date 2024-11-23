import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM

from tqdm.auto import tqdm
import pandas as pd
import json

from pathlib import Path

import yaml

from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input
from babilong.babilong_utils import compare_answers

from hmt_tools.models import load_model_yaml


def process_config(config_path):
    with open(config_path, 'r') as f:
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

    return config

def params_from_config(model, model_config):
    from peft import LoraConfig, TaskType, get_peft_model
    from modeling_rmt.compression import inject_eae
    from transformers import OPTConfig

    if isinstance(model.config, OPTConfig):
        word_emb_dim = model.config.word_embed_proj_dim
    else:
        word_emb_dim = model.config.hidden_size


    if model_config['inject_autoencoder']:
        model = inject_eae(model, word_emb_dim, 16, 2)

    if model_config['use_lora']:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False, 
            # target_modules=['embed_tokens', 'gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj'],
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.1
            )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    input_size = model_config['segment_length']
    memory_size = 1
    n_segments = model_config['bptt_depth']

    if model_config['baseline_only']:
        memory_size = 0
        n_segments = 2

    batch_size = model_config['batch_size']

    block_size = input_size
    block_size -= 2 * memory_size
    block_size -= model_config['num_sensory']
    history_size = (n_segments - 1) * block_size

    mask_size = block_size

    block_size_2 = input_size - (2*memory_size) - model_config['num_sensory']//2

    return {
        'memory_size': memory_size,
        'block_size': block_size,
        'n_segments': n_segments,
        'mask_size': mask_size,
        'block_size_2': block_size_2,
        'batch_size': batch_size,
        'word_emb_dim': word_emb_dim
    }


def babilong_eval(model_config, data_lengths=['0k', '1k'], tasks=['qa1', 'qa2'], results_folder='./babilong_results', device='cuda:0'):
    # read the model config and compute model parameters
    model_config = process_config(model_config)

    token=None
    if model_config['read_token_file'] is not None:
        with open(model_config['read_token_file'], 'r') as f:
            token = f.read()

    # load the model
    tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'], trust_remote_code=True, token=token)
    tokenizer.model_input_names = tokenizer.model_input_names + ['attention_mask']
    
    model = AutoModelForCausalLM.from_pretrained(model_config['model_name'], trust_remote_code=True, token=token)
    params = params_from_config(model, model_config)
    model = load_model_yaml(model=model, memory_size=params['memory_size'], block_size=params['block_size'], n_segments=params['n_segments'], mask_size=params['mask_size'], word_emb_dim=params['word_emb_dim'], **model_config)
    model = model.to(device)
    model = model.eval()

    generate_kwargs = {
        'max_new_tokens': 20,
        'max_length': None,
        'num_beams': 1,
        'do_sample': False,
        'temperature': None,
        'top_p': None,
        'top_k': None,
        'pad_token_id': tokenizer.pad_token_id
    }

    if generate_kwargs['pad_token_id'] is None:
        generate_kwargs['pad_token_id'] = tokenizer.eos_token_id

    use_chat_template = True
    use_instruction = True
    use_examples = True
    use_post_prompt = True

    for task in tqdm(tasks, desc='tasks'):
        # configure the prompt
        prompt_cfg = {
            'instruction': DEFAULT_PROMPTS[task]['instruction'] if use_instruction else '',
            'examples': DEFAULT_PROMPTS[task]['examples'] if use_examples else '',
            'post_prompt': DEFAULT_PROMPTS[task]['post_prompt'] if use_post_prompt else '',
            'template': DEFAULT_TEMPLATE,
            'chat_template': use_chat_template,
        }
        prompt_name = [f'{k}_yes' if prompt_cfg[k] else f'{k}_no' for k in prompt_cfg if k != 'template']
        prompt_name = '_'.join(prompt_name)

        for split_name in tqdm(data_lengths, desc='lengths'):
            # load dataset
            data = datasets.load_dataset('RMT-team/babilong', split_name)
            task_data = data[task]

            # Prepare files with predictions, prompt, and generation configurations
            outfile = Path(f'{results_folder}/{model_config["model_name"]}/{task}_{split_name}_{prompt_name}.csv')
            outfile.parent.mkdir(parents=True, exist_ok=True)
            cfg_file = f'./{results_folder}/{model_config["model_name"]}/{task}_{split_name}_{prompt_name}.json'
            json.dump({'prompt': prompt_cfg, 'generate_kwargs': generate_kwargs}, open(cfg_file, 'w'), indent=4)

            df = pd.DataFrame({'target': [], 'output': [], 'question': []})

            for sample in tqdm(task_data, desc=f'task: {task} length: {split_name}'):
                target = sample['target']
                context = sample['input']
                question = sample['question']

                # format input text
                input_text = get_formatted_input(context, question, prompt_cfg['examples'],
                                                 prompt_cfg['instruction'], prompt_cfg['post_prompt'],
                                                 template=prompt_cfg['template'])

                if use_chat_template:
                    input_text = [{'role': 'user', 'content': input_text}]
                    model_inputs_text = tokenizer.apply_chat_template(input_text, add_generation_prompt=True,
                                                                 return_tensors='pt', tokenize=False)
                    # print(model_inputs_text)
                    # exit()
                    model_inputs_tokenized = tokenizer(model_inputs_text, return_tensors='pt', add_special_tokens=True, return_attention_mask=True).to(device)
                    model_inputs = {'input_ids': model_inputs_tokenized['input_ids'], 'attention_mask': model_inputs_tokenized['attention_mask']}
                else:
                    model_inputs = tokenizer(input_text, return_tensors='pt',
                                             add_special_tokens=True, return_attention_mask=True).to(device)

                sample_length = model_inputs['input_ids'].shape[1]
                with torch.no_grad():
                    output = model.generate(**model_inputs, segment_size=params['block_size'], **generate_kwargs)

                # output = output[0][sample_length:]
                output = output[0]
                output = tokenizer.decode(output, skip_special_tokens=True).strip()

                df.loc[len(df)] = [target, output, question]
                # write results to csv file
                df.to_csv(outfile)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--results_folder', type=str, required=True)
    parser.add_argument('--tasks', type=str, required=True)
    parser.add_argument('--length', type=str, required=True)
    parser.add_argument('--device', type=str, required=False, default='cuda:0')
    args = parser.parse_args()
    
    tasks = args.tasks.split(',')
    length = args.length.split(',')

    babilong_eval(args.config, data_lengths=length, tasks=tasks, results_folder=args.results_folder, device=args.device)
