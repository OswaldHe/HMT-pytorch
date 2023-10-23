import json
import logging
import os
import math
import shutil
from pathlib import Path
from itertools import chain

# from dotenv import load_dotenv
import torch
import numpy as np
import datasets
import transformers
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download

from lm_experiments_tools.trainer_accelerate import TrainerAccelerate as Trainer, TrainerAccelerateArgs as TrainerArgs

from torch.nn.utils.rnn import pad_sequence

import accelerate

# load_dotenv()

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger('')


# if CUDA_VISIBLE_DEVICES is not set make all gpus visible
if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])

logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
# first call to torch.cuda.device_count() sets visible gpus, following calls will not change the result
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

# import transformers  # noqa: E402
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser  # noqa: E402

from lm_experiments_tools.utils import get_cls_by_name, get_optimizer, prepare_run  # noqa: E402
import lm_experiments_tools.optimizers as optimizers  # noqa: E402

# limit # of CPU threads to be used per pytorch worker, otherwise it might use all cpus and throttle gpus
# > 2 fails cause of https://github.com/pytorch/pytorch/issues/56615
# need to upgrade to torch>1.8.1
# torch.set_num_threads(4)
# all gpus set with CUDA_VISIBLE_DEVICES are visible to process, indexing from 0 to ...

parser = HfArgumentParser(TrainerArgs)
parser.add_argument('--task_name', type=str, help='Scrolls task name: "gov_report", "summ_screen_fd", "qmsum", '
                                                  '"narrative_qa", "qasper", "quality", "contract_nli"')
parser.add_argument('--validate_only', action='store_true', default=False,
                    help='Skip training and run only validation. (default: False)')
parser.add_argument('--working_dir', type=str, default='.',
                    help='working dir, should be a dir with t5-experiments repo (default: .)')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--show_valid_examples', type=int, default=0,
                    help='how many valid examples to show during training (default: 0)')
parser.add_argument('--input_seq_len', type=int, default=128, help='input sequnce length (default: 128).')
parser.add_argument('--target_seq_len', type=int, default=16, help='target sequnce length, should be set to '
                                                                   'max(len(target))+1 for EOS (default: 16).')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')

parser.add_argument('--input_prefix', type=str, default='', help='add task prefix to an input string (default: "")')
parser.add_argument('--sliding_window', action='store_true', help='use slinding window attention mask, '
                    'eval on last segment only', default=False)

# parser.add_argument('--use_generate_on_valid', action='store_true', help='use generate methon on validation', default=False)

# model args
parser.add_argument('--from_pretrained', type=str, help='model name in HF Model Hub (default: "")')
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: "")')
parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
                    help='model class name to use (default: transformers:BertForPreTraining)')
parser.add_argument('--memory_cell_cls', type=str, default=None, help='cell class for RMT')
parser.add_argument('--recurrent_wrapper_cls', type=str, default=None, help='recurrent wrapper class for RMT')
parser.add_argument('--model_cpt', type=str, default=None, help='pretrained model checkpoint path')
parser.add_argument('--model_type', type=str, default='encoder-decoder',
                    help='model type, encoder, encoder-decoder, decoder, affects preprocessing '
                         '(default: encoder-decoder)')


# Aydar # RMT args 
parser.add_argument('--input_size', type=int, default=None, help='maximal input size of the backbone model')
parser.add_argument('--num_mem_tokens', type=int, default=None, help='number of memory tokens.')
parser.add_argument('--max_n_segments', type=int, default=1, help='maximal segment number')
parser.add_argument('--vary_n_segments', action='store_true', default=False, help='Randomly choose segment number from 1 to max_n_segments')
parser.add_argument('--segment_alignment', type=str, default=None, help="How to align segments when splitting input")
parser.add_argument('--k2', type=int, default=-1, help='number of last segments used by backward')
parser.add_argument('--freeze_model_weights', action='store_true', default=False,
                    help='Stop training all model weights except memory layers')
parser.add_argument('--backbone_cpt', type=str, default=None, help='backbone model checkpoint path')


# tokenizer
# todo: add wordpiece tokenizers support?
parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (default: 0.0)')
parser.add_argument('--scale_parameter', action='store_true', default=False,
                    help='Adafactor scale_parameter (default: False)')
parser.add_argument('--relative_step', action='store_true', default=False,
                    help='Adafactor relative_step (default: False)')
parser.add_argument('--warmup_init', action='store_true', default=False,
                    help='Adafactor warmup_init (default: False)')


def download_metric():
    scrolls_metric_path = hf_hub_download(repo_id="tau/scrolls", filename="metrics/scrolls.py", repo_type="dataset")
    updated_scrolls_metric_path = (
        os.path.dirname(scrolls_metric_path) + os.path.basename(scrolls_metric_path).replace(".", "_") + ".py"
    )
    shutil.copy(scrolls_metric_path, updated_scrolls_metric_path)
    return updated_scrolls_metric_path


scrolls_metric_path = download_metric()

task_to_metric = {
    'gov_report': ['rouge/rouge1', 'rouge/rouge2', 'rouge/rougeL', 'rouge/rougeLsum', 'rouge/geometric_mean'],
    'summ_screen_fd': ['rouge/rouge1', 'rouge/rouge2', 'rouge/rougeL', 'rouge/rougeLsum', 'rouge/geometric_mean'],
    'qmsum': ['rouge/rouge1', 'rouge/rouge2', 'rouge/rougeL', 'rouge/rougeLsum', 'rouge/geometric_mean'],
    'narrative_qa': ['f1'],
    'qasper': ['f1'],
    'quality': ['exact_match'],
    'contract_nli': ['exact_match']
}

tasks_with_duplicates = {'narrative_qa', 'qasper'}


# https://github.com/tau-nlp/scrolls/blob/5bfb8dbaf3a0128ac8c65922096fd95a645f6ba2/baselines/src/utils/duplicates.py#L1
# some tasks have multiple possible labels for single input, drop_duplicates_in_input will collect such labels
def drop_duplicates_in_input(untokenized_dataset):
    indices_to_keep = []
    id_to_idx = {}
    outputs = []
    for i, (id_, output) in enumerate(zip(untokenized_dataset["id"], untokenized_dataset["output"])):
        if id_ in id_to_idx:
            outputs[id_to_idx[id_]].append(output)
            continue
        indices_to_keep.append(i)
        id_to_idx[id_] = len(outputs)
        outputs.append([output])
    untokenized_dataset = untokenized_dataset.select(indices_to_keep).flatten_indices()
    untokenized_dataset = untokenized_dataset.remove_columns("output")
    untokenized_dataset = untokenized_dataset.add_column("outputs", outputs)
    return untokenized_dataset


if __name__ == '__main__':
    args = parser.parse_args()
    # set current working dir
    args.working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(args.working_dir)

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    from accelerate.logging import get_logger
    logger = get_logger('')

    logger.info(f'num processes: {accelerator.num_processes}')
    logger.info(f'mixed precision: {accelerator.mixed_precision}')

    if args.model_path is None:
        logger.warning('model_path is not set: config, logs and checkpoints will not be saved.')

    # # create model path and save configuration
    # # todo: use prepare run
    # if accelerator.is_main_process and args.model_path is not None:
    #     model_path = Path(args.model_path)
    #     if not model_path.exists():
    #         Path(model_path).mkdir(parents=True)
    #     args_dict = collect_run_configuration(args)
    #     # todo: if model path exists and there is config file, write new config file aside
    #     json.dump(args_dict, open(model_path/'config.json', 'w'), indent=4)
    #     open(model_path / 'git.diff', 'w').write(get_git_diff())

    prepare_run(args, logger, logger_fmt)

    if not args.from_pretrained:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)

    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.model_type == 'decoder':
        from torch.nn.utils.rnn import pad_sequence
        tokenizer.add_special_tokens({'additional_special_tokens': ['[GEN]', '[PAD]']})
        gen_token = tokenizer.encode('[GEN]')[0]
        tokenizer.pad_token_id = tokenizer.encode('[PAD]')[0]
        eos_token = tokenizer.eos_token_id
        
        id_pad_value = tokenizer.pad_token_id

        block_size = args.input_size
        if args.num_mem_tokens not in {0, None}:
            block_size -= 2 * args.num_mem_tokens

        def collate_fn(batch):
            inputs = [b['input'][:args.input_seq_len * 10] for b in batch]
            if 'outputs' in batch[0]:
                # if we have more than 1 label per example (only in valid) take only one of them
                # to compute loss on valid
                target_text = [b['outputs'][0][:args.input_seq_len * 10] for b in batch]
            else:
                target_text = [b['output'][:args.input_seq_len * 10] for b in batch]

            collated = {}
            inputs = tokenizer.batch_encode_plus(list(inputs), max_length=args.input_seq_len, truncation=True, padding=False)
            labels = tokenizer.batch_encode_plus(list(target_text), padding=False)

            full_inputs = [torch.tensor(i[:args.input_seq_len - len(l) - 2] + [gen_token] + l + [eos_token]) for i, l in zip(inputs['input_ids'], labels['input_ids'])]
            labels_mask = [torch.zeros_like(i).bool() for i in full_inputs]
            for i, l in enumerate(labels['input_ids']):
                labels_mask[i][-len(l) - 2:] = True

            full_inputs = pad_sequence(full_inputs, padding_value=tokenizer.pad_token_id).T
            labels_mask = pad_sequence(labels_mask, padding_value=False).T

            gen_inputs = [torch.tensor(i[:args.input_seq_len - len(l) - 2] + [gen_token]) for i, l in zip(inputs['input_ids'], labels['input_ids'])]
            gen_inputs = pad_sequence(gen_inputs, padding_value=tokenizer.pad_token_id).T
            

            collated['input_ids'] = collated['labels'] = full_inputs
            collated['input_ids_generate'] = gen_inputs
            collated['labels_mask'] = labels_mask
            collated['attention_mask'] = (full_inputs != id_pad_value).bool()
            collated['attention_mask_generate'] = (gen_inputs != id_pad_value).bool()

            collated['id'] = [b['id'] for b in batch]
            collated['target_text'] = target_text
            return collated
    else:
        raise NotImplementedError(f'Unknown model type {args.model_type}')

    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}
    # get train dataset
    logger.info(f'preparing dataset for: {args.task_name}')
    dataset = datasets.load_dataset('tau/scrolls', args.task_name)
    train_dataset = dataset['train']
    # shuffle train data each epoch (one loop over train_dataset)
    train_rnd_generator = torch.Generator()
    train_rnd_generator.manual_seed(args.seed)
    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}
    train_dataloader = DataLoader(train_dataset, batch_size=per_worker_batch_size, generator=train_rnd_generator,
                                  collate_fn=collate_fn, **kwargs)
    # get validation dataset
    valid_dataloader = None
    logger.info(f'preparing validation data from: {args.task_name}')
    valid_dataset = dataset['validation']
    if args.task_name in tasks_with_duplicates:
        valid_dataset = drop_duplicates_in_input(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, batch_size=per_worker_batch_size,
                                  drop_last=True,
                                  collate_fn=collate_fn, **kwargs)
    if args.valid_interval is None:
        args.valid_interval = args.log_interval

    # define model
    model_cls = get_cls_by_name(args.model_cls)

    logger.info(f'Using model class: {model_cls}')
    if not args.from_pretrained:
        model_cfg = AutoConfig.from_pretrained(args.model_cfg)
        model = model_cls(config=model_cfg)
    else:
        logger.info(f'Loading pretrained model: {args.from_pretrained}')
        model = model_cls.from_pretrained(args.from_pretrained)

    # add [GEN] token
    model.resize_token_embeddings(len(tokenizer))
    
    ## load cpt of backbone model
    if args.backbone_cpt:
        backbone_cpt = os.path.join(args.backbone_cpt, "model_best.pth")
        cpt = torch.load(backbone_cpt, map_location='cpu')
        model.load_state_dict(cpt['model_state_dict'], strict=False)
        logger.info(f'Loaded baseline state dict from: {args.backbone_cpt}')

    # Pass memory settings to pretrained model
    if args.num_mem_tokens is not None:
        memory_cell_cls = get_cls_by_name(args.memory_cell_cls)
        recurrent_wrapper_cls = get_cls_by_name(args.recurrent_wrapper_cls)
        logger.info(f'Wrapping in: {memory_cell_cls} and {recurrent_wrapper_cls}')
        
        
        cell = memory_cell_cls(model, args.num_mem_tokens)
        model = recurrent_wrapper_cls(cell, 
                                      segment_size=block_size,
                                      max_n_segments=args.max_n_segments, 
                                      vary_n_segments=args.vary_n_segments,
                                      k2=args.k2,
                                      segment_alignment=args.segment_alignment
        )
                                    

        ## load cpt of rmt
        if args.model_cpt:
            model_cpt = os.path.join(args.model_cpt, "model_best/pytorch_model.bin")
            cpt = torch.load(model_cpt, map_location='cpu')
            vocab_size = model.memory_cell.model.gpt_neox.embed_in.weight.shape[0]
            cpt_vocab_size = cpt['memory_cell.model.gpt_neox.embed_in.weight'].shape[0]
            if vocab_size < cpt_vocab_size:
                model.memory_cell.model.resize_token_embeddings(cpt_vocab_size)
            model.load_state_dict(cpt, strict=False)
            logger.info(f'Loaded RMT state dict from: {args.model_cpt}')

    if args.freeze_model_weights:
        for n, p in model.named_parameters():
            # if 'memory' not in n and 'wte' not in n:
            if 'memory' not in n and 'lora' not in n:
                p.requires_grad = False
        logger.info(f'Frozen moodel weights')
        logger.info(f'Remaining parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}')

    # # fix the not-contiguous error with loralib and horovod
    # def make_contiguous(module):
    #     with torch.no_grad():
    #         for param in module.parameters():
    #             param.set_(param.contiguous())
    # make_contiguous(model)
    
    # define optimizer
    optimizer_cls = get_optimizer(args.optimizer)
    if optimizer_cls is None:
        raise RuntimeError(f'{args.optimizer} was not found in optimizers, torch.optim, transformers.optimization')

    logger.info(f'Using optimizer class: {optimizer_cls}')

    # todo: group optimizer params
    if optimizer_cls in [transformers.optimization.Adafactor, optimizers.Adafactor]:
        # https://github.com/huggingface/transformers/pull/9751/files -> transformers 4.3.0
        optimizer = optimizer_cls(model.parameters(), lr=args.lr,
                                  scale_parameter=args.scale_parameter,
                                  relative_step=args.relative_step,
                                  warmup_init=args.warmup_init,
                                  weight_decay=args.weight_decay)
    else:
        optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # for encoder only classification
    def keep_for_metrics_fn(batch, output):
        # select data from batch and model output that would be used to compute metrics
        data = {}
        if 'generation_outputs' in output:
            data['labels'] = batch['target_text']
            data['input_ids'] = batch['input_ids']
            data['labels_mask'] = batch['labels_mask']

            data['generation_outputs'] = output['generation_outputs']

        for key in batch.keys():
            if 'loss' in key: 
                data[key] = batch[key]

        return data

    # HF datasets can compute metrics on each gpu process and then aggregate them on process with rank 0
    # synchronization is done by using temporay files on a shared filesystem
    # rank and number of workers is set by num_process and process_id params
    # BUT our Trainer aggregates all prediction from all gpus!
    #   this will lead to computing metrics for predictions repeated xN_GPUS times
    # need to try:
    # - keep_in_memory=True, may lead to OOM for large validation sets, after sync predictions and targets for the full
    #       validation set would be stored on each GPU -> xN_GPUs RAM
    #   - implemented currently
    # - compute metrics on batch lvl
    # - add support of HF metrics and turn off aggregation in case if metric has .add_batch method
    scrolls_metric = datasets.load_metric(scrolls_metric_path, args.task_name, keep_in_memory=True)

    def metrics_fn(data):
        # compute metrics based on stored labels, predictions, ...
        metrics = {}
        y, p = None, None
        if 'generation_outputs' in data:
            # replace -100 with pad token in labels
            y = data['labels']
            p = tokenizer.batch_decode(data['generation_outputs'], skip_special_tokens=True)
            metrics['exact_match'] = np.mean([y_ == p_[:len(y_)] for p_, y_ in zip (p, y)])
            if args.show_valid_examples > 0:
                for i in range(min(args.show_valid_examples, len(y))):
                    logger.info(f'y: {y[i][:250]}')
                    logger.info(f'p: {p[i][:250]}')
                    logger.info(f'p ids: {len(data["generation_outputs"][i]), data["generation_outputs"][i][:50]}')

                    logger.info('-' * 50)
            
            if not isinstance(y[0], list):
                y = [[_y] for _y in y]
            result = scrolls_metric.compute(predictions=p, references=y)
            for metric_name in task_to_metric[args.task_name]:
                metrics[metric_name] = result[metric_name]

        return metrics

    # accelerate
    model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, None)

    ### booydar
    batch_metrics_fn = lambda _, y: {key: y[key] for key in y.keys() if (('loss' in key) or ('!log' in key))}
    generate_kwargs = {'pad_token_id': tokenizer.pad_token_id}
    trainer = Trainer(args, accelerator, model, optimizer, train_dataloader, valid_dataloader,
                      keep_for_metrics_fn=keep_for_metrics_fn, metrics_fn=metrics_fn,
                      ###booydar
                      batch_metrics_fn=batch_metrics_fn,
                      generate_kwargs=generate_kwargs)

    if not args.validate_only:
        # train loop
        trainer.train()
        # make sure all workers are done
        accelerator.wait_for_everyone()
        # run validation after training
        if args.save_best:
            best_model_path = str(Path(args.model_path) / 'model_best')
            logger.info(f'Loading best saved model from {best_model_path}')
            trainer.load(best_model_path)
        if valid_dataloader is not None:
            logger.info('Runnning validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=False, split='valid')
        if test_dataloader is not None:
            logger.info('Runnning validation on test data:')
            trainer.validate(test_dataloader, write_tb=True, split='test')
        trainer.save_metrics(save_path=args.model_path)
    else:
        # run validation, do not write to tensorboard
        logger.info('Running validation on train set:')
        trainer.validate(train_dataloader, split='train', write_tb=True)
        if valid_dataloader is not None:
            logger.info('Running validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=True, split='valid')
        if test_dataloader is not None:
            logger.info('Runnning validation on test data:')
            trainer.validate(test_dataloader, write_tb=True, split='test')
