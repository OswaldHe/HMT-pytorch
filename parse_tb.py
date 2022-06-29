from pathlib import Path
import json
import pandas as pd

from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
from tqdm.notebook import tqdm


def parse_tensorboard(path, scalars):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}



# HYP

path = Path('/home/bulatov/bulatov/runs/finetune/hyperpartisan_news_detection')
metric_names = ['f1', 'precision', 'recall', 'accuracy']

logs = list(path.glob('**/*tfevents*'))

experiments = []
for p in tqdm(logs):
    expr = json.load(open(p.parent / 'config.json', 'r'))
    metrics = {}
    try:
        metrics = parse_tensorboard(str(p), [f'{m}/iterations/valid' for m in metric_names])
    except Exception as e:
        print(f'error: {e}\n\tskip: {p}')
    try:
        metrics_test = parse_tensorboard(str(p), [f'{m}/iterations/test' for m in metric_names])
    except Exception as e:
        metrics_test = {}
        print(f'error: {e}\n\t no test metrics in: {p}')
    metrics.update(metrics_test)

    if len(metrics) == 0:
        continue
    for m in metric_names:
        if f'{m}/iterations/test' in metrics:
            expr[m] = metrics[f'{m}/iterations/test']['value'].item()
        expr[f'best_valid_{m}'] = metrics[f'{m}/iterations/valid']['value'].max()
    experiments += [expr]

experiments = pd.DataFrame(experiments)
target_cols = ['from_pretrained', 'model_cfg', 'model_cls', 'model_type', 'lr', 'batch_size', 'HVD_SIZE', 'lr_scheduler', 'input_seq_len', 'input_seg_size', 'num_mem_tokens', 'model_path', ]

target_cols += ['f1', 'best_valid_f1', 'precision', 'best_valid_precision', 'recall', 'best_valid_recall', 'accuracy', 'best_valid_accuracy']
experiments = experiments[target_cols]
out_path = 'results/hyp.csv'

experiments.to_csv(out_path, index=False)


# HYP gridsearch

path = Path('/home/bulatov/bulatov/runs/finetune/gridsearch/hyperpartisan_news_detection')
metric_names = ['f1', 'precision', 'recall', 'accuracy']

logs = list(path.glob('**/*tfevents*'))

experiments = []
for p in tqdm(logs):
    expr = json.load(open(p.parent / 'config.json', 'r'))
    metrics = {}
    try:
        metrics = parse_tensorboard(str(p), [f'{m}/iterations/valid' for m in metric_names])
    except Exception as e:
        print(f'error: {e}\n\tskip: {p}')
    try:
        metrics_test = parse_tensorboard(str(p), [f'{m}/iterations/test' for m in metric_names])
    except Exception as e:
        metrics_test = {}
        print(f'error: {e}\n\t no test metrics in: {p}')
    metrics.update(metrics_test)

    if len(metrics) == 0:
        continue
    for m in metric_names:
        if f'{m}/iterations/test' in metrics:
            expr[m] = metrics[f'{m}/iterations/test']['value'].item()
        expr[f'best_valid_{m}'] = metrics[f'{m}/iterations/valid']['value'].max()
    experiments += [expr]

experiments = pd.DataFrame(experiments)
target_cols = ['from_pretrained', 'model_cfg', 'model_cls', 'model_type', 'lr', 'batch_size', 'HVD_SIZE', 'lr_scheduler', 'input_seq_len', 'input_seg_size', 'num_mem_tokens', 'model_path', ]

target_cols += ['f1', 'best_valid_f1', 'precision', 'best_valid_precision', 'recall', 'best_valid_recall', 'accuracy', 'best_valid_accuracy']
experiments = experiments[target_cols]
out_path = 'results/hyp_grid.csv'

experiments.to_csv(out_path, index=False)



# CNLI

path = Path('/home/bulatov/bulatov/runs/finetune/gridsearch/contract_nli')
metric_names = ['exact_match']

logs = list(path.glob('**/*tfevents*'))

experiments = []
for p in tqdm(logs):
    expr = json.load(open(p.parent / 'config.json', 'r'))
    metrics = {}
    try:
        metrics = parse_tensorboard(str(p), [f'{m}/iterations/valid' for m in metric_names])
    except Exception as e:
        print(f'error: {e}\n\tskip: {p}')
    try:
        metrics_test = parse_tensorboard(str(p), [f'{m}/iterations/test' for m in metric_names])
    except Exception as e:
        metrics_test = {}
        print(f'error: {e}\n\t no test metrics in: {p}')
    metrics.update(metrics_test)

    if len(metrics) == 0:
        continue
    for m in metric_names:
        if f'{m}/iterations/test' in metrics:
            expr[m] = metrics[f'{m}/iterations/test']['value'].item()
        expr[f'best_valid_{m}'] = metrics[f'{m}/iterations/valid']['value'].max()
    experiments += [expr]

experiments = pd.DataFrame(experiments)
target_cols = ['from_pretrained', 'model_cfg', 'model_cls', 'model_type', 'lr', 'batch_size', 'HVD_SIZE', 'lr_scheduler', 'input_seq_len', 'input_seg_size', 'num_mem_tokens', 'model_path']

target_cols += ['best_valid_exact_match']
experiments = experiments[target_cols]
out_path = 'results/cnli.csv'

experiments.to_csv(out_path, index=False)