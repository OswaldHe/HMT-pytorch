from pathlib import Path
import json
import pandas as pd

from tensorboard.backend.event_processing import event_accumulator
import pandas as pd


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


path = Path('/home/bulatov/bulatov/runs/finetune/hyperpartisan_news_detection')
metric_names = ['f1', 'precision', 'recall', 'accuracy']

# path = Path('/home/bulatov/bulatov/runs/finetune/contract_nli')
# metric_names = ['exact_match']

logs = list(path.glob('**/*tfevents*'))
len(logs)

from tqdm.notebook import tqdm

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

target_cols += ['f1', 'best_valid_f1', 'precision', 'best_valid_precision', 'recall', 'best_valid_recall', 'accuracy', 'best_valid_accuracy']
experiments = experiments[target_cols]
out_path = 'results/hyp.csv'

# target_cols += ['best_valid_exact_match']
# experiments = experiments[target_cols]
# out_path = 'results/cnli.csv'

# experiments.groupby(['from_pretrained', 'lr', 'lr_scheduler'])[['best_valid_exact_match']].agg(['mean', 'std'])

experiments.to_csv(out_path, index=False)

# from tqdm.notebook import tqdm
# def collect_metrics(path, metrics_names):
#     logs = list(path.glob('**/*tfevents*'))
#     experiments = []
#     for p in tqdm(logs):
#         expr = json.load(open(p.parent / 'config.json', 'r'))
#         metrics = {}
#         try:
#             metrics = parse_tensorboard(str(p), [f'{m}/iterations/valid' for m in metrics_names])
#         except Exception as e:
#             print(f'error: {e}\n\tskip: {p}')
#             continue
#         try:
#             metrics_test = parse_tensorboard(str(p), [f'{m}/iterations/test' for m in metrics_names])
#         except Exception as e:
#             metrics_test = {}
#             print(f'no test metrics in: {p}')
#         metrics.update(metrics_test)

#         for m in metrics_names:
#             if f'{m}/iterations/test' in metrics:
#                 expr[m] = metrics[f'{m}/iterations/test']['value'].item()
#             expr[f'best_valid_{m}'] = metrics[f'{m}/iterations/valid']['value'].max()
#         experiments += [expr]
#     return experiments

# # path = Path('/home/jovyan/t5-experiments/runs/finetune/hyperpartisan_news_detection/')
# path = Path('/home/bulatov/bulatov/runs/finetune/hyperpartisan_news_detection')
# metrics_names = ['f1', 'precision', 'recall', 'accuracy']
# # metrics_names = ['exact_match']

# experiments = collect_metrics(path, metrics_names)
# experiments = pd.DataFrame(experiments)

# metrics = [c for c in experiments.columns if c in metrics_names]
# metrics += [c for c in experiments.columns if 'best_valid_' in c and c.split('best_valid_')[1] in metrics_names]
# metrics

# experiments = experiments[['from_pretrained', 'model_cfg', 'model_cls', 'model_type', 'lr', 'batch_size', 'HVD_SIZE', 'lr_scheduler', 'input_seq_len'] +
#                           metrics + ['model_path']]

# g = experiments.groupby(['from_pretrained', 'lr', 'lr_scheduler'])[metrics].agg(['mean', 'std']).reset_index()
# g[:10]

# # the best hyperparameters per model by metric
# metric_name_to_select_best = 'f1'
# grouped = g[g.groupby(['from_pretrained']).transform(max)[metric_name_to_select_best]['mean'] == g[metric_name_to_select_best]['mean']]

# out_path = 'results/res_df.csv'
# grouped.to_csv(out_path, index=False)