from pathlib import Path
import json
import pandas as pd

from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
from tqdm.notebook import tqdm

TGT_COLS = ['task_name', 'from_pretrained', 'model_cfg', 'model_cls', 'model_type', 'lr', 'batch_size', 'HVD_SIZE', 'lr_scheduler', 'input_seq_len', 'input_seg_size', 'num_mem_tokens', 'model_path', 'sum_loss']

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


def parse_to_csv(path, out_path, target_cols, metric_names):
    path = Path(path)

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
    
    not_found_cols = [col for col in target_cols if col not in experiments.columns]
    if not_found_cols:
        print(f'{not_found_cols} not found in columns!!\ncolumns:{experiments.columns}')
    
    found_cols = [col for col in target_cols if col in experiments.columns]
    experiments = experiments[found_cols]

    experiments.to_csv(out_path, index=False)

    
    
    
# # HYP

# path = Path('/home/bulatov/bulatov/runs/finetune/debug/hyperpartisan_news_detection')
# metric_names = ['f1', 'precision', 'recall', 'accuracy']
# target_cols = ['f1', 'best_valid_f1', 'precision', 'best_valid_precision', 'recall', 'best_valid_recall', 'accuracy', 'best_valid_accuracy']
# out_path = 'results/debug_hyp.csv'

# parse_to_csv(path, out_path, target_cols, metric_names)

# CNLI

path = Path('/home/bulatov/bulatov/runs/finetune/debug/contract_nli')
metric_names = ['exact_match']
target_cols = TGT_COLS + ['best_valid_exact_match']
out_path = 'results/contract_nli.csv'

parse_to_csv(path, out_path, target_cols, metric_names)

# path = Path('/home/bulatov/bulatov/runs_hyp_good_cnli_ok_080822/finetune/debug/contract_nli')
# metric_names = ['exact_match']
# target_cols = TGT_COLS + ['best_valid_exact_match']
# out_path = 'results/debug_cnli.csv'

# parse_to_csv(path, out_path, target_cols, metric_names)
