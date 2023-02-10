from pathlib import Path
import json
import pandas as pd

from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
from tqdm.notebook import tqdm

TGT_COLS = ['task_name', 'from_pretrained', 'model_cfg', 'model_cls', 'model_type', 'lr', 'batch_size', 'HVD_SIZE', 'lr_scheduler', 'input_seq_len', 'max_n_segments', 'num_mem_tokens','segment_ordering','padding_side', 'model_path', 'sum_loss', 'inter_layer_memory', 'memory_layers', 'share_memory_layers','reconstruction_loss_coef', 'num_steps']

SILENT = True
def parse_tensorboard(path, scalars, silent=SILENT):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    # assert all(
    #     s in ea.Tags()["scalars"] for s in scalars
    # ),"" if silent else "some scalars were not found in the event accumulator"
    # return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}
    found_scalars = [s for s in scalars if s in ea.Tags()['scalars']]
    return {k: pd.DataFrame(ea.Scalars(k)) for k in found_scalars}


def parse_to_csv(path, out_path, target_cols, metric_names, silent=SILENT):
    path = Path(path)

    logs = list(path.glob('**/*tfevents*'))

    experiments = []
    for p in tqdm(logs):
        expr = json.load(open(p.parent / 'config.json', 'r'))
        metrics = {}
        try:
            metrics = parse_tensorboard(str(p), [f'{m}/iterations/valid' for m in metric_names])
        except Exception as e:
            if not silent:
                print(f'error: {e}\n\tskip: {p}')
        try:
            metrics_test = parse_tensorboard(str(p), [f'{m}/iterations/test' for m in metric_names])
        except Exception as e:
            metrics_test = {}
            if not silent:
                print(f'error: {e}\n\t no test metrics in: {p}')
        metrics.update(metrics_test)

        if len(metrics) == 0:
            continue
        for m in metric_names:
            if f'{m}/iterations/test' in metrics:
                expr[m] = metrics[f'{m}/iterations/test']['value'].item()
            expr[f'best_valid_{m}'] = metrics[f'{m}/iterations/valid']['value'].max()

        # print(parse_tensorboard(str(p), ['loss/iterations/train'])['loss/iterations/train'].step)
        parsed = parse_tensorboard(str(p), ['loss/iterations/train'])
        if 'loss/iterations/train' in parsed:
            expr['num_steps'] = parsed['loss/iterations/train'].step.max()
        experiments += [expr]

    experiments = pd.DataFrame(experiments)
    # print('\n\ncolumns: ', experiments.columns)
    
    not_found_cols = [col for col in target_cols if col not in experiments.columns]
    if not_found_cols:
        if not silent:
            print(f'{not_found_cols} not found in columns!!\ncolumns:{experiments.columns}')
    
    found_cols = [col for col in target_cols if col in experiments.columns]
    experiments = experiments[found_cols]
    # print('\n\ncolumns: ', experiments.columns)

    experiments.to_csv(out_path, index=False)

    
    
    
# # # HYP

# path = Path('/home/bulatov/bulatov/runs/finetune/debug/hyperpartisan_news_detection')
# metric_names = ['f1', 'precision', 'recall', 'accuracy']
# target_cols = ['f1', 'best_valid_f1', 'precision', 'best_valid_precision', 'recall', 'best_valid_recall', 'accuracy', 'best_valid_accuracy']
# out_path = 'results/hyp_new.csv'

# parse_to_csv(path, out_path, target_cols, metric_names)

# # CNLI

path = Path('/home/bulatov/bulatov/RMT_light/runs/debug/contract_nli')
metric_names = ['exact_match']
target_cols = TGT_COLS + ['best_valid_exact_match']
out_path = 'results/contract_nli_old.csv'

parse_to_csv(path, out_path, target_cols, metric_names)


path = Path('/home/bulatov/bulatov/RMT_light/runs/framework/contract_nli')
metric_names = ['exact_match']
target_cols = TGT_COLS + ['best_valid_exact_match']
out_path = 'results/contract_nli.csv'

parse_to_csv(path, out_path, target_cols, metric_names)


path = Path('/home/bulatov/bulatov/RMT_light/runs/test/contract_nli')
metric_names = ['exact_match']
target_cols = TGT_COLS + ['best_valid_exact_match']
out_path = 'results/contract_nli-2.csv'

parse_to_csv(path, out_path, target_cols, metric_names)


# # QAsper

path = Path('/home/bulatov/bulatov/RMT_light/runs/')
metric_names = ['f1']
target_cols = TGT_COLS + ['best_valid_f1']
out_path = 'results/qasper.csv'

parse_to_csv(path, out_path, target_cols, metric_names)

# path = Path('/home/bulatov/bulatov/runs_hyp_good_cnli_ok_080822/finetune/debug/contract_nli')
# metric_names = ['exact_match']
# target_cols = TGT_COLS + ['best_valid_exact_match']
# out_path = 'results/debug_cnli.csv'

# parse_to_csv(path, out_path, target_cols, metric_names)

# path = Path('/home/bulatov/bulatov/runs/finetune/debug/qmsum')
# metric_names = ['rouge/geometric_mean']
# target_cols = TGT_COLS + ['best_valid_rouge/geometric_mean']
# out_path = 'results/qmsum.csv'

# parse_to_csv(path, out_path, target_cols, metric_names)


# # quality

# path = Path('/home/bulatov/bulatov/runs/finetune/debug/quality')
# metric_names = ['exact_match']
# target_cols = TGT_COLS + ['best_valid_exact_match']
# out_path = 'results/quality_new.csv'

# parse_to_csv(path, out_path, target_cols, metric_names)