import json
import logging
import re
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import typer
from deeppavlov import evaluate_model, train_evaluate_model_from_config
from horovod import run as hvd_run
from tqdm import tqdm

# fixes error with serializing evaluate_model/train_evaluate_model_from_config funcs during hvd call
# should be before utils importing, seems like because of transformers imports 🤷‍♂️
import dill
dill.extend(False)
import cloudpickle  # noqa: F401, E402
dill.extend(True)

from transformers.models.t5 import T5_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: E402

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None

n_gpus = torch.cuda.device_count()
app = typer.Typer()


def expand_dp_path(path, variables):
    """Expand paths from DeepPavlov configs, uses variables from config's metadata.
    """
    while '{' in path and '}' in path:
        path = path.format(**variables)
    path = Path(path).expanduser()
    return path


def hvd_dp_run(config, fn=evaluate_model, check_metrics=False):
    config = deepcopy(config)
    if n_gpus > 1:
        config['train']['class_name'] = 'dp_hvd_trainer:HvdTorchNNTrainer'
        # hvd and gradient accumulation do not work together in DP currently
        config['chainer']['pipe'][2]['sub_batch_size'] = None
        metrics = hvd_run(fn, args=(config,), np=n_gpus, use_gloo=True)
        # check that metrics from all workers are equal
        if check_metrics:
            splits = list(metrics[0].keys())
            if len(splits) == 0:
                logger.info(f'no evaluation splits were found in metrics: {metrics}')
                return {}
            metrics_names = metrics[0][splits[0]].keys()
            if len(metrics_names) == 0:
                logger.info(f'no metrics were found in evaluation results: {metrics}')
                return {}
            for split in splits:
                for name in metrics_names:
                    if len(set([m[split][name] for m in metrics])) > 1:
                        print(metrics)
                        logger.info('metrics should be equal for all hvd workers! stopping...')
                        exit(1)
        return metrics[0]
    else:
        return fn(config)


def evaluate_checkpoint(pretrained_checkpoint: str,
                        task_config_path: str,
                        eval_batch_size: int = 64,
                        train: bool = False,
                        only_evaluate: bool = False,
                        suffix: str = '',
                        finetuned_model_path: Optional[str] = None,
                        train_batch_size: Optional[int] = None,
                        train_subbatch_size: Optional[int] = None,
                        learning_rate: Optional[float] = None,
                        ) -> dict:
    """Evaluate checkpoint on  folder on tasks from `task_config_path`

    Args:
        pretrained_checkpoint (str): path to pretrained model
        task_config_path (str): path to DP config with task (e.g., from GLUE) to evaluate on
        eval_batch_size (int, optional): batch size for validation. Defaults to 64.
        train (bool, optional): Set to True to train `pretrained_checkpoint` on task, set to False to evaluate
            `finetuned_model_path`. Defaults to False.
        suffix (str, optional): suffix to append to MODEL_PATH, e.g., `bs_32/run_0`. Defaults to ''.
        finetuned_model_path (Optional[str], optional): Path to already finetuned model to evaluate. Defaults to None.
        train_batch_size (Optional[int], optional): Train batch size. Defaults to None, uses value from task config.
        train_subbatch_size (Optional[int], optional): Train subbatch size. Defaults to None, uses value from task
            config.

    Returns:
        dict: evaluation results and meta-info
    """
    assert not (train ^ (finetuned_model_path is None)), f'train: {train}, finetuned_model_path: {finetuned_model_path}'
    hf_model = False
    if str(pretrained_checkpoint) not in T5_PRETRAINED_MODEL_ARCHIVE_LIST:
        pretrained_checkpoint = Path(pretrained_checkpoint).resolve()
    else:
        hf_model = True
    finetuned_model_path = Path(finetuned_model_path).resolve() if finetuned_model_path else None
    task_config_path = Path(task_config_path).resolve()
    task_name = task_config_path.stem
    if task_config_path.parent.stem != 'dp_configs':
        task_name = f'{task_config_path.parent.stem}/{task_name}'
    config = json.load(task_config_path.open('r'))

    # use evaluation targets from task config
    # config['train']['evaluation_targets'] = ['valid']
    if hf_model:
        # todo: make more general, without hardcoded ./runs path
        config['metadata']['variables']['PRETRAINED_PATH'] = str(Path('./runs').resolve() / pretrained_checkpoint)
        config['chainer']['pipe'][2]['pretrained_model'] = str(pretrained_checkpoint)
    else:
        config['metadata']['variables']['PRETRAINED_PATH'] = str(pretrained_checkpoint.parent)
    # train model & get metrics
    if train and not only_evaluate:
        config['metadata']['variables']['MODEL_PATH'] = '{PRETRAINED_PATH}/' + task_name + '/' + suffix
        if not hf_model:
            config['chainer']['pipe'][2]['checkpoint'] = '{PRETRAINED_PATH}/' + str(pretrained_checkpoint.name)
        if train_batch_size:
            config['train']['batch_size'] = train_batch_size
        if train_subbatch_size and n_gpus == 1:
            config['chainer']['pipe'][2]['sub_batch_size'] = train_subbatch_size
        if learning_rate:
            config['chainer']['pipe'][2]['optimizer_parameters']['lr'] = learning_rate
        # save config
        model_path = expand_dp_path(config['metadata']['variables']['MODEL_PATH'], config['metadata']['variables'])
        model_path.mkdir(parents=True, exist_ok=True)
        json.dump(config, (model_path / 'config.json').open('w'), indent=2)
        metrics = hvd_dp_run(config, fn=train_evaluate_model_from_config, check_metrics=True)
    elif only_evaluate:
        finetuned_model_config = json.load((finetuned_model_path.parent / 'config.json').open('r'))
        config['chainer']['pipe'][2] = deepcopy(finetuned_model_config['chainer']['pipe'][2])
        config['metadata']['variables']['MODEL_PATH'] = finetuned_model_config['metadata']['variables']['MODEL_PATH']
        config['chainer']['pipe'][2]['load_path'] = str('{MODEL_PATH}/' + str(finetuned_model_path.name.split('.')[0]))
        config['train']['tensorboard_log_dir'] = None
        config['train']['batch_size'] = eval_batch_size
        config['chainer']['pipe'][2]['sub_batch_size'] = eval_batch_size
        metrics = hvd_dp_run(config, fn=evaluate_model, check_metrics=True)
    else:
        # nothing to do
        raise RuntimeError('evaluate_checkpoint: set train or only_evaluate to true')

    load_path = expand_dp_path(config['chainer']['pipe'][2]['load_path'], config['metadata']['variables'])

    if not finetuned_model_path:
        finetuned_model_path = load_path
    if hf_model:
        # todo: ./runs hardcoded again
        finetuned_model = str(finetuned_model_path.relative_to(('./runs' / pretrained_checkpoint).resolve()).parent)
    else:
        finetuned_model = str(finetuned_model_path.relative_to(pretrained_checkpoint.parent).parent)
    return {'pretrained_model': pretrained_checkpoint.parent.name if not hf_model else pretrained_checkpoint.name,
            'pretrained_checkpoint': pretrained_checkpoint.name,
            'finetuned_model': finetuned_model,
            'finetuned_checkpoint': finetuned_model_path.name,
            'task': task_name,
            **{f'{split}_{m}': metrics[split][m] for split in metrics for m in metrics[split]}
            }


@app.command()
def train_mixture(task_config: Path = typer.Option(...),
                  pretrained_checkpoint: Path = typer.Option(...),
                  suffix: str = typer.Option(''),
                  train_batch_size: Optional[int] = typer.Option(None),
                  train_subbatch_size: Optional[int] = typer.Option(None),
                  learning_rate: Optional[float] = typer.Option(None, '--lr')):
    logger.info(f'starting to train {pretrained_checkpoint} on tasks:')
    logger.info(f'\t{task_config.name}')
    _ = evaluate_checkpoint(pretrained_checkpoint, task_config, suffix=suffix, train=True,
                            train_batch_size=train_batch_size, train_subbatch_size=train_subbatch_size,
                            learning_rate=learning_rate
                            )
    logger.info('train mixture - DONE')


@app.command()
def mixture(checkpoint: Path = typer.Option(...),
            task_config: Path = typer.Option(...),
            pretrained_checkpoint: Path = typer.Option(...),
            eval_batch_size: int = typer.Option(64),
            save_best: bool = typer.Option(False),
            ):
    """eval finetuned mixture checkpoint on each task independently
    """
    checkpoints = [checkpoint]
    checkpoints_dir = checkpoint.parent
    if checkpoint.is_dir():
        checkpoints = sorted(checkpoint.glob('*.pth.tar'), key=lambda x: x.stat().st_ctime)[::-1]
        checkpoints_dir = checkpoint
        logger.info(f'starting to evaluate {len(checkpoints)} checkpoints from {checkpoint} on tasks:')
    else:
        logger.info(f'starting to evaluate checkpoint {checkpoint} on tasks:')

    task_configs = [task_config]
    if task_config.is_dir():
        task_configs = [task for task in task_config.glob('*json') if 'mixture' not in task.name]

    for task in task_configs:
        logger.info(f'\t{task.name}')

    results = []
    with tqdm(total=len(checkpoints) * len(task_configs), smoothing=0.0) as pbar:
        for checkpoint in checkpoints:
            for config_path in task_configs:
                eval_results = evaluate_checkpoint(pretrained_checkpoint, config_path, eval_batch_size,
                                                   finetuned_model_path=checkpoint, only_evaluate=True)
                eval_results['is_mixture'] = True
                results += [eval_results]
                pbar.update()
    # todo: write results line by line as soon as we get them?
    results = pd.DataFrame(results)
    results.to_csv(checkpoints_dir / 'metrics.csv')

    if save_best:
        (checkpoints_dir / 'best_ckpts').mkdir(exist_ok=True)
        metrics = [c for c in results.columns if 'valid' in c]
        best_checkpoints = []
        with (checkpoints_dir / 'report.txt').open('w') as fout:
            for task in results['task'].unique():
                task_results = results[results['task'] == task]
                best_run = task_results.sort_values(metrics, ascending=False).dropna(axis=1, how='all').iloc[0]
                fout.write(f'{task}:\n')
                for m in [k for k in best_run.keys() if 'valid' in k]:
                    fout.write(f'\t{m}: {best_run[m]}\n')
                best_checkpoints += [best_run["finetuned_checkpoint"]]
                fout.write(f'\tbest_checkpoint: {best_run["finetuned_checkpoint"]}\n')
        for ckpt in set(best_checkpoints):
            shutil.copy(checkpoints_dir / ckpt, checkpoints_dir / 'best_ckpts' / ckpt)

    logger.info('eval mixture - DONE')
    return results


@app.command()
def single(task_config: Path = typer.Option(...),
           pretrained_checkpoint: Path = typer.Option(...),
           suffix: str = typer.Option(''),
           eval_batch_size: int = typer.Option(64),
           train_batch_size: Optional[int] = typer.Option(None),
           train_subbatch_size: Optional[int] = typer.Option(None),
           learning_rate: Optional[float] = typer.Option(None, '--lr')
           ):
    """train&eval pretrained_checkpoint on each task independently
    """
    # todo: add running multiple trainings on several gpus
    logger.info(f'starting to train/eval {pretrained_checkpoint} on tasks:')
    task_configs = [task_config]
    if task_config.is_dir():
        task_configs = [task for task in task_config.glob('*json') if 'mixture' not in task.name]

    for task in task_configs:
        logger.info(f'\t{task.name}')

    train_subbatch_size = train_batch_size if not train_subbatch_size else train_subbatch_size

    results = []
    for config_path in tqdm(task_configs, smoothing=0.0):
        eval_results = evaluate_checkpoint(pretrained_checkpoint, config_path, eval_batch_size, suffix=suffix,
                                           train=True,
                                           train_batch_size=train_batch_size, train_subbatch_size=train_subbatch_size,
                                           learning_rate=learning_rate
                                           )
        eval_results['is_mixture'] = False
        if not str(pretrained_checkpoint) in T5_PRETRAINED_MODEL_ARCHIVE_LIST:
            pd.DataFrame([eval_results]).to_csv(pretrained_checkpoint.parent /
                                                eval_results['finetuned_model'] / 'metrics.csv')
        else:
            pd.DataFrame([eval_results]).to_csv('./runs' / pretrained_checkpoint /
                                                eval_results['finetuned_model'] / 'metrics.csv')
        results += [eval_results]

    logger.info('train/eval single - DONE')
    return pd.DataFrame(results)


def get_name_and_run(s):
    model_name = ''
    run = 0
    g1, g2, g3 = re.match(r'(.*)/run_(\d+)|(.*)', s).groups()
    if g1:
        model_name = g1
        run = int(g2)
    else:
        model_name = g3
    return model_name, run


@app.command()
def collect_metrics(pretrained_checkpoint: Path = typer.Option(...),
                    clean: bool = typer.Option(False),
                    force: bool = typer.Option(False),
                    ):
    """collect all metrics from all experiments for exact pretrained checkpoint
    """
    logger.info(f'collecting metrics for {pretrained_checkpoint}')
    results = None
    if pretrained_checkpoint.is_dir():
        # e.g., ./runs/t5-small
        metrics_files = pretrained_checkpoint.rglob('metrics.csv')
    else:
        metrics_files = pretrained_checkpoint.parent.rglob('metrics.csv')
    for m in metrics_files:
        m = pd.read_csv(m, index_col=0)
        # rename to be compatible with deprecated format of metrics.csv
        m = m.rename({
                    'mixture_model': 'finetuned_model',
                    'mixture_checkpoint': 'finetuned_checkpoint'
                    }, axis=1)
        if results is None:
            results = m
        else:
            results = results.append(m, ignore_index=True, sort=False)

    # take results for only one pretrained_checkpoint
    results = results[results['pretrained_checkpoint'] == pretrained_checkpoint.name]
    # add fintuned model ckpt_path
    results['ckpt_path'] = results['pretrained_model'] + '/' + results['finetuned_model']
    results['ckpt_path'] += '/' + results['finetuned_checkpoint']
    results['ckpt_path'] = results['ckpt_path'].apply(lambda x: x + '.pth.tar' if x.endswith('model') else x)
    metrics = [c for c in results.columns if 'valid' in c]
    best_models = []
    for task in results['task'].unique():
        task_results = results[(results['task'] == task)]
        task_results['finetuned_model_name'] = task_results['finetuned_model'].apply(lambda x: get_name_and_run(x)[0])
        task_results['finetuned_model_run'] = task_results['finetuned_model'].apply(lambda x: get_name_and_run(x)[1])
        task_results = task_results.dropna(axis=1, how='all')
        task_metrics = sorted(list(set(task_results.columns) & set(metrics)))
        best_run = task_results.sort_values(task_metrics, ascending=False).iloc[0]

        def _take_best_by_metrics(data):
            return data.sort_values(task_metrics, ascending=False).iloc[0]

        # take max for each run (in mixtures we take the best checkpoint)
        task_results = task_results.groupby(['finetuned_model_name', 'finetuned_model_run'])[task_metrics].agg(
            _take_best_by_metrics)
        # take avg between runs
        task_results = task_results.groupby(['finetuned_model_name']).agg(['mean', 'std'])
        print(f'task: {task}')
        print('models:')
        task_results = task_results.sort_values([(m, 'mean') for m in task_metrics], ascending=False)
        for i in range(len(task_results)):
            print(f'\t{task_results.iloc[i].name}')
            for m in task_metrics:
                print(f"\t\t{m}: {task_results.iloc[i][m]['mean']:.5f}+-{task_results.iloc[i][m]['std']:.5f}")
        print('\nbest_run:')
        print('\tfinetuned_model:', best_run['finetuned_model'])
        print('\tfinetuned_checkpoint:', best_run['finetuned_checkpoint'])
        best_models += [best_run['ckpt_path']]
        for m in task_metrics:
            print(f"\t{m}: {best_run[m]}")
        print('-'*75)

    if clean:
        to_clean_dir = pretrained_checkpoint if pretrained_checkpoint.is_dir() else pretrained_checkpoint.parent
        for dir in to_clean_dir.iterdir():
            for p in dir.rglob('*.pth*'):
                if any(str(p).endswith(bm) for bm in results['ckpt_path']):
                    if not any(str(p).endswith(bm) for bm in best_models) and 'best_ckpts' not in str(p):
                        logger.info(f'  DELETE   - {p}')
                        delete = False if not force else True
                        if not force:
                            delete = typer.confirm("Do you really want to delete?")
                        if delete:
                            p.unlink()
                    else:
                        logger.info(f'KEEP BEST  - {p}')
                else:
                    logger.info(f'   SKIP    - {p}')
        logger.info('checkpoints cleaned, saved only the best')

    logger.info('collecting metrics - DONE')


if __name__ == '__main__':
    """
    examples:

    training pretrained_checkpoint on each glue task independently and evaluating:
    export CUDA_VISIBLE_DEVICES=6; python evaluate_model.py single \
        --pretrained-checkpoint ./runs/small_wiki_bs_128/model_1100000.pth \
        --task-config ./dp_configs/glue \
        --suffix bs_32/run_0 \
        --train-batch-size 32

    evaluating checkpoint (trained on all tasks as mixture) on each glue task:
    export CUDA_VISIBLE_DEVICES=0; python evaluate_model.py mixture \
        --checkpoint ./runs/small_wiki_bs_128/glue/mixture/bs_128/ \
        --pretrained-checkpoint ./runs/small_wiki_bs_128/model_1100000.pth \
        --task-config ./dp_configs/glue \
        --save-best

    python evaluate_model.py collect-metrics \
        --pretrained-checkpoint ./runs/small_wiki_bs_128/model_1100000.pth | less
    """
    app()
