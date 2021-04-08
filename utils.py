import importlib
import json
import subprocess
from pathlib import Path

import torch
from transformers import T5Config, T5Tokenizer


def get_cls_by_name(name: str) -> type:
    """Get class by its name and module path.

    Args:
        name (str): e.g., transfomers:T5ForConditionalGeneration, modeling_t5:my_class

    Returns:
        type: found class for `name`
    """
    module_name, cls_name = name.split(':')
    return getattr(importlib.import_module(module_name), cls_name)


def get_git_hash_commit() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def expand_dp_path(path, variables):
    """Expand paths from DeepPavlov configs, uses variables from config's metadata.
    """
    while '{' in path and '}' in path:
        path = path.format(**variables)
    path = Path(path).expanduser()
    return path


def load_experiment(path, t5_configs_path, checkpoint=None, check_commit=True):
    path = Path(path)
    cfg = json.load((path / 'config.json').open('r'))
    model_cfg = Path(t5_configs_path) / cfg['model_cfg'] if cfg['model_cfg'] is not None else None
    model_cls = get_cls_by_name(cfg['model_cls'])
    if check_commit:
        assert cfg['COMMIT'] == get_git_hash_commit(), f"expected commit {cfg['COMMIT']}, " \
                                                       f"but current is {get_git_hash_commit()}"
    # take latest checkpoint
    if checkpoint is None:
        checkpoint = list(sorted(path.glob('*.pth'), key=lambda x: x.stat().st_ctime))[-1]

    if model_cfg is None:
        t5config = T5Config.from_pretrained(cfg['base_model'])
    else:
        t5config = T5Config.from_json_file(model_cfg)

    t5tokenizer = T5Tokenizer.from_pretrained(cfg['base_model'])

    model = model_cls(config=t5config)

    state_dict = torch.load(str(checkpoint), map_location='cpu')
    model.load_state_dict(state_dict["model_state_dict"])
    print(f'Model was loaded from: {checkpoint}')
    model.eval()
    return model, t5tokenizer
