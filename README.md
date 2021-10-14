# t5-experiments

## Install requirements
This repo is based on 🤗 Transfomers implementation of the T5 model.
Data processing pipeline is used from the original [T5 repository](https://github.com/google-research/text-to-text-transfer-transformer) in all settings. Horovod is used for multi-gpu and multi-node training. DeepPavlov library is used for running and tracking fine-tuning experiments and evalutaion.

Install requirements after cloning the repo:
```bash
grep -v "^#" requirements.txt | xargs -n 1 -L 1 pip install
```

###  Install Horovod
Depending on your setup just `pip install horovod==0.23.0` might work.

Building Horovod with NCCL for PyTorch:
```bash
HOROVOD_NCCL_HOME=... HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod[pytorch]==0.23.0 --no-binary=horovod
```

Currently, T5 is using `tensorflow==2.6.0` and Horovod started to support tf2.6 since 0.22.1.

For further details check Horovod documentation: https://horovod.readthedocs.io/en/stable/install_include.html

## Pre-training
### T5-small baseline
```bash
export CUDA_VISIBLE_DEVICES=4,5; horovodrun --gloo -np 2 python run_t5_pretraining.py \
        --batch_size 32 \
        --gradient_accumulation_steps 2 \
        --save_interval 100000 \
        --log_interval 500 \
        --iters 1100000 \
        --data_path ~/data/ThePile/Wikipedia/preprocessed_shards \
        --model_path ./runs/small_wiki_bs_128 \
        --input_seq_len 512 \
        --target_seq_len 192 \
        --lr 5e-05 \
        --weight_decay 1e-05 \
        --model_cfg ./t5configs/t5-small.json \
        --model_cls modeling_t5:T5ForConditionalGeneration
```

### T5-base with custom layers:
and continue interrupted training
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3; horovodrun --gloo -np 4 python run_t5_pretraining.py \
        --batch_size 8 \
        --gradient_accumulation_steps 4 \
        --save_interval 75000 \
        --log_interval 500 \
        --iters 1000000 --data_path ~/data/ThePile/Wikipedia/preprocessed_shards \
        --model_path ./runs/base_wiki_enc_only_cdq_fixed_pos_wo_tanh \
        --input_seq_len 512 \
        --target_seq_len 192 \
        --lr 5e-05 \
        --weight_decay 1e-05 \
        --model_cls modeling_t5:T5ForConditionalGeneration \
        --model_cfg t5configs/t5-base-only-cdQ.json \
        --init_checkpoint ./runs/base_wiki_enc_only_cdq_fixed_pos_wo_tanh/model_150000.pth
```

## Fine-tuning with DeepPavlov
`python -m deeppavlov train config_name`

Gradient accumulation for `dp:T5Text2TextModel`, e.g.:
- `batch_size`: 32
- `sub_batch_size`: 16

means that full batch of size `batch_size` will be splited on two sub-batches of size `sub_batch_size` to accumulate their gradients.

### Fine-tuning on GLUE
Base configuration files are at `./dp_configs/glue`

Fine-tuning and evaluation could be done with command:
```bash
export CUDA_VISIBLE_DEVICES=6; python evaluate_model.py single \
        --pretrained-checkpoint ./runs/small_wiki_bs_128/model_1100000.pth \
        --task-config ./dp_configs/glue \
        --suffix bs_32/run_0 \
        --train-batch-size 32
```
`pretrained-checkpoint` is a path to pretrained checkpoint that would be trained and evaluated, `task-config` is a
folder with DP configs (or single DP config), `suffix` would be appended to a model path. Check `evaluate_model.py` for
more details.

#### GLUE mixture from T5
config: `./dp_configs/glue/glue_mixture.json`

Use `save_every_n_batches` parameter to save the model, set `metrics: []` and `evaluation_targets: []` in DP configs.

Train model on datasets mixture, check all available options in `evaluate_model.py:train_mixture()`:
```bash
export CUDA_VISIBLE_DEVICES=1; python evaluate_model.py train-mixture \
        --pretrained-checkpoint ./runs/small_wiki_bs_128/model_1100000.pth \
        --task-config ./dp_configs/glue/glue_mixture.json  \
        --suffix bs_128 \
        --train-batch-size 128
```

Evaluation for all checkpoints in `checkpoint` folder, saves best checkpoints and evaluation results:
```bash
export CUDA_VISIBLE_DEVICES=0; python evaluate_model.py mixture \
        --checkpoint ./runs/small_wiki_bs_128/glue/mixture/bs_128/ \
        --pretrained-checkpoint ./runs/small_wiki_bs_128/model_1100000.pth \
        --task-config ./dp_configs/glue \
        --save-best
```

#### Collecting results
To get the best scores for all fine-tuned models and tasks run:
```bash
python evaluate_model.py collect-metrics \
        --pretrained-checkpoint ./runs/small_wiki_bs_128/model_1100000.pth --clean > report.txt
```
use `--clean` option to delete all models checkpoints except the best ones for each task.

### Prepare submission for GLUE Leaderboard:
**TBD**

#### QQP
QQP is currently not available via tfds: https://github.com/tensorflow/datasets/pull/3031

to hot-fix this go to the source code of installed tfds `tensorflow_datasets/text/glue.py:215` and replace QQP data url with https://dl.fbaipublicfiles.com/glue/data/QQP.zip

### Fine-tuning on WMT
WMT configs could be found in `./dp_configs/wmt`

Training with Horovod+DeepPavlov:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; horovodrun --gloo -np 8 python -m deeppavlov train ./dp_configs/ende_hvd.json
```

Multi-gpu training and evaluating with `evaluate_model.py` (recommended):
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; python evaluate_model.py single \
        --pretrained-checkpoint ./runs/small_wiki_bs_128/model_1100000.pth \
        --task-config ./dp_configs/wmt/ende.json \
        --suffix bs_128_hvd/run_0 \
        --train-batch-size 16 \
        --lr 5e-05
```

## FP16
Install APEX https://github.com/NVIDIA/apex#quick-start
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

apex.amp is moved to torch.cuda.amp https://github.com/NVIDIA/apex/issues/818, but:

speed: `APEX O1` < `torch.cuda.amp` < `APEX O2`

resources (unordered):
 - https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
 - https://pytorch.org/docs/stable/notes/amp_examples.html
 - https://spell.ml/blog/mixed-precision-training-with-pytorch-Xuk7YBEAACAASJam
 - https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/
 - https://github.com/horovod/horovod/issues/1089
 - https://github.com/NVIDIA/apex/issues/818

### FP16 for t5 pretraining
add `--fp16` and `--apex_opt_lvl O2` or `--apex_opt_lvl O1` (default) as arguments to `run_t5_pretraining.py`

## Adafactor optimizer
Adafactor optimizer might be used in two settings:

- auto learning rate (`--scale_parameter`, `--relative_step`, `--warmup_init` for `run_t5_pretraining.py` or in DP config in `optimizer_parameters` section)
- external learning rate (`--lr 0.001` and do not use `--relative_step` flag)

`warmup_init` works only with `relative_step`, so you can't setup your own learning rate with warmup.

e.g. for DP config
```json
"optimizer": "Adafactor",
"optimizer_parameters": {
        "lr": 1e-03,
        "weight_decay": 0.0,
        "scale_parameter": true,
        "relative_step": false,
        "warmup_init": false
}
```