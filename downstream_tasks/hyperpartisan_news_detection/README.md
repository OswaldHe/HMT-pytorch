# Hyperpartisan News Detection dataset
Binary classification of long text articles (50% > 512 tokens).
This is a small dataset (645 train samples), but could be good for initial experiments.

data: https://zenodo.org/record/1489920#.YpY8ZSJBz0o

> The data contains only articles for which a consensus among the crowdsourcing workers existed. It contains a total of 645 articles. Of these, 238 (37%) are hyperpartisan and 407 (63%) are not.

## Train/dev/test splits
We follow Longformer procedure to get train/dev/test split:
- get files (`hp_preprocess.py`, `hp-splits.json`) from https://github.com/allenai/longformer/pull/112/files
- follow instructions from `hp_preprocess.py`
- `hp-splits.json` defines how the dataset was split in Longformer paper

This results in train/dev/test = 516/64/65.

Documents length in bert-uncased tokens:
```
min: 19.0 max: 5538.0 mean: 744.2 median: 548.0 q0.9: 1529.6
```

There is a newer version of the dataset https://zenodo.org/record/5776081 with train/valid/test split from SemEval 2019 Task 4 organizers. However, we use Longformer splits to compare results with them.

## Fine-tuning
Example of bash scripts with hyperparams grid search and finetuning for encoder-only `finetune_hyp.sh` and encoder-decoder models `finetune_hyp_enc_dec.sh`.
You will need to modify `DATA_PATH` and check other parameters as well.
```bash
CUDA_VISIBLE_DEVICES=0 NP=1 ./finetune_hyp.sh
```
