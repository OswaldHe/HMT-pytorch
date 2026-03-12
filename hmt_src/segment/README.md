# Wiki-727K Segmentation

Train sentence-boundary segmentation with either:
- `--model_type bert` using a Hugging Face sequence classifier (`--model_name`),
- `--model_type lstm` using a lightweight BiLSTM classifier with a Hugging Face tokenizer.

Dataset loading mirrors:
- `load_dataset("TankNee/wiki-727k", num_proc=8, trust_remote_code=True)` by default.
- each sample uses `text` + `label` (boundary list).

## Train + Eval + Test

```bash
accelerate launch hmt_src/segment/train.py \
  --dataset_name TankNee/wiki-727k \
  --model_type bert \
  --model_name prajjwal1/bert-small \
  --max_train_docs 100000 \
  --max_valid_docs 10000 \
  --max_test_docs 10000 \
  --pos_weight 5.0 \
  --num_proc 8 \
  --num_epochs 2 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --save_checkpoint \
  --checkpoint_dir /work1/jasoncong/jameszhang23/HMT-OPT-pytorch/checkpoints/wiki727k_bert_small
```

## Weighted CE (Class Imbalance)

Manual positive class weight:

```bash
accelerate launch hmt_src/segment/train.py \
  --dataset_name TankNee/wiki-727k \
  --model_type bert \
  --model_name prajjwal1/bert-small \
  --pos_weight 5.0 \
  --save_checkpoint \
  --checkpoint_dir /work1/jasoncong/jameszhang23/HMT-OPT-pytorch/checkpoints/wiki727k_bert_small_wce
```

Auto positive class weight (`neg_count / pos_count`, capped by `--max_pos_weight`):

```bash
accelerate launch hmt_src/segment/train.py \
  --dataset_name TankNee/wiki-727k \
  --model_type bert \
  --model_name prajjwal1/bert-small \
  --auto_pos_weight \
  --max_pos_weight 20 \
  --save_checkpoint \
  --checkpoint_dir /work1/jasoncong/jameszhang23/HMT-OPT-pytorch/checkpoints/wiki727k_bert_small_auto_wce
```

## Use Alternate Dataset (Optional)

```bash
accelerate launch hmt_src/segment/train.py \
  --dataset_name saeedabc/wiki727k \
  --model_type bert \
  --model_name prajjwal1/bert-small \
  --drop_titles \
  --num_proc 8 \
  --save_checkpoint \
  --checkpoint_dir /work1/jasoncong/jameszhang23/HMT-OPT-pytorch/checkpoints/wiki727k_bert_small_untitled
```

## LSTM Example

```bash
accelerate launch hmt_src/segment/train.py \
  --dataset_name TankNee/wiki-727k \
  --model_type lstm \
  --model_name bert-base-uncased \
  --num_proc 8 \
  --save_checkpoint \
  --checkpoint_dir /work1/jasoncong/jameszhang23/HMT-OPT-pytorch/checkpoints/wiki727k_lstm
```

## Segment A Document

```bash
python hmt_src/segment/segment.py \
  --input_txt /path/to/document.txt \
  --checkpoint_dir /work1/jasoncong/jameszhang23/HMT-OPT-pytorch/checkpoints/wiki727k_bert_small/best \
  --model_type auto \
  --output_json /path/to/segments.json
```

## Segment With BGE Reranker

Use first sentence as query and second sentence as passage.  
If reranker score `< --reranker_threshold`, the second sentence is treated as a new segment start.

```bash
python hmt_src/segment/segment.py \
  --input_txt /path/to/document.txt \
  --model_type reranker \
  --reranker_model_name BAAI/bge-reranker-base \
  --reranker_threshold 0.0 \
  --output_json /path/to/segments_reranker.json
```
