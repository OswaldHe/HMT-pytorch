#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_NAMES=(bert-base-cased roberta-base microsoft/deberta-v3-base google/electra-base-discriminator)
MODEL_TYPE=encoder
MODEL_CLSS=(Bert Roberta DebertaV2 Electra)
TASK_NAME=contract_nli
# dataset TRAIN SIZE 7191, 20 epochs with total_batch_size 32 -> ~4500 iterations

# BART 256 512 1024
# LED 1024 4096 16384
SRC_LEN=256
# BART 1024
# LED 1024
# 
TGT_LEN=1024

METRIC=exact_match

for (( i=0; i<${#MODEL_NAMES[@]}; i++ ))
do
MODEL_NAME=${MODEL_NAMES[i]}
MODEL_CLS=${MODEL_CLSS[i]}
for LR in 1e-04 5e-05 1e-05
do
for SCHEDULER in linear constant_with_warmup
do
for N in 1 2 3
do
horovodrun --gloo -np $NP python run_finetuning_scrolls.py \
        --task_name $TASK_NAME \
        --model_path ./runs/finetune/$TASK_NAME/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${SRC_LEN}/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls transformers:${MODEL_CLS}ForSequenceClassification \
        --input_seq_len $SRC_LEN \
        --target_seq_len $TGT_LEN \
        --batch_size 16 --gradient_accumulation_steps 1 \
        --iters 2500 \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps 250 \
        --data_n_workers 2 \
        --log_interval 100 --valid_interval 100 \
        --optimize_metric $METRIC --optimize_mode max \
        --seed $(($N+42)) \
        --clip_grad_value 5.0
done
done
done
done
echo "run_bert_pretraining.py done"
echo "done"
