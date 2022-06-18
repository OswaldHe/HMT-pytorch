#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

# BART 256 512 1024
# LED 1024 4096 16384
# SRC_LEN=512
# BART 1024
# LED 1024
# 
# TGT_LEN=1024

MODEL_NAMES=(bert-base-cased google/electra-base-discriminator microsoft/deberta-v3-base roberta-base)
MODEL_TYPE=encoder
MODEL_ATTRS=(bert electra deberta roberta)
TASK_NAME=contract_nli
METRIC=exact_match

LR=1e-05
SCHEDULER=linear
MODEL_CLS=modeling_rmt:RMTEncoderForSequenceClassification

MAX_INPUT_SIZES=(1636 1497 1996)
MEMORY_SIZES=(100 10 10)
INPUT_SIZE=512

for N in 2 3 4
do

for (( j=0; j<${#MEMORY_SIZES[@]}; j++ ))
do
MEMORY_SIZE=${MEMORY_SIZES[j]}
MAX_INPUT_SIZE=${MAX_INPUT_SIZES[j]}

for (( i=0; i<${#MODEL_NAMES[@]}; i++ ))
do
MODEL_NAME=${MODEL_NAMES[i]}
MODEL_ATTR=${MODEL_ATTRS[i]}

echo N, MODEL_NAME MODEL_ATTR, MEMORY_SIZE, MAX_INPUT_SIZE
echo $N, $MODEL_NAME $MODEL_ATTR, $MEMORY_SIZE, $MAX_INPUT_SIZE

horovodrun --gloo -np $NP python run_finetuning_scrolls_rmt.py \
        --task_name $TASK_NAME \
        --model_path ../runs/finetune/${TASK_NAME}/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${MAX_INPUT_SIZE}_mem${MEMORY_SIZE}/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls $MODEL_CLS \
        --model_attr $MODEL_ATTR \
        --input_seq_len $MAX_INPUT_SIZE \
        --input_size $INPUT_SIZE \
        --target_seq_len 2 \
        --num_mem_tokens $MEMORY_SIZE \
        --bptt_depth -1 \
        --backbone_trainable \
        --batch_size 1 --gradient_accumulation_steps 8 \
        --save_best --iters 3000 \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps 100 \
        --data_n_workers 2 \
        --log_interval 50 --valid_interval 50 \
        --optimize_metric $METRIC --optimize_mode max \
        --seed $(($N+42)) \
        --clip_grad_value 5.0
done
done
done
done
done
echo "run_bert_pretraining.py done"
echo "done"