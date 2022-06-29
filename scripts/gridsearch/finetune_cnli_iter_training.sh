#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_NAMES=(roberta-base bert-base-cased microsoft/deberta-v3-base)
MODEL_ATTRS=(roberta bert deberta)
MODEL_CLS=modeling_rmt:RMTEncoderForSequenceClassification
MODEL_TYPE=encoder
TASK_NAME=contract_nli
METRIC=exact_match

INPUT_SEQ_LENS=(998 968 1452 1497 1836)
MEMORY_SIZES=(10 25 25 10 50)


for N in 3
do

for (( j=0; j<${#MEMORY_SIZES[@]}; j++ ))
do
MEMORY_SIZE=${MEMORY_SIZES[j]}
INPUT_SEQ_LEN=${INPUT_SEQ_LENS[j]}

for (( n=0; n<${#MODEL_NAMES[@]}; n++ ))
do
MODEL_NAME=${MODEL_NAMES[n]}
MODEL_ATTR=${MODEL_ATTRS[n]}

for LR in 5e-05 1e-05
do

for SCHEDULER in constant_with_warmup linear
do

echo N, MODEL_NAME MODEL_ATTR, MEMORY_SIZE, INPUT_SEQ_LEN
echo $N, $MODEL_NAME $MODEL_ATTR, $MEMORY_SIZE, $INPUT_SEQ_LEN

horovodrun --gloo -np $NP python run_finetuning_scrolls_rmt.py \
        --task_name $TASK_NAME \
        --model_path ../runs/finetune/gridsearch/$TASK_NAME/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${INPUT_SEQ_LEN}_mem${MEMORY_SIZE}/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls $MODEL_CLS \
        --model_attr $MODEL_ATTR \
        --input_seq_len $INPUT_SEQ_LEN \
        --input_size 512 \
        --target_seq_len 2 \
        --num_mem_tokens $MEMORY_SIZE \
        --bptt_depth -1 \
        --backbone_trainable \
        --batch_size 1 --gradient_accumulation_steps 8 \
        --iters 5000 \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr $LR --lr_scheduler $SCHEDULER --num_warmup_steps 100 \
        --data_n_workers 0 \
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