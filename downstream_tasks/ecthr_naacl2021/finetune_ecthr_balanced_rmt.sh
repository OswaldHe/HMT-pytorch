#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./finetun_hyp.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_NAMES=(roberta-base)
MODEL_TYPE=encoder
MODEL_CLSS=(Roberta)

TBS=32 # total batch size
BS=8 # batch size per gpu, * grad_acc_steps, set to 4 for len 7984
WD=1e-03

METRIC=accuracy
ITERS=3200

PATIENCE=15

DATA_PATH=/home/jovyan/data/ecthr_naacl2021/dataset

# RMT
INPUT_SIZE=512  # segment length
MAX_N_SEGMENTS=16
MEMORY_SIZE=10

for (( i=0; i<${#MODEL_NAMES[@]}; i++ ))
do
MODEL_NAME=${MODEL_NAMES[i]}
MODEL_CLS=${MODEL_CLSS[i]}
for SRC_LEN in 499 998 1996 3992 7984
do
for LR in 5e-05 2e-05 1e-05
do
for SCHEDULER in linear constant_with_warmup
do
for N in 1 2 3
do
rmt_params=rmt_seglen_${INPUT_SIZE}_msz_${MEMORY_SIZE}_sum_loss
MODEL_PATH=./runs/finetune/ecthr_bc_balanced/$MODEL_NAME/${rmt_params}_lr${LR}_${SCHEDULER}_adamw_wd${WD}_bs${TBS}_p${PATIENCE}_len${SRC_LEN}_it${ITERS}/run_$N
echo $MODEL_PATH
horovodrun --gloo -np $NP python -m downstream_tasks.ecthr_naacl2021.run_finetuning_ecthr_rmt \
        --data_path ${DATA_PATH}/train.jsonl \
        --valid_data_path ${DATA_PATH}/dev.jsonl \
        --test_data_path ${DATA_PATH}/test.jsonl \
        --subsample balanced \
        --model_path $MODEL_PATH \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --backbone_cls transformers:${MODEL_CLS}ForSequenceClassification \
        --model_cls modeling_rmt:RMTEncoderForSequenceClassification \
        --input_seq_len $SRC_LEN --data_n_workers 2 \
        --input_size $INPUT_SIZE \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS \
        --backbone_trainable \
        --bptt_depth -1 \
        --sum_loss \
        --iters $ITERS \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --optimizer AdamW  --weight_decay $WD \
        --lr $LR --lr_scheduler $SCHEDULER --num_warmup_steps 150 \
        --log_interval 50 --valid_interval 50 --early_stopping_patience $PATIENCE \
        --optimize_metric $METRIC --optimize_mode max --save_best \
        --seed $(($N+42))
find $MODEL_PATH | grep .pth | xargs -l rm -rf
done
done
done
done
done
echo "run_finetuning_hyp.py done"
echo "done"
