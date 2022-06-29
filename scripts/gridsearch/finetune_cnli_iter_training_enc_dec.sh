#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_NAME=t5-base
MODEL_CLS=modeling_rmt_enc_dec:RMTEncoderDecoderForConditionalGeneration
BACKBONE_CLS=transformers:T5ForConditionalGeneration
MODEL_TYPE=encoder-decoder
TASK_NAME=contract_nli
METRIC=exact_match

INPUT_SEQ_LENS=(1497 998 972 1377)
MEMORY_SIZES=(10 10 25 50)

SCHEDULERS=(constant_with_warmup linear)

for N in 1 2
do

for (( j=0; j<${#MEMORY_SIZES[@]}; j++ ))
do
MEMORY_SIZE=${MEMORY_SIZES[j]}
INPUT_SEQ_LEN=${INPUT_SEQ_LENS[j]}

# for (( i=1; i<${#MODEL_NAMES[@]}; i++ ))

LR=1e-03

for (( s=0; s<2; s++ ))
do
SCHEDULER=${SCHEDULERS[s]}

echo N, MODEL_NAME MODEL_ATTR, MEMORY_SIZE, INPUT_SEQ_LEN
echo $N, $MODEL_NAME $MODEL_ATTR, $MEMORY_SIZE, $INPUT_SEQ_LEN

horovodrun --gloo -np $NP python run_finetuning_scrolls_rmt.py \
        --task_name $TASK_NAME \
        --model_path ../runs/finetune/gridsearch/$TASK_NAME/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${INPUT_SEQ_LEN}_mem${MEMORY_SIZE}/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls $MODEL_CLS \
        --backbone_cls $BACKBONE_CLS \
        --input_seq_len $INPUT_SEQ_LEN \
        --input_size 512 \
        --target_seq_len 2 \
        --use_generate_on_valid \
        --num_mem_tokens $MEMORY_SIZE \
        --bptt_depth -1 \
        --backbone_trainable \
        --batch_size 1 --gradient_accumulation_steps 8 \
        --iters 3000 \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr $LR --lr_scheduler $SCHEDULER --num_warmup_steps 100 \
        --data_n_workers 2 \
        --log_interval 50 --valid_interval 250 \
        --optimize_metric $METRIC --optimize_mode max \
        --seed $(($N+42)) \
        --clip_grad_value 5.0
done
done
done
done
echo "run_bert_pretraining.py done"
echo "done"