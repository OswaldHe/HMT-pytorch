#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_NAMES=(bert-base-cased)
MODEL_TYPE=encoder
MODEL_ATTRS=(bert)

MEMORY_SIZES=(0 100)
INPUT_SIZES=(512 818)

for (( j=0; j<${#MEMORY_SIZES[@]}; j++ ))
do
MEMORY_SIZE=${MEMORY_SIZES[j]}
INPUT_SIZE=${INPUT_SIZES[j]}
for LR in 1e-04 5e-05
do
for SCHEDULER in linear constant_with_warmup
do
for N in 1 2 3
do
for (( i=0; i<${#MODEL_NAMES[@]}; i++ ))
do
MODEL_NAME=${MODEL_NAMES[i]}
MODEL_CLS=modeling_rmt:RMTEncoderForSequenceClassification
MODEL_ATTR=${MODEL_ATTRS[i]}
horovodrun --gloo -np $NP python run_finetuning_hyp_rmt.py \
        --data_path /home/kuratov/data/hyperpartisan_news_detection/train.jsonl \
        --valid_data_path /home/kuratov/data/hyperpartisan_news_detection/dev.jsonl \
        --test_data_path /home/kuratov/data/hyperpartisan_news_detection/test.jsonl \
        --model_path ../runs/finetune/hyperpartisan_news_detection/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${INPUT_SIZE}_mem${MEMORY_SIZE}/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls $MODEL_CLS \
        --model_attr $MODEL_ATTR \
        --input_seq_len $INPUT_SIZE \
        --target_seq_len 2 \
        --num_mem_tokens $MEMORY_SIZE \
        --bptt_depth -1 \
        --backbone_trainable \
        --batch_size 2 --gradient_accumulation_steps 8 \
        --iters 3000 \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps 100 \
        --data_n_workers 2 \
        --log_interval 50 --valid_interval 50 \
        --optimize_metric f1 --optimize_mode max \
        --seed $(($N+42)) \
        --clip_grad_value 5.0
# RM model path/best_model
done
done
done
done
done
echo "run_bert_pretraining.py done"
echo "done"