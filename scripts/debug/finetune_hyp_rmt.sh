#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

TASK_NAME=hyperpartisan_news_detection

MODEL_TYPE=encoder
MODEL_NAME=microsoft/deberta-v3-base

INPUT_SIZES=(1016 1014 1010 1002 986 954 890)
MEMORY_SIZES=(1 2 4 8 16 32 64 )

LR=1e-05
SCHEDULER=linear
MODEL_CLS=modeling_rmt:RMTEncoderForSequenceClassification

for N in 1 2
do

for (( j=0; j<${#MEMORY_SIZES[@]}; j++ ))
do
MEMORY_SIZE=${MEMORY_SIZES[j]}
INPUT_SIZE=${INPUT_SIZES[j]}


echo N, MODEL_NAME, MEMORY_SIZE, INPUT_SIZE,  MODEL_TYPE,  MODEL_CLS
echo $N, $MODEL_NAME, $MEMORY_SIZE, $INPUT_SIZE,  $MODEL_TYPE,  $MODEL_CLS

horovodrun --gloo -np $NP python run_finetuning_hyp_rmt.py \
        --data_path /home/kuratov/data/hyperpartisan_news_detection/train.jsonl \
        --valid_data_path /home/kuratov/data/hyperpartisan_news_detection/dev.jsonl \
        --test_data_path /home/kuratov/data/hyperpartisan_news_detection/test.jsonl \
        --model_path ../runs/finetune/debug/${TASK_NAME}/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${INPUT_SEQ_LEN}-${TGT_LEN}_mem${MEMORY_SIZE}_bs32_iters${ITERS}_sum_loss/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls $MODEL_CLS \
        --input_seq_len $INPUT_SIZE \
        --input_size 512 \
        --target_seq_len 2 \
        --num_mem_tokens $MEMORY_SIZE \
        --segment_ordering regular \
        --bptt_depth -1 \
        --backbone_trainable \
        --sum_loss \
        --batch_size 1 --gradient_accumulation_steps 8 \
        --iters 3000 \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr $LR --lr_scheduler $SCHEDULER --num_warmup_steps 100 \
        --data_n_workers 2 \
        --log_interval 50 --valid_interval 50 \
        --optimize_metric f1 --optimize_mode max \
        --seed $(($N+42)) \
        --clip_grad_value 5.0
done
done
done
done
done
echo "run_bert_pretraining.py done"
echo "done"