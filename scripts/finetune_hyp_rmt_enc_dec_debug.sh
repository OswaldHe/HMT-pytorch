#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

TASK_NAME=hyperpartisan_news_detection

MODEL_NAMES=(facebook/bart-base t5-base)
MODEL_ATTRS=(model)
MODEL_TYPE=encoder-decoder
MODEL_CLS=modeling_rmt_enc_dec_debug:RMTEncoderDecoderForConditionalGeneration
BACKBONE_CLSS=(transformers:BartForConditionalGeneration)
LR=1e-04
SCHEDULER=linear

INPUT_SIZES=(1002 1503 2004 1644)
MEMORY_SIZES=(10 10 10 100)

for N in 1 2 3
do

for (( j=0; j<${#MEMORY_SIZES[@]}; j++ ))
do
MEMORY_SIZE=${MEMORY_SIZES[j]}
INPUT_SIZE=${INPUT_SIZES[j]}

# for (( i=0; i<${#MODEL_NAMES[@]}; i++ ))
# do
MODEL_NAME=${MODEL_NAMES[i]}
MODEL_ATTR=${MODEL_ATTRS[i]}

echo N, MODEL_NAME, MEMORY_SIZE, MAX_INPUT_SIZE
echo $N, $MODEL_NAME, $MEMORY_SIZE, $MAX_INPUT_SIZE

horovodrun --gloo -np $NP python run_finetuning_hyp_rmt_debug.py \
        --data_path /home/kuratov/data/hyperpartisan_news_detection/train.jsonl \
        --valid_data_path /home/kuratov/data/hyperpartisan_news_detection/dev.jsonl \
        --test_data_path /home/kuratov/data/hyperpartisan_news_detection/test.jsonl \
        --model_path ../runs/finetune/$TASK_NAME/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${INPUT_SIZE}_mem${MEMORY_SIZE}/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls $MODEL_CLS \
        --model_attr $MODEL_ATTR \
        --backbone_cls transformers:BartForConditionalGeneration \
        --use_generate_on_valid \
        --input_seq_len $INPUT_SIZE \
        --input_size 512 \
        --target_seq_len 2 \
        --num_mem_tokens $MEMORY_SIZE \
        --bptt_depth -1 \
        --backbone_trainable \
        --batch_size 1 --gradient_accumulation_steps 16 \
        --iters 1000 \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps 100 \
        --data_n_workers 2 \
        --log_interval 50 --valid_interval 50 \
        --optimize_metric f1 --optimize_mode max \
        --seed $(($N+42)) \
        --clip_grad_value 5.0

# done
done
done
echo "run_bert_pretraining.py done"
echo "done"
