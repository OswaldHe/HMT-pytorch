#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

TASK_NAME=hyperpartisan_news_detection
MODEL_NAME=facebook/bart-base
MODEL_ATTR=model
MODEL_TYPE=encoder-decoder
MODEL_CLS=modeling_rmt_enc_dec:RMTEncoderDecoderForConditionalGeneration
BACKBONE_CLS=transformers:BartForConditionalGeneration

# INPUT_SEQ_LENS=(1002 1503 2004)
# MEMORY_SIZES=(10 10 10)

INPUT_SEQ_LENS=(512)
MEMORY_SIZES=(0)

SCHEDULERS=(linear constant_with_warmup)

for N in 1 2
do

for (( j=0; j<${#MEMORY_SIZES[@]}; j++ ))
do
MEMORY_SIZE=${MEMORY_SIZES[j]}
INPUT_SEQ_LEN=${INPUT_SEQ_LENS[j]}

for LR in 1e-05 5e-05 1e-04
do

for (( s=0; s<2; s++ ))
do
SCHEDULER=${SCHEDULERS[s]}

echo N, MODEL_NAME MODEL_ATTR, MEMORY_SIZE, INPUT_SEQ_LEN
echo $N, $MODEL_NAME $MODEL_ATTR, $MEMORY_SIZE, $INPUT_SEQ_LEN

horovodrun --gloo -np $NP python run_finetuning_hyp_rmt.py \
        --data_path /home/kuratov/data/hyperpartisan_news_detection/train.jsonl \
        --valid_data_path /home/kuratov/data/hyperpartisan_news_detection/dev.jsonl \
        --test_data_path /home/kuratov/data/hyperpartisan_news_detection/test.jsonl \
        --model_path ../runs/debug/$TASK_NAME/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${INPUT_SEQ_LEN}_mem${MEMORY_SIZE}/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls $MODEL_CLS \
        --model_attr ${MODEL_ATTR} \
        --backbone_cls $BACKBONE_CLS \
        --use_generate_on_valid \
        --input_seq_len $INPUT_SEQ_LEN \
        --input_size 512 \
        --num_mem_tokens $MEMORY_SIZE \
        --bptt_depth -1 \
        --backbone_trainable \
        --save_best \
        --batch_size 2 --gradient_accumulation_steps 4 \
        --iters 1000 \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps 100 \
        --data_n_workers 2 \
        --log_interval 50 --valid_interval 50 \
        --optimize_metric f1 --optimize_mode max \
        --seed $(($N+42)) \
        --clip_grad_value 5.0

done
done
done
done
echo "run_bert_pretraining.py done"
echo "done"