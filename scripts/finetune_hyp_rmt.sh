#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_NAME=rmbert-512-cased
MODEL_TYPE=encoder
MODEL_CLS=modeling_rmt:RMTEncoderForSequenceClassification

SCHEDULER=linear
for N in 1 2 3
do
for MEMORY_ARGS in 924,50 824,100 924,0 824,0 512,0
do
IFS=","; set -- $MEMORY_ARGS;
horovodrun --gloo -np $NP python run_finetuning_hyp_rmt.py \
        --data_path /home/kuratov/data/hyperpartisan_news_detection/train.jsonl \
        --valid_data_path /home/kuratov/data/hyperpartisan_news_detection/dev.jsonl \
        --test_data_path /home/kuratov/data/hyperpartisan_news_detection/test.jsonl \
        --model_path ../runs/finetune/hyperpartisan_news_detection/$MODEL_NAME/lr1e-05_${SCHEDULER}_adamw_wd1e-03/run_$N \
        --from_pretrained bert-base-cased \
        --model_type $MODEL_TYPE \
        --model_cls $MODEL_CLS \
        --input_seq_len $1 \
        --target_seq_len 2 \
        --num_mem_tokens $2 \
        --bptt_depth -1 \
        --backbone_trainable \
        --batch_size 2 --gradient_accumulation_steps 8 \
        --iters 30 \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr 1e-05 --lr_scheduler $SCHEDULER --num_warmup_steps 100 \
        --data_n_workers 2 \
        --log_interval 50 --valid_interval 50 \
        --optimize_metric f1 --optimize_mode max \
        --seed $(($N+42)) \
        --clip_grad_value 5.0
done
done
echo "run_bert_pretraining.py done"
echo "done"
