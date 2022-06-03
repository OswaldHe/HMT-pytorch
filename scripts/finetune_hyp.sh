#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_NAME=t5-base
MODEL_TYPE=encoder-decoder
MODEL_CLS=transformers:T5ForConditionalGeneration

for SCHEDULER in linear constant_with_warmup
do
for N in 1 2 3 4 5
do
horovodrun --gloo -np $NP python run_finetuning_hyp.py \
        --data_path /home/kuratov/data/hyperpartisan_news_detection/train.jsonl \
        --valid_data_path /home/kuratov/data/hyperpartisan_news_detection/dev.jsonl \
        --test_data_path /home/kuratov/data/hyperpartisan_news_detection/test.jsonl \
        --model_path ./runs/finetune/$MODEL_NAME/hyperpartisan_news_detection/lr1e-04_${SCHEDULER}_adamw_wd1e-03/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls $MODEL_CLS \
        --use_generate_on_valid \
        --input_seq_len 512 \
        --target_seq_len 2 \
        --batch_size 4 --gradient_accumulation_steps 2 \
        --save_best --iters 1000 \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr 1e-04 --lr_scheduler $SCHEDULER --num_warmup_steps 100 \
        --data_n_workers 2 \
        --log_interval 50 --valid_interval 50 \
        --optimize_metric f1 --optimize_mode max \
        --seed $(($N+42)) \
        --clip_grad_value 5.0
done
done
echo "run_bert_pretraining.py done"
echo "done"
