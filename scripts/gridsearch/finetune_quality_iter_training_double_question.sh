#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_NAME=t5-base
MODEL_TYPE=encoder-decoder
MODEL_CLS=modeling_rmt_enc_dec_double_question:RMTEncoderDecoderForConditionalGeneration
BACKBONE_CLS=transformers:T5ForConditionalGeneration
TASK_NAME=quality
# PREFIX="summarize: "

# SCROLLS TASKS (train/valid/test)
# gov_report (17457/972/973) 10ep ~5500 iters with bs 32
# summ_screen_fd (3673/338/337) 10ep ~1150 iters with bs 32
# qmsum (1257/272/281) 20ep ~800 iters with bs 32
# narrative_qa (55003/5878/10306) 2ep ~3500 iters with bs 32
# qasper (2567/1726/1399) 20ep ~1605 iters with bs 32
# quality (2523/2086/2128) 20ep ~1600 iters with bs 32
# contract_nli (7191/1037/2091) 20ep ~4500 iters with bs 32

# BART 256 512 1024
# LED 1024 4096 16384
# SRC_LEN=256
# BART 1024
# LED 1024
# 

INPUT_SEQ_LENS=(998 1497 918 1377)
MEMORY_SIZES=(10 10 50 50)

TGT_LEN=1024

METRIC=exact_match
SCHEDULER=linear

for SCHEDULER in linear constant_with_warmup
do

for N in 1 2 
do

for (( i=0; i<${#INPUT_SEQ_LENS[@]}; i++ ))
do
MEMORY_SIZE=${MEMORY_SIZES[j]}
INPUT_SEQ_LEN=${INPUT_SEQ_LENS[j]}

for LR in 1e-04 5e-05
do


horovodrun --gloo -np $NP python run_finetuning_scrolls_rmt_debug.py \
        --task_name $TASK_NAME \
        --model_path ../runs/finetune/gridsearch/$TASK_NAME/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${INPUT_SEQ_LEN}-${TGT_LEN}_mem${MEMORY_SIZE}/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls $MODEL_CLS \
        --backbone_cls $BACKBONE_CLS \
        --use_generate_on_valid \
        --backbone_trainable \
        --input_seq_len $INPUT_SEQ_LEN \
        --target_seq_len $TGT_LEN \
        --num_mem_tokens $MEMORY_SIZE \
        --input_size 512 \
        --bptt_depth -1 \
        --batch_size 1 --gradient_accumulation_steps 8 \
        --save_best --iters 2000 \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps 200 \
        --data_n_workers 0 \
        --log_interval 100 --valid_interval 250 \
        --optimize_metric $METRIC --optimize_mode max \
        --seed $(($N+42)) \
        --clip_grad_value 5.0
done
done
done
done
echo "run_bert_pretraining.py done"
echo "done"
