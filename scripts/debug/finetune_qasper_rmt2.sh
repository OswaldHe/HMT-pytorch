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

#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_NAME=t5-base
MODEL_TYPE=encoder-decoder
MODEL_CLS=modeling_rmt_enc_dec:RMTEncoderDecoderForConditionalGeneration
BACKBONE_CLS=transformers:T5ForConditionalGeneration
TASK_NAME=qasper

ITERS=1600
TBS=32
BS=2

TGT_LEN=1024

INPUT_SEQ_LENS=(1002)
MEMORY_SIZES=(0)

for (( j=0; j<${#MEMORY_SIZES[@]}; j++ ))
do
MEMORY_SIZE=${MEMORY_SIZES[j]}
INPUT_SEQ_LEN=${INPUT_SEQ_LENS[j]} 

METRIC=f1
SCHEDULER=linear

for LR in 2e-04
do

for N in 2
do

echo RUNNING: TASK_NAME SRC_LEN MODEL_NAME MEMORY_SIZE INPUT_SEQ_LEN LR N
echo RUNNING: $TASK_NAME $SRC_LEN $MODEL_NAME $MEMORY_SIZE $INPUT_SEQ_LEN $LR $N
horovodrun --gloo -np $NP python run_finetuning_scrolls_rmt.py \
        --task_name $TASK_NAME \
        --model_path ../runs/finetune/debug/${TASK_NAME}/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${INPUT_SEQ_LEN}-${TGT_LEN}_mem${MEMORY_SIZE}_bs${TBS}_iters${ITERS}_sl/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls $MODEL_CLS \
        --backbone_cls $BACKBONE_CLS \
        --use_generate_on_valid \
        --input_seq_len $INPUT_SEQ_LEN \
        --input_size 512 \
        --target_seq_len $TGT_LEN \
        --num_mem_tokens $MEMORY_SIZE \
        --sum_loss \
        --segment_ordering regular \
        --bptt_depth -1 \
        --backbone_trainable \
        --save_best \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --iters $ITERS \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --log_interval $(($ITERS/20)) --valid_interval $(($ITERS/20)) \
        --optimize_metric $METRIC --optimize_mode max \
        --show_valid_examples 5 \
        --early_stopping_patience 15 \
        --seed $(($N+42))
done
done
done
done
echo "run_bert_pretraining.py done"
echo "done"