export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

accelerate env 

# Uncomment to disable wandb tracking
export WANDB_MODE=offline

accelerate launch hmt_src/main.py \
    --task_name=emozilla/pg19 \
    --use_lora \
    --learning_rate=1e-4 \
    --model_name=Qwen/Qwen2.5-7B \
    --lr_decay \
    --lr_decay_gamma=0.7 \
    --training_step=600 \
    --num_sensory=32 \
    --bptt_depth=6 \
    --train_set_split=50 \
    --num_seg_save=8 \
    --batch_size=2 \
    --test_length=30000 \
    --wandb_run=pg19_qwen \
    --wandb_project=large_model_hmt \
    --wandb_entity=zifanhe1202-ucla \