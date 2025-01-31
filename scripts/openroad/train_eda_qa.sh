export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

accelerate env 

# Uncomment to disable wandb tracking
# export WANDB_MODE=offline

accelerate launch hmt_src/main.py \
    --task_name=eda_qa \
    --use_lora \
    --learning_rate=1e-4 \
    --model_name=meta-llama/Llama-3.1-8B-Instruct \
    --token_file "your_huggingface_token" \
    --training_step=2000 \
    --num_sensory=32 \
    --segment_length=512 \
    --bptt_depth=12 \
    --num_seg_save=13 \
    --load_from_ckpt "llama3.1-8b-instruct-eda-base" \
    --save_ckpt "llama3.1-8b-instruct-eda-qa-finetune" \
    --batch_size=1 \
    --num_epochs=2 \
    --eval_step=10 \
    --test_step=10 \
    --shuffle \
    --wandb_run=llama3-8b-openroad-qa \
    --wandb_project=openroad_hmt \
    --wandb_entity=your_wandb_entity \