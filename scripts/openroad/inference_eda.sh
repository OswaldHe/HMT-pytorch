export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

accelerate env 

# Uncomment to disable wandb tracking
export WANDB_MODE=offline

accelerate launch hmt_src/main.py \
    --task_name=eda_corpus \
    --use_lora \
    --learning_rate=1e-4 \
    --model_name=meta-llama/Llama-3.1-8B-Instruct \
    --token_file "your_huggingface_token_file" \
    --training_step=1000 \
    --num_sensory=32 \
    --segment_length=512 \
    --bptt_depth=6 \
    --num_seg_save=8 \
    --inference_only \
    --load_from_ckpt "llama3.1-8b-instruct-eda-base" \
    --batch_size=1 \
    --shuffle \
    --eval_step=25 \
    --test_length=2048 \
    --wandb_run=llama3-8b-eda \
    --wandb_project=openroad_hmt \
    --wandb_entity=your_wandb_entity \