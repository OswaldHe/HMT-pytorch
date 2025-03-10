# HMT: Hierarchical Memory Transformer

![hmt](/img/hmt_flow_v2.png)

Hierarchical Memory Transformer (HMT) is a novel framework that enables and improves models' long-context processing ability by imitating human memorization behavior. Leveraging memory-augmented segment-level recurrence, we organize the memory hierarchy by preserving tokens from early input tokens segments, passing memory embeddings along the sequence, and recalling relevant information from history.

## Features

- **Easy to use**: If you pretrained a new LLM and want to augment with HMT, simply push the model checkpoints to huggingface and use the `--model_name` argument to pull your model. 
- **Command line centric**: To play with different configurations of HMT during training, simply modify the argument in the command line. There is no need to modify the source code.
- **Memory efficient**: With small and fixed segment lengths for inputs, HMT can still achieve comparable or better effectiveness than models inferencing with longer context, which consumes more GPU VRAM.
- **Long context with topic switching**: HMT is equipped with a memory recall mechanism, which can handle multiple topics in a single long document to filter distractive information.

## Updates
**[2025/01/22]**: The paper is accepted by NAACL 2025 Main Conference.

**[2024/05/12]**: The paper is available on [Arxiv](https://arxiv.org/abs/2405.06067)

## Code Structure

- `hmt_src/main.py`: Entrypoint of HMT command line.
- `hmt_src/pubmedqa_ds_preprocess.py`: Script used to preprocess PubMedQA dataset.
- `modeling_rmt/language_modeling.py`: The HMT model source code.
- `modeling_rmt/long_mem_cross_attn.py`: The memory recall module.


## Instructions
The code adapts the recurrent memory transformer repository (https://github.com/booydar/recurrent-memory-transformer). To use HMT, please install the dependencies in `requirement.txt`.

```
conda create -n hmt python=3.10 -y
conda activate hmt
pip install -r requirement.txt
```

*For AMD GPUs:* Please install the ROCm version of pytorch
```
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/rocm5.6
```


1. Run `accelerate config` and set up the distributed training/evaluation environment based on your cluster configuration. If you use AMD MI210 GPUs, here is an example configuration with DeepSpeed for 4 GPU cluster:
```
- `Accelerate` version: 0.25.0
- Platform: Linux-5.15.0-91-generic-x86_64-with-glibc2.35
- Python version: 3.10.13
- Numpy version: 1.24.4
- PyTorch version (GPU?): 2.1.0+rocm5.6 (True)
- PyTorch XPU available: False
- PyTorch NPU available: False
- System RAM: 503.71 GB
- GPU type: AMD Instinct MI210
- `Accelerate` default config:
        - compute_environment: LOCAL_MACHINE
        - distributed_type: DEEPSPEED
        - mixed_precision: bf16
        - use_cpu: False
        - debug: False
        - num_processes: 4
        - machine_rank: 0
        - num_machines: 1
        - rdzv_backend: static
        - same_network: True
        - main_training_function: main
        - deepspeed_config: {'gradient_accumulation_steps': 1, 'gradient_clipping': 1.0, 'offload_optimizer_device': 'none', 'offload_param_device': 'none', 'zero3_init_flag': False, 'zero_stage': 2}
        - downcast_bf16: no
        - tpu_use_cluster: False
        - tpu_use_sudo: False
        - tpu_env: []
```
2. Run the following command to try PG-19 dataset with Llama 2 7B as the backbone:
```
accelerate launch hmt_src/main.py --task_name=emozilla/pg19 --use_lora --learning_rate=1e-4 --model_name=meta-llama/Llama-2-7b-hf --lr_decay --lr_decay_gamma=0.6 --training_step=600 --num_sensory=32 --bptt_depth=6 --train_set_split=50 --num_seg_save=8 --batch_size=2 --test_length=30000
```
You can use `--inference_only` flag to only perform inferencing for GPU profiling.

For more functionalities of HMT, please check `accelerate launch hmt_src/main.py --help`.

# Examples on Training and Evaluating
The [scripts](./scripts/README.md) folder contains the some examples to to train and evaluate the models on pretrained dataset and downstream tasks. 

## Author's Note

This work is developed and experimented on AMD GPUs, which is a relatively new attempt considering that a majority of deep learning/natural language processing works use NVIDIA GPUs. It is possible that we can utilize tools that currently do not support ROCm (e.g., FlashAttention, bitsandbytes, etc.) to further optimize memory utilization, inference speed, and effectiveness. If you have any new ideas in improving the HMT framework on AMD/NVIDIA GPUs or new evaluation results using these tools, please don't hesitate creating a Github issue and we will look into it.

If you encounter bugs or have any questions, you can send an email to Zifan (`zifanhe1202@g.ucla.edu`) or submit a Github issue.
