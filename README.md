# HMT: Hierarchical Memory Transformer

![hmt](/img/hmt_flow_v2.png)

## Instructions
The code adapts the recurrent memory transformer repository (https://github.com/booydar/recurrent-memory-transformer). Before using the code, pull this repository and replace or add the files. Then install the dependencies in `requirement.txt`.

1. Run `accelerate config` based on your cluster configuration. If you use AMD MI210 GPUs, here is an example configuration with DeepSpeed for 4 GPU cluster:
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

## Reference
```
@inproceedings{
        bulatov2022recurrent,
        title={Recurrent Memory Transformer},
        author={Aydar Bulatov and Yuri Kuratov and Mikhail Burtsev},
        booktitle={Advances in Neural Information Processing Systems},
        editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
        year={2022},
        url={https://openreview.net/forum?id=Uynr3iPhksa}
}

@misc{bulatov2023scaling,
      title={Scaling Transformer to 1M tokens and beyond with RMT}, 
      author={Aydar Bulatov and Yuri Kuratov and Mikhail S. Burtsev},
      year={2023},
      eprint={2304.11062},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```




