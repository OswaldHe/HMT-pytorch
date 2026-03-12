from copy import deepcopy

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from .language_modeling import Summary_Memory_RecurrentWrapper, Memory_Only_RecurrentWrapper
from .memory_cell import MemoryCell


def generate_model(args, base_model, logger, tokenizer=None):
    """Configure the recurrent model wrapper and return it with sequence parameters."""
    if args.baseline_only:
        logger.warning(
            "training and evaluating only the backbone. remember to align the segment rightward"
        )
        num_mem_embed = 0
        block_size = args.segment_length
        history_size = block_size
        mask_size = block_size
    else:
        num_mem_embed = args.mem_recall_size
        n_segments = args.bptt_depth
        block_size = args.segment_length
        # block_size -= 2 * num_mem_embed
        # block_size -= args.num_sensory
        history_size = n_segments * block_size
        mask_size = block_size

    logger.info("Preparing recurrent model wrapper...")
    if getattr(args, "recurrent_type") == "summary_memory":
        logger.info("Using Summary-Memory Recurrent Wrapper")
        wrapper_cls = Summary_Memory_RecurrentWrapper
    elif getattr(args, "recurrent_type") == "memory_only":
        logger.info("Using Memory-Only Recurrent Wrapper")
        wrapper_cls = Memory_Only_RecurrentWrapper
    else:
        raise ValueError(f"Unknown recurrent_type: {getattr(args, 'recurrent_type')}")

    if args.rmt_only or args.baseline_only:
        model = wrapper_cls(
            base_model,
            num_mem_embed=num_mem_embed,
            num_prepend=0,
            segment_size=block_size,
            mask_size=mask_size,
            n_cell_out=args.num_seg_save,
            rmt_only=args.rmt_only,
            baseline_only=args.baseline_only,
            dynamic_seg=getattr(args, "dynamic_seg", False),
            dynamic_seg_checkpoint=getattr(args, "dynamic_seg_checkpoint", None),
            lm_tokenizer=tokenizer,
        )
    else:
        model = wrapper_cls(
            base_model,
            num_mem_embed=num_mem_embed,
            num_prepend=args.num_sensory,
            mem_hidden_dim=args.mem_hidden_dim,
            mem_window_size=args.mem_window_size,
            segment_size=block_size,
            mask_size=mask_size,
            n_cell_out=args.num_seg_save,
            mem_mlp=args.mem_mlp,
            mem_mlp_hidden_dim=args.mem_mlp_hidden_dim,
            dynamic_seg=getattr(args, "dynamic_seg", False),
            dynamic_seg_checkpoint=getattr(args, "dynamic_seg_checkpoint", None),
            lm_tokenizer=tokenizer,
        )

    if args.load_from_ckpt is not None:
        state_dict = get_fp32_state_dict_from_zero_checkpoint(args.load_from_ckpt)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded model weights from {args.load_from_ckpt}")

    return model, block_size, history_size
