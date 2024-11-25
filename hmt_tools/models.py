from modeling_rmt.language_modeling import RecurrentWrapper, MemoryCell
import torch
import copy

def load_model(args, model, memory_size, block_size, n_segments, mask_size, word_emb_dim, 

               is_qa_task=False, max_level_segs=None, cpu=False):
    """Load an RMT Wrapped model based on the arguments. 

    :param args: The argument object taken from the command line
    :type args: argparse.Namespace
    :param memory_size: _description_
    :type memory_size: _type_
    :param block_size: _description_
    :type block_size: _type_
    :param n_segments: _description_
    :type n_segments: _type_
    :param mask_size: _description_
    :type mask_size: _type_
    :param word_emb_dim: _description_
    :type word_emb_dim: _type_
    :param is_qa_task: Whether the task to perform is an QA Task, defaults to False
    :type is_qa_task: bool, optional
    :return: The Model
    :rtype: _type_
    """

    if args.rmt_only or args.baseline_only:
        cell = MemoryCell(model, 
                    num_mem_tokens=memory_size,
                    num_prepend=0)

        model = RecurrentWrapper(cell,
                                segment_size=block_size,
                                max_n_segments=n_segments,
                                mask_size=mask_size,
                                n_cell_out=args.num_seg_save,
                                segment_alignment=args.segment_alignment
                                )
    else:
        print("Initializing Memory Cell")
        cell = MemoryCell(model, 
                        num_mem_tokens=memory_size,
                        num_prepend=args.num_sensory)

        if args.hmt_stage_1:
            model = RecurrentWrapper(cell,
                                segment_size=block_size,
                                max_n_segments=n_segments,
                                mask_size=mask_size,
                                n_cell_out=args.num_seg_save,
                                segment_alignment=args.segment_alignment
                                )
        else:
            if args.load_from_ckpt is not None and args.hmt_stage_2:
                ori_model = RecurrentWrapper(cell,
                                segment_size=block_size,
                                max_n_segments=n_segments,
                                mask_size=mask_size,
                                n_cell_out=args.num_seg_save,
                                segment_alignment=args.segment_alignment)
                
                # state_dict = get_fp32_state_dict_from_zero_checkpoint(args.load_from_ckpt)
                # ori_model.load_state_dict(state_dict)
                state_dict = torch.load(args.load_from_ckpt,map_location='cpu' if cpu else 'cuda:0')
                # Remove the 'module.' prefix from all state dict keys
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                state_dict = new_state_dict
                ori_model.load_state_dict(state_dict)
                cell = copy.deepcopy(ori_model.memory_cell)

            print("Initializing Recurrent Wrapper")
            model = RecurrentWrapper(cell,
                                emb=copy.deepcopy(model.get_input_embeddings()),
                                word_emb_dim=word_emb_dim,
                                hidden_dim=args.mem_recall_hidden_dim,
                                ltm_context=args.mem_recall_context,
                                segment_size=block_size,
                                max_n_segments=max_level_segs if max_level_segs is not None else n_segments,
                                mask_size=mask_size,
                                n_cell_out=args.num_seg_save,
                                segment_alignment=args.segment_alignment,
                                is_qa_task=is_qa_task
                                )
            print("Loading Checkpoint")
            if args.load_from_ckpt is not None and not args.hmt_stage_2:
                # checkpoint_dir = os.path.dirname(args.load_from_ckpt)
                state_dict = torch.load(args.load_from_ckpt,map_location='cpu' if cpu else 'cuda:0')
                # Remove the 'module.' prefix from all state dict keys
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  # FIXME: while saving the state dicts while the model is wrapped, all the weights have a model. prefix and needed to be removed when loading. We should fix this by unwrapping before saving. 
                # if args.model_name == 'openlm-research/open_llama_3b_v2':
                #     new_state_dict = {k.replace('base_model.model.', ''): v for k, v in new_state_dict.items()}
                state_dict = new_state_dict
                model.load_state_dict(state_dict)
                # tag = os.path.basename(args.load_from_ckpt)
                # state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, 'opt-350m')
                # model.load_state_dict(state_dict)

    return model


def load_model_yaml(model, memory_size, block_size, n_segments, mask_size, word_emb_dim, 
               is_qa_task=False, max_level_segs=None, cpu=False, **kwargs):
    """Load an RMT Wrapped model based on the arguments. 

    :param args: The argument object taken from the command line
    :type args: argparse.Namespace
    :param memory_size: _description_
    :type memory_size: _type_
    :param block_size: _description_
    :type block_size: _type_
    :param n_segments: _description_
    :type n_segments: _type_
    :param mask_size: _description_
    :type mask_size: _type_
    :param word_emb_dim: _description_
    :type word_emb_dim: _type_
    :param is_qa_task: Whether the task to perform is an QA Task, defaults to False
    :type is_qa_task: bool, optional
    :return: The Model
    :rtype: _type_
    """

    if kwargs['rmt_only'] or kwargs['baseline_only']:
        cell = MemoryCell(model, 
                    num_mem_tokens=memory_size,
                    num_prepend=0)

        model = RecurrentWrapper(cell,
                                segment_size=block_size,
                                max_n_segments=n_segments,
                                mask_size=mask_size,
                                n_cell_out=kwargs['num_seg_save'],
                                segment_alignment=kwargs['segment_alignment']
                                )
    else:
        cell = MemoryCell(model, 
                        num_mem_tokens=memory_size,
                        num_prepend=kwargs['num_sensory'])

        if kwargs['hmt_stage_1']:
            model = RecurrentWrapper(cell,
                                segment_size=block_size,
                                max_n_segments=n_segments,
                                mask_size=mask_size,
                                n_cell_out=kwargs['num_seg_save'],
                                segment_alignment=kwargs['segment_alignment']
                                )
        else:
            if kwargs['load_from_ckpt'] is not None and kwargs['hmt_stage_2']:
                ori_model = RecurrentWrapper(cell,
                                segment_size=block_size,
                                max_n_segments=n_segments,
                                mask_size=mask_size,
                                n_cell_out=kwargs['num_seg_save'],
                                segment_alignment=kwargs['segment_alignment'])
                
                # state_dict = get_fp32_state_dict_from_zero_checkpoint(args.load_from_ckpt)
                # ori_model.load_state_dict(state_dict)
                state_dict = torch.load(kwargs['load_from_ckpt'],map_location='cpu' if cpu else 'cuda:0')
                # Remove the 'module.' prefix from all state dict keys
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                state_dict = new_state_dict
                ori_model.load_state_dict(state_dict)
                cell = copy.deepcopy(ori_model.memory_cell)

            model = RecurrentWrapper(cell,
                                emb=copy.deepcopy(model.get_input_embeddings()),
                                word_emb_dim=word_emb_dim,
                                hidden_dim=kwargs['mem_recall_hidden_dim'],
                                ltm_context=kwargs['mem_recall_context'],
                                segment_size=block_size,
                                max_n_segments=max_level_segs if max_level_segs is not None else n_segments,
                                mask_size=mask_size,
                                n_cell_out=kwargs['num_seg_save'],
                                segment_alignment=kwargs['segment_alignment'],
                                is_qa_task=is_qa_task
                                )
            
            if kwargs['load_from_ckpt'] is not None and not kwargs['hmt_stage_2']:
                # checkpoint_dir = os.path.dirname(args.load_from_ckpt)
                state_dict = torch.load(kwargs['load_from_ckpt'],map_location='cpu' if cpu else 'cuda:0')
                # Remove the 'module.' prefix from all state dict keys
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  # FIXME: while saving the state dicts while the model is wrapped, all the weights have a model. prefix and needed to be removed when loading. We should fix this by unwrapping before saving. 
                # if args.model_name == 'openlm-research/open_llama_3b_v2':
                #     new_state_dict = {k.replace('base_model.model.', ''): v for k, v in new_state_dict.items()}
                state_dict = new_state_dict
                model.load_state_dict(state_dict)
                # tag = os.path.basename(args.load_from_ckpt)
                # state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, 'opt-350m')
                # model.load_state_dict(state_dict)

    return model

