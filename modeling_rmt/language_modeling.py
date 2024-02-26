import math
import torch
import copy
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from modeling_rmt.long_mem_cross_attn import CrossAttentionMemory
from accelerate.logging import get_logger
import random

class MemoryCell(torch.nn.Module):
    def __init__(self, base_model, num_mem_tokens, num_prepend):
        super().__init__()
        self.model = base_model
        self.n_prepend = num_prepend
        self.prepend_list = None
        self.create_memory(num_mem_tokens)

    def create_memory(self, num_mem_tokens):
        self.num_mem_tokens = num_mem_tokens
        embeddings = self.model.get_input_embeddings()
        if num_mem_tokens > 0:
            memory_dim =  getattr(self.model.config, 'n_embd', self.model.config.hidden_size)
            memory_weights = torch.randn((num_mem_tokens, memory_dim)) * embeddings.weight.data.std()
            self.register_parameter('memory', torch.nn.Parameter(memory_weights, requires_grad=True))
            # signifier_weights = torch.randn((1, memory_dim)) * embeddings.weight.data.std()
            # self.register_parameter('signifier', torch.nn.Parameter(signifier_weights, requires_grad=False))

        self.read_memory_position = range(num_mem_tokens)
        self.write_memory_position = range(-num_mem_tokens, 0)

    def set_memory(self, input_shape):
        memory = self.memory.repeat(input_shape[0], 1, 1)
        return memory

    def forward(self, input_ids, memory_state=None, prepend_state=None, browse=False, **kwargs):
        input_ids = input_ids.cuda()
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                kwargs[k] = v.cuda()
        if memory_state is None and self.num_mem_tokens > 0:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(input_ids, memory_state, prepend_state=prepend_state, **kwargs)

        out = self.model(**seg_kwargs)
        n_prepend = self.n_prepend//2 if browse else self.n_prepend
        out, new_memory_state = self.process_output(out, 0 if prepend_state is None else n_prepend, **kwargs)
        input_ids = input_ids.cpu()
        for k, v in kwargs.items():
                if torch.is_tensor(v):
                    kwargs[k] = v.cpu()
        return out, new_memory_state, self.prepend_list
    
    def generate(self, input_ids, memory_state, attention_mask, **generate_kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(input_ids, memory_state, attention_mask=attention_mask)
        out = self.model.generate(inputs_embeds=seg_kwargs['inputs_embeds'], attention_mask=seg_kwargs['attention_mask'], **generate_kwargs)
        return out

    def process_input(self, input_ids, memory_state, prepend_state=None, **kwargs):
        seg_kwargs = dict(**kwargs)

        inputs_embeds = kwargs.get('inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        if prepend_state is not None:
            prepend_embeds = self.model.get_input_embeddings()(prepend_state)
            inputs_embeds = torch.cat([prepend_embeds, inputs_embeds], dim=1)
        if memory_state is not None:
            inputs_embeds = torch.cat([memory_state, inputs_embeds, memory_state], dim=1)
        
        self.prepend_list = input_ids[:,-self.n_prepend:] if self.n_prepend != 0 else None

        seg_kwargs['input_ids'] = None
        seg_kwargs['inputs_embeds'] = inputs_embeds
        if kwargs.get('attention_mask') is not None:
            seg_kwargs['attention_mask'] = self.pad_attention_mask(kwargs['attention_mask'], inputs_embeds.shape, 0 if prepend_state is None else self.n_prepend)
        seg_kwargs['output_hidden_states'] = True
        return seg_kwargs
    
    def pad_attention_mask(self, attention_mask, shape, n_prepend):
        if self.num_mem_tokens in {0, None}:
            mask = torch.ones(*shape[:2], dtype=torch.int64).to(attention_mask.device)
            mask[:, (n_prepend):] = attention_mask
            return mask
        else:
            mask = torch.ones(*shape[:2], dtype=torch.int64).to(attention_mask.device)
            mask[:, (self.num_mem_tokens+n_prepend):-self.num_mem_tokens] = attention_mask
            return mask
    
    def process_output(self, model_outputs, n_prepend, **kwargs):
        if self.num_mem_tokens not in {0, None}:
            out = CausalLMOutputWithCrossAttentions()
            memory_state = model_outputs.hidden_states[-1][:, -self.num_mem_tokens:]
            out['logits'] = model_outputs.logits[:, (self.num_mem_tokens+n_prepend):-self.num_mem_tokens]
            out['logits'] = out['logits'].cpu()

            if kwargs.get('output_hidden_states'):
                out['hidden_states'] = [lh[:, (self.num_mem_tokens+n_prepend):-self.num_mem_tokens] for lh in model_outputs.hidden_states]
            if kwargs.get('output_attentions'):
                out['attentions'] = model_outputs['attentions']
        else:
            out = CausalLMOutputWithCrossAttentions()
            memory_state = None
            out['logits'] = model_outputs.logits[:, (n_prepend):]
            
            if kwargs.get('output_hidden_states'):
                out['hidden_states'] = [lh[:, (n_prepend):] for lh in model_outputs.hidden_states]
            if kwargs.get('output_attentions'):
                out['attentions'] = model_outputs['attentions']
            
        return out, memory_state 

class SegmentIterator:
    def __init__(self, **kwargs):
        self.iter_content = kwargs
        self.pointer = 0
        self.empty = False
    
    def next(self, segment_length):
        segment = {}
        for k, tensor in self.iter_content.items():
            if tensor is not None:
                if self.pointer > tensor.shape[1]:
                    self.empty = True
                    return None
                segment[k] = tensor[:, self.pointer:self.pointer+segment_length]
        
        self.pointer += segment_length
        return segment
    
    def is_empty(self):
        return self.empty

class RecurrentWrapper(torch.nn.Module):
    def __init__(self, memory_cell, emb=None, word_emb_dim=4096, hidden_dim=4096, ltm_context=100, **rmt_kwargs):
        super().__init__()
        self.memory_cell = memory_cell
        self.rmt_config = rmt_kwargs
        self.ltm_context = ltm_context
        self.logger = get_logger('')
        if emb is not None:
            memory_weights = torch.randn((1, word_emb_dim)) * emb.weight.data.std()
            self.register_parameter('mem', torch.nn.Parameter(memory_weights, requires_grad=True))
            self.cross_attn = CrossAttentionMemory(word_emb_dim, hidden_dim)
        else:
            self.cross_attn = None

    def forward(self, input_ids, labels=None, labels_mask=None, inputs_embeds=None, attention_mask=None, output_attentions=None, output_hidden_states=None, segment_size=1022, extra_size=16, mode='train', profile=False):
        memory_state = None
        prepend_state = None
        seg_iter = SegmentIterator(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        cell_outputs = []
        n_cell_out = self.rmt_config.get('n_cell_out')
        memory_seq = None

        total_hist = []

        seg_num = 0
        segment = None

        if profile:
            self.logger.info('start inferencing segments')

        while not seg_iter.is_empty():
            segment = seg_iter.next(segment_size)
            if segment is None:
                break
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            browse = False
            if self.cross_attn is not None:
                s_mem = self.mem.repeat(segment['input_ids'].shape[0], 1, 1)
                seg = copy.deepcopy(segment)
                seg['input_ids'] = seg['input_ids'][:,:(segment_size//2)]
                seg['attention_mask'] = seg['attention_mask'][:,:(segment_size//2)]
                _, q_mem, _ = self.memory_cell(**seg, memory_state=s_mem)
                browse_thres = 0
                if mode == 'test':
                    browse_thres = 2
                memory_state, hist, browse = self.cross_attn(memory_seq, q_mem, mode, seg_num if seg_num < self.ltm_context else self.ltm_context, browse_thres)
                if hist is not None:
                    total_hist.extend(hist)
            
            if browse and mode == 'test':
                # proceed extra tokens
                extra_seg = seg_iter.next(extra_size)
                if extra_seg is not None:
                    for k, tensor in extra_seg.items():
                        segment[k] = torch.cat([segment[k], tensor], dim=1)

            browse = browse or mode == 'browse'
            start.record()
            cell_out, memory_state, prepend_state = self.memory_cell(**segment, memory_state=memory_state, prepend_state=prepend_state, browse=browse, output_hidden_states=True)
            end.record()

            if profile:
                torch.cuda.synchronize()
                self.logger.info('segment ' + str(seg_num) + ' elapsed time: ' + str(start.elapsed_time(end)) + ' ms')

            cell_outputs.append(cell_out)
            if len(cell_outputs) > n_cell_out:
                cell_outputs.pop(0)
            
            if self.cross_attn is not None:
                if memory_seq is None:
                    memory_seq = memory_state.cpu()
                else:
                    memory_seq = torch.cat([memory_seq, memory_state.cpu()], dim=1)
                    if memory_seq.shape[1] > self.ltm_context:
                        memory_seq = memory_seq[:,-self.ltm_context:,:]

            if memory_state is not None:
                self.manage_gradients(memory_state, seg_num)

            seg_num+=1
        
        if profile:
            self.logger.info('end inferencing segments')

        out = self.process_outputs(cell_outputs, labels=labels, 
                                   labels_mask=labels_mask,
                                   output_attentions=output_attentions, 
                                   output_hidden_states=output_hidden_states)
        return out, total_hist
    
    def generate(self, input_ids, attention_mask, **generate_kwargs):
        #TODO: rewrite generation function
        memory_state = None
        segmented = self.segment(input_ids=input_ids, attention_mask=attention_mask)

        for seg_num, segment in enumerate(segmented[:-1]):
            cell_out, memory_state = self.memory_cell(**segment, memory_state=memory_state, output_hidden_states=True)

        final_segment = segmented[-1]
        out = self.memory_cell.generate(**final_segment, memory_state=memory_state, **generate_kwargs)

        return out

    def segment(self, segment_size, **kwargs):
        segments = []
        for k, tensor in kwargs.items():
            if tensor is not None:
                k_segments = self.split_tensor(tensor, segment_size)
                for s, k_seg in enumerate(k_segments):
                    if s < len(segments):
                        segments[s][k] = k_seg
                    else:
                        segments.append({k: k_seg})

        return segments
    
    def split_tensor(self, tensor, segment_size):
        align = self.rmt_config.get('segment_alignment')
        if align in {'left', None}:
            split_inds = list(range(0, tensor.shape[1], segment_size)) + [tensor.shape[1]]
            segments = [tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])]
        elif align in {'right', None}:
            split_inds = (list(range(tensor.shape[1], 0, -segment_size)) + [0])[::-1]
            segments = [tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])]
        elif align == 'center':
            n_seg = math.ceil(tensor.shape[1] / segment_size)
            segments = torch.chunk(tensor, n_seg, dim=1)
        else:
            raise NotImplementedError
        return segments

    def process_outputs(self, cell_outputs, **kwargs):
        mask_size = self.rmt_config.get('mask_size')
        out = CausalLMOutputWithCrossAttentions()
        full_logits = torch.cat([o.logits for o in cell_outputs], dim=1)
        full_hidden_states = tuple([torch.cat(layer_hs, dim=1) for layer_hs in zip(*[o.hidden_states for o in cell_outputs])])

        labels = kwargs.get('labels')
        if labels is not None:
            shift_labels = labels[..., -mask_size:].contiguous()
            shift_logits = full_logits[..., -(mask_size+1):-1, :].contiguous()
            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            
            loss_fct = CrossEntropyLoss()
            out['loss'] = loss_fct(flat_logits.cuda(), flat_labels.cuda())
            out['ppl'] = torch.exp(out['loss'])
        else:
            out['loss'] = 0
            out['ppl'] = 0

        out['logits'] = full_logits
        segment_keys = ['loss', 'logits']
        if kwargs.get('output_attentions'):
            segment_keys.append('attentions')
        if kwargs.get('output_hidden_states'):
            segment_keys.append('hidden_states')
            out['hidden_states'] = full_hidden_states

        for seg_num, o in enumerate(cell_outputs):
            for key, value in o.items():
                if any([sk in key for sk in segment_keys]):
                    out[f'{key}_{seg_num}'] = value

        return out 
        
    def manage_gradients(self, memory_state, seg_num):
        k2, max_n_segments = self.rmt_config.get('k2'), self.rmt_config.get('max_n_segments')
        if seg_num == 0 \
            or k2 in {-1, None} \
            or seg_num + k2 > max_n_segments:
                return True
        
        memory_state = memory_state.detach()
        return False