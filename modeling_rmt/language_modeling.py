import math
import torch
import copy
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from modeling_rmt.long_mem_cross_attn import CrossAttentionMemory

class MemoryCell(torch.nn.Module):
    def __init__(self, base_model, num_mem_tokens):
        super().__init__()
        self.model = base_model
        self.n_prepend = 32
        self.prepend_list = None
        self.create_memory(num_mem_tokens)

    def create_memory(self, num_mem_tokens):
        self.num_mem_tokens = num_mem_tokens
        embeddings = self.model.get_input_embeddings()
        if num_mem_tokens > 0:
            memory_dim =  getattr(self.model.config, 'n_embd', self.model.config.word_embed_proj_dim)
            memory_weights = torch.randn((num_mem_tokens, memory_dim)) * embeddings.weight.data.std()
            signifier_weights = torch.randn((1, memory_dim)) * embeddings.weight.data.std()
            self.register_parameter('memory', torch.nn.Parameter(memory_weights, requires_grad=True))
            self.register_parameter('signifier', torch.nn.Parameter(signifier_weights, requires_grad=False))

        self.read_memory_position = range(num_mem_tokens)
        self.write_memory_position = range(-num_mem_tokens, 0)

    def set_memory(self, input_shape):
        memory = self.memory.repeat(input_shape[0], 1, 1)
        return memory

    def forward(self, input_ids, memory_state=None, prepend_state=None, **kwargs):
        input_ids = input_ids.cuda()
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                kwargs[k] = v.cuda()
        if memory_state is None and self.num_mem_tokens > 0:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(input_ids, memory_state, prepend_state=prepend_state, **kwargs)

        out = self.model(**seg_kwargs)
        out, new_memory_state = self.process_output(out, 0 if prepend_state is None else self.n_prepend, **kwargs)
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
        
        self.prepend_list = input_ids[:,-self.n_prepend:]

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


import random
class RecurrentWrapper(torch.nn.Module):
    def __init__(self, memory_cell, emb=None, **rmt_kwargs):
        super().__init__()
        self.memory_cell = memory_cell
        self.rmt_config = rmt_kwargs
        if emb is not None:
            memory_weights = torch.randn((1, 2560)) * emb.weight.data.std()
            self.register_parameter('mem', torch.nn.Parameter(memory_weights, requires_grad=True))
            self.cross_attn = CrossAttentionMemory(30, 2560, 4096)
        else:
            self.cross_attn = None

    def forward(self, input_ids, labels=None, labels_mask=None, inputs_embeds=None, attention_mask=None, output_attentions=None, output_hidden_states=None, segment_size=1022):
        memory_state = None
        prepend_state = None
        segmented = self.segment(segment_size=segment_size, input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        cell_outputs = []
        n_cell_out = self.rmt_config.get('n_cell_out')
        memory_seq = None
        for seg_num, segment in enumerate(segmented):
            if self.cross_attn is not None:
                s_mem = self.mem.repeat(segment['input_ids'].shape[0], 1, 1)
                seg = copy.deepcopy(segment)
                seg['input_ids'] = seg['input_ids'][:,:(segment_size//2)]
                seg['attention_mask'] = seg['attention_mask'][:,:(segment_size//2)]
                _, q_mem, _ = self.memory_cell(**seg, memory_state=s_mem)
                memory_state = self.cross_attn(memory_seq, q_mem)
            cell_out, memory_state, prepend_state = self.memory_cell(**segment, memory_state=memory_state, prepend_state=prepend_state, output_hidden_states=True)
            cell_outputs.append(cell_out)
            if len(cell_outputs) > n_cell_out:
                cell_outputs.pop(0)
            
            if self.cross_attn is not None:
                if memory_seq is None:
                    memory_seq = memory_state.cpu()
                else:
                    memory_seq = torch.cat([memory_seq, memory_state.cpu()], dim=1)
                    if memory_seq.shape[1] > 100:
                        memory_seq = memory_seq[:,-100:,:]

            if memory_state is not None:
                self.manage_gradients(memory_state, seg_num)

        out = self.process_outputs(cell_outputs, labels=labels, 
                                   labels_mask=labels_mask,
                                   output_attentions=output_attentions, 
                                   output_hidden_states=output_hidden_states)
        return out
    
    def generate(self, input_ids, attention_mask, **generate_kwargs):
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
            out['loss'] = loss_fct(flat_logits, flat_labels)
        else:
            out['loss'] = 0

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