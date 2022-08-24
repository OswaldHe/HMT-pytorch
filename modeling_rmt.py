import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers import PreTrainedModel, AutoModelForSequenceClassification

import math
import types

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

def encoder_memory_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = False,
    output_hidden_states: Optional[bool] = False,
    return_dict: Optional[bool] = True,
    memory_storage = None
) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
    '''
    Copy-pasted from BERT encoder
    '''
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None
    all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

    next_decoder_cache = () if use_cache else None
    for i, layer_module in enumerate(self.layer):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_head_mask = head_mask[i] if head_mask is not None else None
        past_key_value = past_key_values[i] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            # TBD: load logger 
            # if use_cache:
            #     logger.warning(
            #         "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            #     )
            #     use_cache = False

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, past_key_value, output_attentions)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer_module),
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            num_mem = memory_storage['num_mem_tokens']
            if (i in memory_storage) and (i > 0):
                layer_memory = memory_storage[i]
                for j, h in enumerate(hidden_states):
                    hidden_states[j][:layer_memory[j].shape[0]] = layer_memory[j]

            # print(f'hidden states shape: {len(hidden_states), hidden_states[0].shape}\n memory storage:{memory_storage.keys()}')
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
            
        hidden_states = layer_outputs[0]
        if i in memory_storage:
            # print(f'replacing ms[i] {memory_storage[i][0][0][:10]}... to {[h[:num_mem] for h in hidden_states][0][0][:10]}')
            memory_storage[i] = [h[:num_mem] for h in hidden_states]

        # print(f'Overrided method message: hidden states shape: {len(hidden_states), hidden_states[0].shape}\n memory storage:{memory_storage}')
        if use_cache:
            next_decoder_cache += (layer_outputs[-1],)
        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if self.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                next_decoder_cache,
                all_hidden_states,
                all_self_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=next_decoder_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
    )


class RMTEncoderForSequenceClassification():
    '''
    Usage
    
    Way 1: from config
    rmt = RMTEncoderForSequenceClassification(config=config)

    Way 2: from HF model name
    model_name = "bert-base-cased"
    rmt = RMTEncoderForSequenceClassification.from_pretrained(model_name)

    Way 3: from instance of HF model
    model = AutoModelForSequenceClassification("bert-base-cased") 
    rmt = RMTEncoderForSequenceClassification(base_model=model)

    '''
    def __init__(self, config=None, base_model=None, **kwargs):
        if config is not None:
            self.model = AutoModelForSequenceClassification(config, **kwargs)
        
        if base_model is not None:
            self.model = base_model


    def from_pretrained(from_pretrained, **kwargs):
        base_model = AutoModelForSequenceClassification.from_pretrained(from_pretrained, **kwargs)
        rmt = RMTEncoderForSequenceClassification(base_model=base_model)
        return rmt
        

    def set_params(self, 
                backbone_cls=None,
                num_mem_tokens=0, 
                inter_layer_memory=False,
                segment_ordering='regular',
                padding_side = 'left',
                input_size=None, 
                input_seg_size=None, 
                bptt_depth=-1, 
                drop_empty_segments=True,
                sum_loss=False,
                tokenizer=None
                  ):
        
        if input_size is not None:
            self.input_size = input_size
        else:
            self.input_size =  self.base_model.embeddings.position_embeddings.weight.shape[0]
        self.input_seg_size = input_seg_size

        self.bptt_depth = bptt_depth
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token = torch.tensor([tokenizer.cls_token_id])
        self.sep_token = torch.tensor([tokenizer.sep_token_id])
        
        self.num_mem_tokens = num_mem_tokens
        self.segment_ordering = segment_ordering
        self.padding_side = padding_side
        self.drop_empty_segments = drop_empty_segments
        self.sum_loss = sum_loss
        self.extend_word_embeddings()

        if inter_layer_memory:
            self.override_encoder_forward()


    def set_memory(self, memory=None):
        if memory is None:
            mem_token_ids = self.mem_token_ids.to(device=self.device)
            memory = self.base_model.embeddings.word_embeddings(mem_token_ids)
        return memory
    

    def extend_word_embeddings(self):
        vocab_size = self.base_model.embeddings.word_embeddings.weight.shape[0]
        extended_vocab_size = vocab_size + self.num_mem_tokens
        self.mem_token_ids = torch.arange(vocab_size, vocab_size + self.num_mem_tokens)
        self.base_model.resize_token_embeddings(extended_vocab_size)
    
    
    def override_encoder_forward(self):
        self.memory_storage = {'num_mem_tokens': self.num_mem_tokens}
        memory_forward = lambda *args, **kwargs: encoder_memory_forward(*args, **kwargs, memory_storage=self.memory_storage)
        self.base_model.encoder.forward = types.MethodType(memory_forward, self.base_model.encoder)


    def __call__(self, input_ids, **kwargs):
        memory = self.set_memory()
        segmented = self.pad_and_segment(input_ids)
        segmented = list(zip(*segmented))

        if self.segment_ordering in {'regular', 'last_memory_only'}:
            pass
        elif self.segment_ordering == 'reversed':
            segmented = segmented[::-1]
        elif self.segment_ordering == 'bidirectional':
            segmented = segmented + segmented[::-1][1:]
        elif self.segment_ordering == 'repeat_first':
            segmented = segmented + segmented[:1]
        else:
            raise ValueError(f'Unknown segment ordering: {self.segment_ordering}')

        self.memory_storage = {'num_mem_tokens': self.num_mem_tokens}
        outputs = []
        for seg_num, segment_data in enumerate(segmented):
            input_ids, attention_mask, token_type_ids = segment_data
            if memory.ndim == 2:
                memory = memory.repeat(input_ids.shape[0], 1, 1)
            if (self.bptt_depth > -1) and (len(segmented) - seg_num > self.bptt_depth): 
                memory = memory.detach()

            seg_kwargs = dict(**kwargs)
            if self.drop_empty_segments:
                non_empty_mask = [not torch.equal(input_ids[i], self.empty) for i in range(len(input_ids))]
                if sum(non_empty_mask) == 0:
                    continue
                input_ids = input_ids[non_empty_mask]
                attention_mask = attention_mask[non_empty_mask]
                token_type_ids = token_type_ids[non_empty_mask]
                seg_kwargs['labels'] = seg_kwargs['labels'][non_empty_mask]

                inputs_embeds = self.base_model.embeddings.word_embeddings(input_ids)
                inputs_embeds[:, 1:1+self.num_mem_tokens] = memory[non_empty_mask]
            else:
                inputs_embeds = self.base_model.embeddings.word_embeddings(input_ids)
                inputs_embeds[:, 1:1+self.num_mem_tokens] = memory

            seg_kwargs['inputs_embeds'] = inputs_embeds
            seg_kwargs['attention_mask'] = attention_mask
            seg_kwargs['token_type_ids'] = token_type_ids
            
            out = self.model.forward(**seg_kwargs, output_hidden_states=True)
            outputs.append(out)

            if self.drop_empty_segments:
                memory[non_empty_mask] = out.hidden_states[-1][:, :self.num_mem_tokens]
            else:
                memory = out.hidden_states[-1][:, :self.num_mem_tokens]

        for i, o in enumerate(outputs):
            out[f'loss_{i}'] = o['loss'].mean()

        if self.sum_loss:
            out['loss'] = torch.stack([o['loss'] for o in outputs]).sum(dim=-1)
            
        return out

    def pad_and_segment(self, input_ids):
        
        sequence_len = input_ids.shape[1]
        input_seg_size = self.input_size - self.num_mem_tokens - 3 
        if self.input_seg_size is not None and self.input_seg_size < input_seg_size:
            input_seg_size = self.input_seg_size
            
        n_segments = math.ceil(sequence_len / input_seg_size)

        augmented_inputs = []
        for input in input_ids:
            input = input[input != self.pad_token_id][1:-1]

            if self.padding_side == 'left':
                seg_sep_inds = [0] + list(range(len(input), 0, -input_seg_size))[::-1] # chunk so that first segment has various size
            else:
                seg_sep_inds = list(range(0, len(input), input_seg_size)) + [len(input)]
            input_segments = [input[s:e] for s, e in zip(seg_sep_inds, seg_sep_inds[1:])]

            def pad_add_special_tokens(tensor, seg_size):
                tensor = torch.cat([self.cls_token.to(device=self.device),
                                    self.mem_token_ids.to(device=self.device),
                                    self.sep_token.to(device=self.device),
                                    tensor.to(device=self.device),
                                    self.sep_token.to(device=self.device)])
                pad_size = seg_size - tensor.shape[0]
                if pad_size > 0:
                    tensor = F.pad(tensor, (0, pad_size))
                return tensor

            input_segments = [pad_add_special_tokens(t, self.input_size) for t in input_segments]
            empty = torch.Tensor([]).int()
            self.empty = pad_add_special_tokens(empty, self.input_size)
            empty_segments = [self.empty for i in range(n_segments - len(input_segments))]
            input_segments = empty_segments + input_segments

            augmented_input = torch.cat(input_segments)
            augmented_inputs.append(augmented_input)
            
        augmented_inputs = torch.stack(augmented_inputs)
        attention_mask = torch.ones_like(augmented_inputs)
        attention_mask[augmented_inputs == self.pad_token_id] = 0

        token_type_ids = torch.zeros_like(attention_mask)

        input_segments = torch.chunk(augmented_inputs, n_segments, dim=1)
        attention_mask = torch.chunk(attention_mask, n_segments, dim=1)
        token_type_ids = torch.chunk(token_type_ids, n_segments, dim=1)
    
        return input_segments, attention_mask, token_type_ids


    def to(self, device):
        self.model = self.model.to(device)
        
    
    def cuda(self):
        self.model.cuda()


    def __getattr__(self, attribute):
        return getattr(self.model, attribute)


    def parameters(self, **kwargs):
        return self.model.parameters(**kwargs)

    def named_parameters(self, **kwargs):
        return self.model.named_parameters(**kwargs)
