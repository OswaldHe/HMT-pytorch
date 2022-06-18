import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers import PreTrainedModel, AutoModel
# from modeling_t5 import T5ForConditionalGeneration

import math

SEGMENTABLE = ['input_ids', 'inputs_embeds', 'token_type_ids', 'position_ids', 'attention_mask']
PAD_ZEROS = ['token_type_ids', 'attention_mask']


class RMTEncoderDecoderForConditionalGeneration():
    def __init__(self, config=None, base_model=None, **kwargs):
        if config is not None:
            self.config = config
            self.model = AutoModel(config, **kwargs)
        
        if base_model is not None:
            self.model = base_model


    def from_pretrained(from_pretrained, **kwargs):
        # print(f'Creating from pretrained: {from_pretrained}')
        base_model = AutoModel.from_pretrained(from_pretrained, **kwargs)
        rmt = RMTEncoderDecoderForConditionalGeneration(base_model=base_model)
        rmt.from_pretrained = from_pretrained
        return rmt
        

    def set_params(self, 
                    model_attr='', 
                    backbone_cls=None,
                    input_size=None, 
                    input_seg_size=None, 
                    num_mem_tokens=0, 
                    bptt_depth=-1, 
                    pad_token_id=0, 
                    eos_token_id=1,
                    cls_token_id=101, 
                    sep_token_id=102):
        # print('model attr: ', model_attr)

        if backbone_cls is not None:
            self.model = backbone_cls.from_pretrained(self.from_pretrained)

        if model_attr:
            self.encoder = getattr(self.model, model_attr).encoder
            self.decoder = getattr(self.model, model_attr).decoder
        else:
            self.encoder = self.model.encoder
            self.decoder = self.model.decoder
        self.embeddings = self.encoder.embed_tokens
        self.input_size = self.embeddings.weight.shape[0] if input_size is None else input_size
        self.input_seg_size = input_seg_size

        self.bptt_depth = bptt_depth
        self.pad_token_id = pad_token_id
        self.eos_token = torch.tensor([eos_token_id])
        # self.cls_token = torch.tensor([cls_token_id])
        # self.sep_token = torch.tensor([sep_token_id])
        self.num_mem_tokens = num_mem_tokens
        self.extend_word_embeddings()
        
        # ## HOTFIX
        # if 'bart' in backbone_cls.__name__.lower():
        #     getattr(self.model, model_attr).shared.num_embeddings += self.num_mem_tokens


    def set_memory(self, memory=None):
        if memory is None:
            mem_token_ids = self.mem_token_ids.to(device=self.device)
            # print('setting memory')
            # print('mem_token_ids', mem_token_ids.shape)
            memory = self.embeddings(mem_token_ids)
            # print('memory', memory.shape)
        return memory
    
    def extend_word_embeddings(self):
        vocab_size = self.embeddings.weight.shape[0]
        extended_vocab_size = vocab_size + self.num_mem_tokens
        self.mem_token_ids = torch.arange(vocab_size, vocab_size + self.num_mem_tokens).long()
        # self.encoder.resize_token_embeddings(extended_vocab_size)
        # self.decoder.resize_token_embeddings(extended_vocab_size)
        self.resize_token_embeddings(extended_vocab_size)
        self.embeddings = self.encoder.embed_tokens
        # self.set_output_embeddings(self.embeddings)
        # print('vocab_size', vocab_size)
        # print('extended_vocab_size', extended_vocab_size)
        # print('self.embeddings', self.embeddings.weight.shape)
        # print('self.decoder.embed_tokens', self.decoder.embed_tokens.weight.shape)
        # print('self.mem_token_ids', self.mem_token_ids)


    def __call__(self, input_ids, **kwargs):
        memory = self.set_memory()
        segmented = self.pad_and_segment(input_ids)
        for seg_num, segment_data in enumerate(zip(*segmented)):
            input_ids, attention_mask, token_type_ids = segment_data
            if memory.ndim == 2:
                memory = memory.repeat(input_ids.shape[0], 1, 1)
                # print('memory0',  memory.shape)
            if (self.bptt_depth > -1) and (len(segmented) - seg_num > self.bptt_depth): 
                memory = memory.detach()
            # print('memory1',  memory.shape)

            inputs_embeds = self.embeddings(input_ids)
            inputs_embeds[:, 1:1+self.num_mem_tokens] = memory

            seg_kwargs = dict(**kwargs)
            seg_kwargs['inputs_embeds'] = inputs_embeds
            seg_kwargs['attention_mask'] = attention_mask

            out = self.model.forward(**seg_kwargs, output_hidden_states=True)
            # print('out: ', out.keys())
            memory = out.encoder_hidden_states[-1][:, :self.num_mem_tokens]

        return out


    def generate(self, input_ids, **kwargs):
        memory = self.set_memory()
        segmented = self.pad_and_segment(input_ids)
        min_length, max_length = kwargs.pop('min_length'), kwargs.pop('max_length')
        for seg_num, segment_data in enumerate(zip(*segmented)):
            input_ids, attention_mask, token_type_ids = segment_data
            if memory.ndim == 2:
                memory = memory.repeat(input_ids.shape[0], 1, 1)
            if (self.bptt_depth > -1) and (len(segmented) - seg_num > self.bptt_depth): 
                memory = memory.detach()

            inputs_embeds = self.embeddings(input_ids)
            inputs_embeds[:, 1:1+self.num_mem_tokens] = memory

            seg_kwargs = dict(**kwargs)
            seg_kwargs['inputs_embeds'] = inputs_embeds
            seg_kwargs['attention_mask'] = attention_mask
            if seg_num < len(segmented[0])-1:
                labels = torch.ones(inputs_embeds.shape[0], inputs_embeds.shape[1], device=inputs_embeds.device, dtype=input_ids.dtype)
                out = self.model.forward(**seg_kwargs, output_hidden_states=True, labels=labels)
                memory = out.encoder_hidden_states[-1][:, :self.num_mem_tokens]
            else:
                out = self.model.generate(**seg_kwargs, output_hidden_states=True, min_length=min_length, max_length=max_length)
            
        return out

    def pad_and_segment(self, input_ids):
        
        sequence_len = input_ids.shape[1]
        input_seg_size = self.input_size - self.num_mem_tokens - 1
        if self.input_seg_size is not None and self.input_seg_size < input_seg_size:
            input_seg_size = self.input_seg_size
            
        n_segments = math.ceil(sequence_len / input_seg_size)

        augmented_inputs = []
        for input in input_ids:
            # print('input != self.pad_token_id ', (input != self.pad_token_id).shape, (input != self.pad_token_id).sum())
            # 1/0
            input = input[input != self.pad_token_id][1:-1]

            seg_sep_inds = [0] + list(range(len(input), 0, -input_seg_size))[::-1] # chunk so that first segment has various size
            input_segments = [input[s:e] for s, e in zip(seg_sep_inds, seg_sep_inds[1:])]

            def pad_add_special_tokens(tensor, seg_size):
                tensor = torch.cat([
                                    # self.cls_token.to(device=self.device),
                                    self.mem_token_ids.to(device=self.device),
                                    # self.sep_token.to(device=self.device),
                                    tensor.to(device=self.device),
                                    # self.sep_token.to(device=self.device),
                                    self.eos_token.to(device=self.device)
                                    ])
                pad_size = seg_size - tensor.shape[0]
                if pad_size > 0:
                    tensor = F.pad(tensor, (0, pad_size))
                return tensor

            input_segments = [pad_add_special_tokens(t, self.input_size) for t in input_segments]
            empty = torch.Tensor([]).int()
            empty_segments = [pad_add_special_tokens(empty, self.input_size) for i in range(n_segments - len(input_segments))]
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