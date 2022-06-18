import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers import PreTrainedModel, AutoModelForConditionalGeneration

import math

SEGMENTABLE = ['input_ids', 'inputs_embeds', 'token_type_ids', 'position_ids', 'attention_mask']
PAD_ZEROS = ['token_type_ids', 'attention_mask']

# ## todo
# * how to extend embedding simultaneously 4 enc&dec?
# * update forward procedure
# * generate method


class RMTEncoderDecoderForConditionalGeneration():
    def __init__(self, config=None, base_model=None):
        if config is not None:
            self.model = AutoModelForConditionalGeneration(config)
        
        if base_model is not None:
            self.model = base_model

        self.num_mem_tokens = 0


    def from_pretrained(from_pretrained):
        base_model = AutoModelForSequenceClassification.from_pretrained(from_pretrained)
        rmt = RMTEncoderForSequenceClassification(base_model=base_model)
        return rmt


    def set_mem_tokens(self, num_mem_tokens, bptt_depth=-1):
        self.bptt_depth = bptt_depth
        self.num_mem_tokens = num_mem_tokens
        self.extend_word_embeddings()


    def set_memory(self, memory=None):
        if memory is None:
            mem_token_ids = self.mem_token_ids.to(device=self.device)
            memory = self.model.encoder.embeddings.word_embeddings(mem_token_ids)
        return memory


    def extend_word_embeddings(self):
        vocab_size = self.model.encoder.embeddings.word_embeddings.weight.shape[0]
        extended_vocab_size = vocab_size + self.num_mem_tokens + 1
        self.mem_token_ids = torch.arange(vocab_size, vocab_size + self.num_mem_tokens)
        self.pad_token_id = vocab_size + self.num_mem_tokens
        self.model.resize_token_embeddings(extended_vocab_size)


    def __call__(self, memory=None, return_memory=False, **kwargs):
        memory = self.set_memory(memory)
        
        segmented_kwargs = self.pad_and_segment(**kwargs)
        for seg_num, seg_kwargs in enumerate(segmented_kwargs):
            if seg_kwargs['input_ids'] is not None:
                input_embeds = self.model.encoder.embeddings.word_embeddings(seg_kwargs.pop('input_ids'))
            else:
                input_embeds = seg_kwargs.pop('inputs_embeds')
            
            if memory.ndim == 2:
                memory = memory.repeat(input_embeds.shape[0], 1, 1)
            if (self.bptt_depth > -1) and (len(segmented_kwargs) - seg_num > self.bptt_depth): 
                memory = memory.detach()

            seg_kwargs['inputs_embeds'] = torch.hstack((memory, input_embeds))
            
            out = self.model.forward(**seg_kwargs, output_hidden_states=True)
            memory = out.hidden_states[-1][:, :self.num_mem_tokens]

        if return_memory:
            return out, memory

        return out

    
    def generate(self, **kwargs):
        raise(NotImplementedError)

    def pad_and_segment(self, **kwargs):

        context_size = self.model.encoder.embeddings.position_embeddings.weight.shape[0] - self.num_mem_tokens

        if 'input_ids' in kwargs:
            sequence_len = kwargs['input_ids'].shape[1]
        elif 'input_embeds' in kwargs:
            sequence_len = kwargs['input_embeds'].shape[1]
        else:
            raise(ValueError)

        n_segments = math.ceil(sequence_len / context_size)
        segmented_kwargs = {}
        for label, value in kwargs.items():
            if label in SEGMENTABLE and isinstance(value, torch.Tensor):
                pad_length = n_segments * context_size - value.shape[1]
                padded_value = self.pad(label, value, pad_length)

                segmented_value = list(torch.chunk(padded_value, n_segments, dim=1) )
                
                if label not in {'input_ids', 'input_embeds'}:
                    for i, seg in enumerate(segmented_value):
                        segmented_value[i] = self.pad(label, seg, self.num_mem_tokens)
                segmented_kwargs[label] = segmented_value
            else:
                segmented_kwargs[label] = [value] * n_segments
        
        segmented_kwargs = [dict(zip(segmented_kwargs.keys(), seg_values)) for seg_values in zip(*segmented_kwargs.values())]
        return segmented_kwargs


    def pad(self, label, value, pad_length):
        if label in PAD_ZEROS:
            pad_value = 0
        elif label == 'input_ids':
            pad_value = self.pad_token_id
        elif label == 'input_embeds':
            pad_value = self.model.encoder.embeddings.word_embeddings(self.pad_token_id.to(device=self.device))

        padded_value = F.pad(value, (pad_length, 0, 0, 0), "constant", pad_value)
        return padded_value


    def to(self, device):
        self.model = self.model.to(device)
        
    
    def cuda(self):
        self.model.cuda()


    def __getattr__(self, attribute):
        return getattr(self.model, attribute)


    def parameters(self, **kwargs):
        return self.model.parameters(**kwargs)

