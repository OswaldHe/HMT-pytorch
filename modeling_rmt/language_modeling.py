import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from .base import RMTBaseModel
class RMTDecoderForCausalLM(RMTBaseModel):
    def extend_word_embeddings(self, num_mem_tokens, tokenizer):
            
        vocab_size = self.model.config.vocab_size
        extended_vocab_size = vocab_size + num_mem_tokens
        self.num_mem_tokens = num_mem_tokens
        self.register_buffer('mem_token_ids', torch.arange(vocab_size, vocab_size + num_mem_tokens))
        self.model.resize_token_embeddings(extended_vocab_size)

        # self.memory_position = list(range(num_mem_tokens)) + list(range(num_mem_tokens))
        self.model.embeddings = self.model.get_input_embeddings()

    def set_memory(self, input_shape):
        memory = self.model.embeddings(self.mem_token_ids)
        memory = memory.repeat(input_shape[0], 1, 1)
        return memory

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        kwargs = {'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
                  'position_ids': position_ids, 'inputs_embeds': inputs_embeds,
                  'labels': labels, 'output_attentions': output_attentions,
                  'output_hidden_states': output_hidden_states, 'return_dict': return_dict,
                  }

        if not hasattr(self, 'memory') or self.memory is None:
            self.memory = self.set_memory(input_ids.shape)

        segment_input_ids = self.pad_and_segment(input_ids)[0]

        seg_kwargs, non_empty_mask = self.prepare_kwargs(segment_input_ids, kwargs)
        seg_kwargs['inputs_embeds'][:, :self.num_mem_tokens] = self.memory
        # seg_kwargs['inputs_embeds'][:, -self.num_mem_tokens:] = self.memory
        labels = seg_kwargs.pop('labels')
        out = self.model(**seg_kwargs)

        self.memory = out.hidden_states[-1][:, -self.num_mem_tokens:].detach()

        ### Calculate loss excluding memory 
        lm_logits = out.logits[:, self.num_mem_tokens:-self.num_mem_tokens]
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        out['loss'] = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return out


    def pad_add_special_tokens(self, tensor, segment_size):
        input_elements = []
        input_elements += [self.mem_token_ids, tensor, self.mem_token_ids]
        tensor = torch.cat(input_elements)

        pad_size = segment_size - tensor.shape[0]
        if pad_size > 0:
            tensor = F.pad(tensor, (0, pad_size))
        return tensor

    def train(self, *args, **kwargs):
        self.memory = None
        super().train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        self.memory = None
        super().eval(*args, **kwargs)