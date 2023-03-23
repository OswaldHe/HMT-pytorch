import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from .base import RMTBaseModel
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

# from .base import RMTBaseModel
class RMTDecoderForCausalLM(RMTBaseModel):
    def extend_word_embeddings(self, num_mem_tokens, tokenizer):
            
        vocab_size = self.model.config.vocab_size
        extended_vocab_size = vocab_size + num_mem_tokens
        self.num_mem_tokens = num_mem_tokens
        self.register_buffer('mem_token_ids', torch.arange(vocab_size, vocab_size + num_mem_tokens))
        self.model.resize_token_embeddings(extended_vocab_size)

        self.read_memory_position = range(num_mem_tokens)
        self.write_memory_position = range(-num_mem_tokens, 0)
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

        if not hasattr(self, 'memory_states') or self.memory_states is None:
            init_memory = self.set_memory(input_ids.shape)
            self.memory_states = [(None, init_memory)]
        
        memory = self.memory_states[-1][1].detach()
        memory.requires_grad = True

        segment_input_ids = self.pad_and_segment(input_ids)[0]

        seg_kwargs, non_empty_mask = self.prepare_kwargs(segment_input_ids, kwargs)
        seg_kwargs['inputs_embeds'][:, self.read_memory_position] = memory
        # seg_kwargs['inputs_embeds'][:, self.write_memory_position] = self.memory
        
        labels = seg_kwargs.pop('labels')
        out = self.model(**seg_kwargs)
        
        new_memory = out.hidden_states[-1][:, self.write_memory_position]
        self.memory_states.append((memory, new_memory))
        self.trim_memory_states()

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
        self.memory_states = None
        super().train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        self.memory_states = None
        super().eval(*args, **kwargs)

    def trim_memory_states(self):
        k2 = self.rmt_config.get('k2')
        if not k2 or k2 == -1:
            return 
        while len(self.memory_states) > k2:
            del self.memory_states[0]

    def truncated_backward(self, k1, k2):
        memory_states = self.memory_states
        if k1 != -1:
            raise NotImplementedError
        
        for i in range(k2 - 1 if k2 != -1 else len(memory_states)):
            curr_grad = memory_states[-i-1][0].grad
            memory_states[-i-2][1].backward(curr_grad, retain_graph=False)

            # if we get all the way back to the "init_memory", stop
            if memory_states[-i-2][0] is None:
                break
