import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from .base import RMTBaseModel

class RMTDecoderLMHead(RMTBaseModel):
    def set_params(self, num_mem_tokens, tokenizer, **rmt_config):
        self.rmt_config = rmt_config
        self.extract_special_tokens(tokenizer)
        self.create_memory(num_mem_tokens)

        self.segment_size = rmt_config['input_size'] - 2 * num_mem_tokens - tokenizer.num_special_tokens_to_add()
        if 'sep_token' in tokenizer.special_tokens_map:
            self.segment_size -= 1

    def create_memory(self, num_mem_tokens):
        self.num_mem_tokens = num_mem_tokens
        embeddings = self.model.get_input_embeddings()
        memory_weights = torch.randn((num_mem_tokens, self.model.config.n_embd)) * embeddings.weight.data.std()
        self.register_parameter('memory', torch.nn.Parameter(memory_weights, requires_grad=True))

        self.read_memory_position = range(num_mem_tokens)
        self.write_memory_position = range(-num_mem_tokens, 0)

    def set_memory(self, input_shape):
        memory = self.memory.repeat(input_shape[0], 1, 1)
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
        
        memory = self.memory_states[-1][1].detach()#.to(input_ids.device)
        memory.requires_grad = True

        segment_input_ids = self.pad_and_segment(input_ids)[0]
        seg_kwargs, non_empty_mask = self.prepare_kwargs(segment_input_ids, memory, kwargs)
        
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
        # pad_size = segment_size - tensor.shape[0]
        # if pad_size > 0:
        #     tensor = F.pad(tensor, (0, pad_size))
        return tensor
    
    def prepare_kwargs(self, segment_input_ids, memory, kwargs):
        seg_kwargs = dict(**kwargs)
        non_empty_mask = [s is not None for s in segment_input_ids]
        if sum(non_empty_mask) == 0:
            return None, non_empty_mask
            
        input_ids = torch.stack([s for s in segment_input_ids if s is not None])
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([memory, inputs_embeds, memory], dim=1)

        seg_kwargs['input_ids'] = None
        seg_kwargs['inputs_embeds'] = inputs_embeds
        if seg_kwargs.get('labels') is not None:
            seg_kwargs['labels'] = seg_kwargs['labels'][non_empty_mask]
        seg_kwargs['attention_mask'] = self.get_attention_mask(inputs_embeds)
        # if seg_kwargs.get('token_type_ids') is not None:
        #     seg_kwargs['token_type_ids'] = self.get_token_type_ids(inputs_embeds)
        seg_kwargs['output_hidden_states'] = True

        return seg_kwargs, non_empty_mask
    
    def get_attention_mask(self, tensor):
        mask = torch.ones(*tensor.shape[:2], dtype=torch.int64).to(tensor.device)
        mask[tensor == self.pad_token_id] = 0
        return mask

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
            memory_states[-i-2][1].backward(curr_grad, retain_graph=k2>2)

            # if we get all the way back to the "init_memory", stop
            if memory_states[-i-2][0] is None:
                break


import types
import copy
import re
# class RMTDecoderMemoryLayers(RMTDecoderForCausalLM):
class RMTDecoderMemoryLayers(RMTDecoderLMHead):
    def set_params(self, num_mem_tokens, tokenizer, **rmt_config):
        super().set_params(num_mem_tokens, tokenizer, **rmt_config)
        self.add_memory_layers()
        self.override_encoder_forward(rmt_config.get('memory_forward_func'))

    def override_encoder_forward(self, memory_forward_func):
        if self.rmt_config.get('memory_layers') is None:
            return
        if memory_forward_func is None:
            from rmt_utils.decoder.memory_layers import memory_layers_forward
            memory_forward_func = memory_layers_forward
        new_forward = lambda *args, **kwargs: memory_forward_func(*args, **kwargs, rmt_parent=self)
        self.model.base_model.forward = types.MethodType(new_forward, self.model.base_model)

    def add_memory_layers(self):
        memory_layers, share_memory_layers = self.rmt_config.get('memory_layers'), self.rmt_config.get('share_memory_layers')
        if memory_layers is None:
            self.memory_layers = None
        else:
            if memory_layers == 'all':
                memory_layers = range(len(self.model.base_model.h))
            else:
                raise NotImplementedError
                
            if share_memory_layers:
                memory_layer = copy.deepcopy(self.model.base_model.h[0])
                self.memory_layers = [memory_layer for _ in range(len(memory_layers))]
                for n, p in memory_layer.named_parameters():
                    param_name = re.sub('\.', '_', f'memory_{n}')
                    self.register_parameter(param_name, p)
            else:
                self.memory_layers = [copy.deepcopy(self.model.base_model.h[int(l)]) for l in memory_layers]
                for ln, layer in enumerate(self.memory_layers):
                    for n, p in layer.named_parameters():
                        param_name = re.sub('\.', '_', f'{ln}_memory_{n}')
                        self.register_parameter(param_name, p)


import math
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class RMTDecoderLMHeadMultiSeg(RMTBaseModel):
    def set_params(self, num_mem_tokens, tokenizer, **rmt_config):
        self.rmt_config = rmt_config
        self.extract_special_tokens(tokenizer)
        self.create_memory(num_mem_tokens)

        self.segment_size = rmt_config['input_size'] - 2 * num_mem_tokens - tokenizer.num_special_tokens_to_add()
        if 'sep_token' in tokenizer.special_tokens_map:
            self.segment_size -= 1

    def create_memory(self, num_mem_tokens):
        self.num_mem_tokens = num_mem_tokens
        embeddings = self.model.get_input_embeddings()
        memory_weights = torch.randn((num_mem_tokens, self.model.config.n_embd)) * embeddings.weight.data.std()
        self.register_parameter('memory', torch.nn.Parameter(memory_weights, requires_grad=True))

        self.read_memory_position = range(num_mem_tokens)
        self.write_memory_position = range(-num_mem_tokens, 0)

    def set_memory(self, input_shape):
        create_memory = self.training \
                        or self.rmt_config.get('reinit_mem_each_fwd') \
                        or not hasattr(self, 'memory_state') \
                        or self.rmt_config['max_n_segments'] == 1 
        if create_memory:
            memory = self.memory.repeat(input_shape[0], 1, 1)
        else:
            memory = self.memory_state[:input_shape[0]]
        return memory
    
    def detach_memory(self, seg_num):
        k2, max_n_segments = self.rmt_config.get('k2'), self.rmt_config.get('max_n_segments')
        if seg_num == 0 \
            or k2 in {-1, None} \
            or seg_num + k2 > max_n_segments:
            return False
        return True
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, labels_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        kwargs = {'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
                  'position_ids': position_ids, 'inputs_embeds': inputs_embeds,
                  'labels_mask': labels_mask, #'pos_weight': pos_weight,
                  'labels': labels, 'output_attentions': output_attentions,
                  'output_hidden_states': output_hidden_states, 'return_dict': return_dict,
                  }

        memory = self.set_memory(input_ids.shape)
        segmented = self.pad_and_segment(input_ids, labels)

        base_model_outputs = []
        for seg_num, segment in enumerate(zip(*segmented)):

            seg_kwargs, non_empty_mask = self.prepare_kwargs(segment, memory, kwargs)
            if sum(non_empty_mask) == 0:
                continue
            
            if self.detach_memory(seg_num):
                memory = memory.detach()

            seg_kwargs['inputs_embeds'][:, self.read_memory_position] = memory[non_empty_mask]
            seg_kwargs['inputs_embeds'][:, self.write_memory_position] = memory[non_empty_mask]
            out = self.model(**seg_kwargs)
            base_model_outputs.append(out)
            
            memory[non_empty_mask] = out.hidden_states[-1][:, self.write_memory_position]

        self.memory_state = memory
        # return (base_model_outputs, kwargs)
        out = self.process_outputs(base_model_outputs, kwargs)
        return out
    
    def pad_and_segment(self, input_ids, labels=None):
        segmented_batch = []
        segmented_batch_labels = []

        if labels is None:
            labels = [None] * input_ids.shape[0]
        batch_labels = labels
        for seq, labels in zip(input_ids, batch_labels):

            align = self.rmt_config.get('segment_alignment')
            if align in {'right', None}:
                split_inds = (list(range(len(seq), 0, -self.segment_size)) + [0])[::-1]
            elif align == 'left':
                split_inds = list(range(0, len(seq), self.segment_size)) + [len(seq)]
            elif align == 'center':
                n_seg = math.ceil(len(seq) / self.segment_size)
                split_inds = list(range(0, len(seq), math.ceil(len(seq) / n_seg))) + [len(seq)]
            else:
                raise NotImplementedError
            input_segments = [seq[start:end] for (start, end) in zip(split_inds, split_inds[1:])]
            # add empty segment markers if needed
            n_empty_segments = self.rmt_config['max_n_segments'] - len(input_segments)
            input_segments = [None] * n_empty_segments + input_segments
            segmented_batch.append(input_segments)

            if labels is not None:
                labels_segments = [labels[start:end] for (start, end) in zip(split_inds, split_inds[1:])]
                labels_segments = [None] * n_empty_segments + labels_segments
                segmented_batch_labels.append(labels_segments)

        segmented_batch = [[sample[seg_num] for sample in segmented_batch]
                           for seg_num in range(self.rmt_config['max_n_segments'])]
        segmented_batch_labels = [[sample[seg_num] for sample in segmented_batch_labels]
                                  for seg_num in range(self.rmt_config['max_n_segments'])]

        return segmented_batch, segmented_batch_labels

    def prepare_kwargs(self, segment, memory, kwargs):
        segment_input_ids, segment_labels = segment
        seg_kwargs = dict(**kwargs)
        non_empty_mask = [s is not None for s in segment_input_ids]
        if sum(non_empty_mask) == 0:
            return None, non_empty_mask

        input_ids = torch.stack([s for s in segment_input_ids if s is not None])
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([memory, inputs_embeds, memory], dim=1)

        seg_kwargs['input_ids'] = None
        seg_kwargs['inputs_embeds'] = inputs_embeds
        seg_kwargs['attention_mask'] = self.get_attention_mask(inputs_embeds)
        seg_kwargs['output_hidden_states'] = True
        if seg_kwargs['labels'] is not None:
            labels = torch.stack([el for el, m in zip(segment_labels, non_empty_mask) if m])
            memory_labels = torch.ones((labels.shape[0], self.num_mem_tokens), dtype=labels.dtype, device=labels.device) * -100
            seg_kwargs['labels'] = torch.cat((memory_labels, labels, memory_labels), dim=1)
            seg_kwargs['labels'][:, self.num_mem_tokens] = -100
        seg_kwargs.pop('labels_mask')

        return seg_kwargs, non_empty_mask
    
    def get_attention_mask(self, tensor):
        mask = torch.ones(*tensor.shape[:2], dtype=torch.int64).to(tensor.device)
        mask[tensor == self.pad_token_id] = 0
        return mask
    
    def process_outputs(self, model_outputs, kwargs):
        full_logits = torch.cat([o.logits[:, self.num_mem_tokens:-self.num_mem_tokens] for o in model_outputs], dim=1)
        truncated_hs = [[lh[:, self.num_mem_tokens:-self.num_mem_tokens] for lh in o.hidden_states] for o in model_outputs]
        full_hidden_states = tuple([torch.cat(layer_hs, dim=1) for layer_hs in zip(*truncated_hs)])

        rmt_out = CausalLMOutputWithCrossAttentions()
        full_labels = kwargs.get('labels')
        if full_labels is not None:
            shift_labels = full_labels[..., 1:].contiguous()
            shift_logits = full_logits[..., :-1, :].contiguous()
            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            
            loss_fct = CrossEntropyLoss()
            labels_mask = kwargs.get('labels_mask')
            if labels_mask is not None:
                shift_mask = labels_mask[..., :-1].contiguous()
                flat_labels = flat_labels[shift_mask.view(-1)]
                flat_logits = flat_logits[shift_mask.view(-1)]
                
            rmt_out['loss'] = loss_fct(flat_logits, flat_labels)

        rmt_out['logits'] = full_logits
        segment_keys = ['loss', 'logits']
        if kwargs.get('output_attentions'):
            segment_keys.append('attentions')
        if kwargs.get('output_hidden_states'):
            segment_keys.append('hidden_states')
            rmt_out['hidden_states'] = full_hidden_states

        for seg_num, out in enumerate(model_outputs):
            for key, value in out.items():
                if any([sk in key for sk in segment_keys]):
                    rmt_out[f'{key}_{seg_num}'] = value

        return rmt_out 


# class RMTDecoderScaleMem(RMTDecoderMemoryLayers):
#    def extend_word_embeddings(self, num_mem_tokens, tokenizer):
#         vocab_size = self.model.config.vocab_size
#         extended_vocab_size = vocab_size + num_mem_tokens
#         self.num_mem_tokens = num_mem_tokens
#         self.register_buffer('mem_token_ids', torch.arange(vocab_size, vocab_size + num_mem_tokens))
#         self.model.resize_token_embeddings(extended_vocab_size)

#         # fix scale and tie weights
#         embeddings = self.model.get_input_embeddings()
#         embeddings.weight.data[-num_mem_tokens:] = embeddings.weight.data[-num_mem_tokens:].normal_(mean=0.0, std=embeddings.weight.data.std()) \
#                                                     / 100 + embeddings.weight.data[tokenizer.eos_token_id]
#         self.model.set_input_embeddings(embeddings)
#         self.model.tie_weights()

#         self.read_memory_position = range(num_mem_tokens)
#         self.write_memory_position = range(-num_mem_tokens, 0)
#         self.model.embeddings = self.model.get_input_embeddings()


# class RMTDecoderMemFromDot(RMTDecoderMemoryLayers):
#    def extend_word_embeddings(self, num_mem_tokens, tokenizer):
#         vocab_size = self.model.config.vocab_size
#         extended_vocab_size = vocab_size + num_mem_tokens
#         self.num_mem_tokens = num_mem_tokens
#         self.register_buffer('mem_token_ids', torch.arange(vocab_size, vocab_size + num_mem_tokens))
#         self.model.resize_token_embeddings(extended_vocab_size)

#         # fix scale and tie weights
#         embeddings = self.model.get_input_embeddings()
#         embeddings.weight.data[-num_mem_tokens:] = embeddings.weight.data[-num_mem_tokens:].normal_(mean=0.0, std=embeddings.weight.data.std()) \
#                                                     / 100 + embeddings.weight.data[tokenizer.encode('.')[0]]
#         self.model.set_input_embeddings(embeddings)
#         self.model.tie_weights()

#         self.read_memory_position = range(num_mem_tokens)
#         self.write_memory_position = range(-num_mem_tokens, 0)
#         self.model.embeddings = self.model.get_input_embeddings()