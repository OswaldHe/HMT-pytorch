import math
import torch
import torch.nn.functional as F

class RMTBaseModel(torch.nn.Module):
    def __init__(self, base_model, **rmt_kwargs):
        super().__init__()
        self.model = base_model
        self.set_params(**rmt_kwargs)

    def set_params(self, num_mem_tokens, tokenizer, **rmt_config):
        self.rmt_config = rmt_config
        self.extract_special_tokens(tokenizer)
        self.extend_word_embeddings(num_mem_tokens, tokenizer)

        self.segment_size = rmt_config['input_size'] - num_mem_tokens - tokenizer.num_special_tokens_to_add()
        if 'sep_token' in tokenizer.special_tokens_map:
            self.segment_size -= 1

    def set_memory(self, memory=None):
        if memory is None:
            mem_token_ids = self.mem_token_ids
            memory = self.model.embeddings(mem_token_ids)
        return memory

    def extract_special_tokens(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        self.special_token_ids = [tokenizer.pad_token_id]
        for token in ['cls_token', 'sep_token', 'eos_token', 'bos_token']:
            token_id = getattr(tokenizer, f'{token}_id')
            if token_id is not None:
                self.register_buffer(token, torch.tensor([token_id]))
                self.special_token_ids.append(token_id)
            else:
                setattr(self, token, None)

    def extend_word_embeddings(self, num_mem_tokens, tokenizer):
            
        vocab_size = self.model.config.vocab_size
        extended_vocab_size = vocab_size + num_mem_tokens
        self.num_mem_tokens = num_mem_tokens
        self.register_buffer('mem_token_ids', torch.arange(vocab_size, vocab_size + num_mem_tokens))
        self.model.resize_token_embeddings(extended_vocab_size)

        special_tokens = tokenizer.special_tokens_map
        mem_start_ind = int('cls_token' in special_tokens or 'bos_token' in special_tokens)
        self.memory_position = range(mem_start_ind, mem_start_ind + num_mem_tokens)
        
        if hasattr(self.model.base_model, 'embeddings'): # enc-only
            self.model.embeddings = self.model.base_model.embeddings.word_embeddings
        elif hasattr(self.model.encoder, 'embed_tokens'): # enc-dec
            self.model.embeddings = self.model.encoder.embed_tokens
        else:
            raise NotImplementedError

    def forward(self, **kwargs):
       raise NotImplementedError

    def pad_and_segment(self, input_ids):
        segmented_batch = []
        for seq in input_ids:
            drop_mask = sum([seq == t for t in self.special_token_ids])
            seq = seq[(1 - drop_mask).bool()]
            seq = seq[:self.segment_size * self.rmt_config['max_n_segments']]

            align = self.rmt_config.get('segment_alignment')
            if align in {'right', None}:
                split_inds = (list(range(len(seq), 0, -self.segment_size)) + [0])[::-1]
                input_segments = [seq[start:end] for (start, end) in zip(split_inds, split_inds[1:])]
            elif align == 'left':
                split_inds = list(range(0, len(seq), self.segment_size)) + [len(seq)]
                input_segments = [seq[start:end] for (start, end) in zip(split_inds, split_inds[1:])]
            else:
                n_seg = math.ceil(len(seq) / self.segment_size)
                input_segments = torch.chunk(seq, n_seg)

            input_segments = [self.pad_add_special_tokens(t, self.rmt_config['input_size']) for t in input_segments]

            # add empty segment markers if needed
            n_empty_segments = self.rmt_config['max_n_segments'] - len(input_segments)
            input_segments = [None] * n_empty_segments + input_segments

            segmented_batch.append(input_segments)

        segmented_batch = [[sample[seg_num] for sample in segmented_batch] \
                            for seg_num in range(self.rmt_config['max_n_segments'])]
        return segmented_batch

    def pad_add_special_tokens(self, **kwargs):
        raise NotImplementedError

    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask


class RMTEncoderForSequenceClassification(RMTBaseModel):
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        # todo: replace copy-pasted args with @functools.wraps(self.model.forward) decorator
        # need to change Trainer's usage of inspect.getfullargspec to inspect.signature to support @wraps
        kwargs = {'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
                  'position_ids': position_ids, 'inputs_embeds': inputs_embeds,
                  'labels': labels, 'output_attentions': output_attentions,
                  'output_hidden_states': output_hidden_states, 'return_dict': return_dict,
                  }

        memory = self.set_memory()
        memory = memory.repeat(input_ids.shape[0], 1, 1)
        segmented = self.pad_and_segment(input_ids)

        losses = []
        for seg_num, segment_input_ids in enumerate(segmented):                
            if (self.rmt_config['bptt_depth'] > -1) and (len(segmented) - seg_num > self.rmt_config['bptt_depth']): 
                memory = memory.detach()

            seg_kwargs = dict(**kwargs)
            seg_kwargs['output_hidden_states'] = True

            non_empty_mask = [s is not None for s in segment_input_ids]
            if sum(non_empty_mask) == 0:
                continue
            input_ids = torch.stack([s for s in segment_input_ids if s is not None])
            attention_mask = self.get_attention_mask(input_ids)
            token_type_ids = self.get_token_type_ids(input_ids)
            seg_kwargs['labels'] = seg_kwargs['labels'][non_empty_mask]

            inputs_embeds = self.model.embeddings(input_ids)
            inputs_embeds[:, self.memory_position] = memory[non_empty_mask]

            seg_kwargs['input_ids'] = None
            seg_kwargs['inputs_embeds'] = inputs_embeds
            seg_kwargs['attention_mask'] = attention_mask
            seg_kwargs['token_type_ids'] = token_type_ids

            out = self.model(**seg_kwargs)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            losses.append(out['loss'])

        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None

        for i, l in enumerate(losses):
            out[f'loss_{i}'] = l.mean()

        if self.rmt_config['sum_loss']:
            out['loss'] = torch.stack(losses).sum(dim=0)

        return out
    
    def pad_add_special_tokens(self, tensor, segment_size):
        input_elements = []
        input_elements += [self.cls_token, self.mem_token_ids, self.sep_token, tensor, self.sep_token]
        tensor = torch.cat(input_elements)

        pad_size = segment_size - tensor.shape[0]
        if pad_size > 0:
            tensor = F.pad(tensor, (0, pad_size))
        return tensor
    
    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)


import types
import rmt_utils
class RMTEncoderMemoryLayers(RMTEncoderForSequenceClassification):
    def set_params(self, num_mem_tokens, tokenizer, **rmt_config):
        super().set_params(num_mem_tokens, tokenizer, **rmt_config)
        self.add_memory_layers()
        self.override_encoder_forward(rmt_config.get('memory_forward_func'))

    def override_encoder_forward(self, memory_forward_func):
        if memory_forward_func is None:
            from rmt_utils.encoder.memory_layers import memory_layers_forward
            memory_forward_func = memory_layers_forward
        encoder_forward = lambda *args, **kwargs: memory_forward_func(*args, **kwargs, rmt_parent=self)
        self.model.base_model.encoder.forward = types.MethodType(encoder_forward, self.model.base_model.encoder)

    def add_memory_layers(self):
        memory_layers, share_memory_layers = self.rmt_config.get('memory_layers'), self.rmt_config.get('share_memory_layers')
        if memory_layers is None:
            self.memory_layers = None
        else:
            if memory_layers == 'all':
                memory_layers = range(len(self.model.base_model.encoder.layer))
            else:
                raise NotImplementedError
                
            if share_memory_layers:
                memory_layer = copy.deepcopy(self.model.base_model.encoder.layer[0])
                self.memory_layers = [memory_layer for _ in range(len(memory_layers))]
                for n, p in memory_layer.named_parameters():
                    param_name = re.sub('\.', '_', f'memory_{n}')
                    self.register_parameter(param_name, p)
            else:
                self.memory_layers = [copy.deepcopy(self.model.base_model.encoder.layer[int(l)]) for l in memory_layers]
                for ln, layer in enumerate(self.memory_layers):
                    for n, p in layer.named_parameters():
                        param_name = re.sub('\.', '_', f'{ln}_memory_{n}')
                        self.register_parameter(param_name, p)


class RMTEncoderDecoderForConditionalGeneration(RMTBaseModel):
    def forward(self, input_ids, attention_mask=None,
                head_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        # todo: replace copy-pasted args with @functools.wraps(self.model.forward) decorator
        # need to change Trainer's usage of inspect.getfullargspec to inspect.signature to support @wraps
        kwargs = {'attention_mask': attention_mask, 
                  'head_mask': head_mask, 'inputs_embeds': inputs_embeds,
                  'labels': labels, 'output_attentions': output_attentions,
                  'output_hidden_states': output_hidden_states, 'return_dict': return_dict,
                  }
        memory = self.set_memory()
        memory = memory.repeat(input_ids.shape[0], 1, 1)
        segmented = self.pad_and_segment(input_ids)
        
        losses = {}
        for seg_num, segment_input_ids in enumerate(segmented):
            if (self.rmt_config['bptt_depth'] > -1) and (len(segmented) - seg_num > self.rmt_config['bptt_depth']): 
                memory = memory.detach()

            seg_kwargs = dict(**kwargs)
            seg_kwargs['output_hidden_states'] = True
            
            non_empty_mask = [s is not None for s in segment_input_ids]
            if sum(non_empty_mask) == 0:
                continue
            input_ids = torch.stack([s for s in segment_input_ids if s is not None])
            attention_mask = self.get_attention_mask(input_ids)
            seg_kwargs['labels'] = seg_kwargs['labels'][non_empty_mask]

            inputs_embeds = self.model.embeddings(input_ids)
            inputs_embeds[:, self.memory_position] = memory[non_empty_mask]
    
            seg_kwargs['inputs_embeds'] = inputs_embeds
            seg_kwargs['attention_mask'] = attention_mask
                
            out = self.model.forward(**seg_kwargs)
            memory[non_empty_mask] = out.encoder_hidden_states[-1][:, self.memory_position]

            losses[f'loss_{seg_num}'] = out['loss']

        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None
                    
        for k, loss in losses.items():
            out[k] = loss

        if self.rmt_config['sum_loss']:
            out['loss'] = torch.stack(losses).sum(dim=0)

        return out

    def generate(self, input_ids, **kwargs):
        memory = self.set_memory()
        memory = memory.repeat(input_ids.shape[0], 1, 1)
        segmented = self.pad_and_segment(input_ids)

        for seg_num, segment_input_ids in enumerate(segmented):                
            if (self.rmt_config['bptt_depth'] > -1) and (len(segmented) - seg_num > self.rmt_config['bptt_depth']): 
                memory = memory.detach()

            seg_kwargs = dict(**kwargs)
            seg_kwargs['output_hidden_states'] = True
            non_empty_mask = [s is not None for s in segment_input_ids]
            if sum(non_empty_mask) == 0:
                continue
            input_ids = torch.stack([s for s in segment_input_ids if s is not None])
            attention_mask = self.get_attention_mask(input_ids)

            inputs_embeds = self.model.embeddings(input_ids)
            inputs_embeds[:, self.memory_position] = memory[non_empty_mask]

            seg_kwargs['inputs_embeds'] = inputs_embeds
            seg_kwargs['attention_mask'] = attention_mask

            if seg_num == len(segmented) - 1:
                out = self.model.generate(**seg_kwargs)
            else:
                for param in ['min_length', 'max_length']:
                    if param in seg_kwargs:
                        seg_kwargs.pop(param)
                        
                out = self.model.encoder(**seg_kwargs)
                memory[non_empty_mask] = out.last_hidden_state[:, self.memory_position]
        
        return out

    def pad_add_special_tokens(self, tensor, segment_size):
        input_elements = []
        if self.bos_token is not None:
            input_elements.append(self.bos_token)
        input_elements += [self.mem_token_ids, tensor, self.eos_token]
        tensor = torch.cat(input_elements)

        pad_size = segment_size - tensor.shape[0]
        if pad_size > 0:
            tensor = F.pad(tensor, (0, pad_size))
        return tensor


import types
class RMTEncoderDecoderMemoryLayers(RMTEncoderDecoderForConditionalGeneration):
    def set_params(self, num_mem_tokens, tokenizer, **rmt_config):
        super().set_params(num_mem_tokens, tokenizer, **rmt_config)
        self.add_memory_layers()
        self.override_encoder_forward(rmt_config.get('memory_forward_func'))

    def override_encoder_forward(self, memory_forward_func):
        if memory_forward_func is None:
            from rmt_utils.encoder_decoder.memory_layers import memory_layers_forward
            memory_forward_func = memory_layers_forward
        encoder_forward = lambda *args, **kwargs: memory_forward_func(*args, **kwargs, rmt_parent=self)
        self.model.base_model.encoder.forward = types.MethodType(encoder_forward, self.model.base_model.encoder)

    def add_memory_layers(self):
        memory_layers, share_memory_layers = self.rmt_config.get('memory_layers'), self.rmt_config.get('share_memory_layers')
        if memory_layers is None:
            self.memory_layers = None
        else:
            if memory_layers == 'all':
                memory_layers = range(len(self.model.encoder.block))
            else:
                raise NotImplementedError
                
            if share_memory_layers:
                memory_layer = copy.deepcopy(self.model.encoder.block[0])
                self.memory_layers = [memory_layer for _ in range(len(memory_layers))]
                for n, p in memory_layer.named_parameters():
                    param_name = re.sub('\.', '_', f'memory_{n}')
                    self.register_parameter(param_name, p)
            else:
                self.memory_layers = [copy.deepcopy(self.model.encoder.block[int(l)]) for l in memory_layers]
                for ln, layer in enumerate(self.memory_layers):
                    for n, p in layer.named_parameters():
                        param_name = re.sub('\.', '_', f'{ln}_memory_{n}')
                        self.register_parameter(param_name, p)


import copy
import re
from torch.nn import CrossEntropyLoss
class RMTEncoderDecoderMemoryLoss(RMTEncoderDecoderMemoryLayers):
    def set_params(self, num_mem_tokens, tokenizer, **rmt_config):
        super().set_params(num_mem_tokens, tokenizer, **rmt_config)
        self.add_reconstruction_layers()

    def add_reconstruction_layers(self):
        self.rec_attn = copy.deepcopy(self.model.decoder.block[-1])
        self.rec_cls = copy.deepcopy(self.model.lm_head)

        for n, p in self.rec_attn.named_parameters():
            param_name = re.sub('\.', '_', f'rec_attn_{n}')
            self.register_parameter(param_name, p)
        
        for n, p in self.rec_cls.named_parameters():
            param_name = re.sub('\.', '_', f'rec_cls_{n}')
            self.register_parameter(param_name, p)

    
    def segment_reconstruction_forward(self, reconstruction_labels, encoder_out):

        attention_mask = torch.ones(encoder_out.shape[1]).to(device=reconstruction_labels.device)
        attention_mask[self.num_mem_tokens:] = 0

        rec_attn_out = self.rec_attn(encoder_out, attention_mask=attention_mask)
        rec_logits = self.rec_cls(rec_attn_out[0])

        loss_fct = CrossEntropyLoss()
        reconstruction_loss = loss_fct(rec_logits.view(-1, rec_logits.size(-1)), reconstruction_labels.view(-1))
        reconstruction_loss

        return reconstruction_loss
    
    def forward(self, input_ids, **kwargs):
        memory = self.set_memory()
        memory = memory.repeat(input_ids.shape[0], 1, 1)
        segmented = self.pad_and_segment(input_ids)
        
        losses = {}
        reconstruction_loss = 0
        for seg_num, segment_input_ids in enumerate(segmented):
            if (self.rmt_config['bptt_depth'] > -1) and (len(segmented) - seg_num > self.rmt_config['bptt_depth']): 
                memory = memory.detach()

            seg_kwargs = dict(**kwargs)
            seg_kwargs['output_hidden_states'] = True
            
            non_empty_mask = [s is not None for s in segment_input_ids]
            if sum(non_empty_mask) == 0:
                continue
            input_ids = torch.stack([s for s in segment_input_ids if s is not None])
            attention_mask = self.get_attention_mask(input_ids)
            seg_kwargs['labels'] = seg_kwargs['labels'][non_empty_mask]

            inputs_embeds = self.model.embeddings(input_ids)
            inputs_embeds[:, self.memory_position] = memory[non_empty_mask]
    
            seg_kwargs['inputs_embeds'] = inputs_embeds
            seg_kwargs['attention_mask'] = attention_mask
                
            out = self.model.forward(**seg_kwargs)
            memory[non_empty_mask] = out.encoder_hidden_states[-1][:, self.memory_position]

            losses[f'loss_{seg_num}'] = out['loss']

            segment_reconstruction_loss = self.segment_reconstruction_forward(input_ids, out['encoder_last_hidden_state'])
            out[f'rec_loss_{seg_num}'] = segment_reconstruction_loss
            reconstruction_loss += segment_reconstruction_loss

        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None
                    
        for k, loss in losses.items():
            out[k] = loss

        if self.rmt_config['sum_loss']:
            out['loss'] = torch.stack(losses).sum(dim=0)

        out['reconstruction_loss'] = reconstruction_loss
        
        rec_coef = self.rmt_config['reconstruction_loss_coef']
        out['loss'] = out['reconstruction_loss'] * rec_coef + out['loss'] * (1 - rec_coef)

        return out


from rmt_utils.encoder_decoder.horizontal_memory import horizontal_memory_forward
class RMTEncoderDecoderHorizontalMemory(RMTEncoderDecoderMemoryLayers):
    def set_params(self, num_mem_tokens, tokenizer, **rmt_config):
        super().set_params(num_mem_tokens, tokenizer, **rmt_config)
        self.add_memory_layers()
        
        memory_forward_func = rmt_config.get('memory_forward_func')
        if not memory_forward_func:
            memory_forward_func = horizontal_memory_forward
        self.override_encoder_forward(memory_forward_func)

    def set_memory(self, memory=None):
        self.memory_storage = {}
        if memory is None:
            mem_token_ids = self.mem_token_ids
            memory = self.model.embeddings(mem_token_ids)
        
        # fill layer memories 
        memory_input = self.pad_add_special_tokens(mem_token_ids, self.num_mem_tokens)
        mem_out = self.model.encoder(memory_input.reshape((1, -1)))
        
        return memory

    def forward(self, input_ids, **kwargs):
        memory = self.set_memory()
        memory = memory.repeat(input_ids.shape[0], 1, 1)
        segmented = self.pad_and_segment(input_ids)
        
        losses = []
        for seg_num, segment_input_ids in enumerate(segmented):
            self.memory_storage['seg_num'] = seg_num
            if (self.rmt_config['bptt_depth'] > -1) and (len(segmented) - seg_num > self.rmt_config['bptt_depth']): 
                memory = memory.detach()

            seg_kwargs = dict(**kwargs)
            seg_kwargs['output_hidden_states'] = True
            
            # print('segment_input_ids', len(segment_input_ids), segment_input_ids)
            non_empty_mask = [s is not None for s in segment_input_ids]
            if sum(non_empty_mask) == 0:
                continue
            self.memory_storage['non_empty_mask'] = non_empty_mask
            # print('non_empty_mask', non_empty_mask)
            input_ids = torch.stack([s for s in segment_input_ids if s is not None])
            attention_mask = self.get_attention_mask(input_ids)
            # token_type_ids = self.get_token_type_ids(input_ids)
            seg_kwargs['labels'] = seg_kwargs['labels'][non_empty_mask]

            inputs_embeds = self.model.embeddings(input_ids)
            inputs_embeds[:, self.memory_position] = memory[non_empty_mask]

            seg_kwargs['inputs_embeds'] = inputs_embeds
            seg_kwargs['attention_mask'] = attention_mask
            # seg_kwargs['token_type_ids'] = token_type_ids

            out = self.model.forward(**seg_kwargs)
            memory[non_empty_mask] = out.encoder_hidden_states[-1][:, self.memory_position]

            losses.append(out['loss'])

        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None
                    
        for i, l in enumerate(losses):
            out[f'loss_{i}'] = l.mean()

        if self.rmt_config['sum_loss']:
            out['loss'] = torch.stack(losses).sum(dim=0)
            
        return out


    def generate(self, input_ids, **kwargs):        
        memory = self.set_memory()
        memory = memory.repeat(input_ids.shape[0], 1, 1)
        segmented = self.pad_and_segment(input_ids)

        for seg_num, segment_input_ids in enumerate(segmented):                
            if (self.rmt_config['bptt_depth'] > -1) and (len(segmented) - seg_num > self.rmt_config['bptt_depth']): 
                memory = memory.detach()

            seg_kwargs = dict(**kwargs)
            seg_kwargs['output_hidden_states'] = True
            non_empty_mask = [s is not None for s in segment_input_ids]
            if sum(non_empty_mask) == 0:
                continue
            self.memory_storage['non_empty_mask'] = non_empty_mask
            input_ids = torch.stack([s for s in segment_input_ids if s is not None])
            
            attention_mask = self.get_attention_mask(input_ids)
            # token_type_ids = self.get_token_type_ids(input_ids)

            inputs_embeds = self.model.embeddings(input_ids)
            inputs_embeds[:, self.memory_position] = memory[non_empty_mask]

            seg_kwargs['inputs_embeds'] = inputs_embeds
            seg_kwargs['attention_mask'] = attention_mask

            if seg_num == len(segmented) - 1:
                out = self.model.generate(**seg_kwargs)
            else:
                for param in ['min_length', 'max_length']:
                    if param in seg_kwargs:
                        seg_kwargs.pop(param)
                        
                out = self.model.encoder(**seg_kwargs)
                memory[non_empty_mask] = out.last_hidden_state[:, self.memory_position]
        
        return out