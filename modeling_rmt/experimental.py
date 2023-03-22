import torch
from modeling_rmt.sequence_classification import *
from modeling_rmt.conditional_generation import *

class RMTEncoderMemFromSep(RMTEncoderForSequenceClassification):
    
    def extend_word_embeddings(self, num_mem_tokens, tokenizer):        
        vocab_size = self.model.config.vocab_size
        extended_vocab_size = vocab_size + num_mem_tokens
        self.num_mem_tokens = num_mem_tokens
        self.register_buffer('mem_token_ids', torch.arange(vocab_size, vocab_size + num_mem_tokens))
        self.model.resize_token_embeddings(extended_vocab_size)

        special_tokens = tokenizer.special_tokens_map
        mem_start_ind = int('cls_token' in special_tokens or 'bos_token' in special_tokens)
        self.memory_position = range(mem_start_ind, mem_start_ind + num_mem_tokens)
        self.model.embeddings = self.model.get_input_embeddings()

        self.reinit_memory_embeddings()

    def reinit_memory_embeddings(self):
        sep_embedding = self.model.embeddings.weight[self.sep_token][0]
        memory_weights = torch.stack([sep_embedding] * self.num_mem_tokens)
        noise_scale = self.model.embeddings.weight.std() / 10
        noise = torch.randn_like(memory_weights) * noise_scale
        self.model.embeddings.weight.data[self.memory_position] = memory_weights + noise


class RMTEncoderDecoderMemFromEos(RMTEncoderDecoderForConditionalGeneration):
   def extend_word_embeddings(self, num_mem_tokens, tokenizer):
            
        vocab_size = self.model.config.vocab_size
        extended_vocab_size = vocab_size + num_mem_tokens
        self.num_mem_tokens = num_mem_tokens
        self.register_buffer('mem_token_ids', torch.arange(vocab_size, vocab_size + num_mem_tokens))
        self.model.resize_token_embeddings(extended_vocab_size)

        # fix scale and tie weights
        embeddings = self.model.get_input_embeddings()
        embeddings.weight.data[-num_mem_tokens:] = embeddings.weight.data[-num_mem_tokens:].normal_(mean=0.0, std=23.19373) / 2 + embeddings.weight.data[tokenizer.eos_token_id]
        self.model.set_input_embeddings(embeddings)
        self.model.tie_weights()
        # end

        special_tokens = tokenizer.special_tokens_map
        mem_start_ind = int('cls_token' in special_tokens or 'bos_token' in special_tokens)
        self.memory_position = range(mem_start_ind, mem_start_ind + num_mem_tokens)
        self.model.embeddings = self.model.get_input_embeddings()
    

import torch
import torch.nn.functional as F
from modeling_rmt.base import RMTBaseModel
class RMTEncoderTBPTT(RMTEncoderForSequenceClassification):

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        kwargs = {'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
                  'position_ids': position_ids, 'inputs_embeds': inputs_embeds,
                  'labels': labels, 'output_attentions': output_attentions,
                  'output_hidden_states': output_hidden_states, 'return_dict': return_dict,
                  }

        init_memory = self.set_memory(input_ids.shape)
        segmented = self.pad_and_segment(input_ids)
        if self.num_mem_tokens == 0:
            segmented = segmented[-1:]

        memory_states = [(None, init_memory)]
        base_model_outputs = []
        for seg_num, segment_input_ids in enumerate(segmented):
            seg_kwargs, non_empty_mask = self.prepare_kwargs(segment_input_ids, kwargs)
            if sum(non_empty_mask) < 1:
                raise NotImplementedError
            
            memory = memory_states[-1][1].detach()
            memory.requires_grad = True
            seg_kwargs['inputs_embeds'][:, self.memory_position] = memory[non_empty_mask]
            out = self.model(**seg_kwargs)
            base_model_outputs.append(out)
            
            self.memory_states = memory_states
            new_memory = out.hidden_states[-1][:, self.memory_position]
            memory_states.append((memory, new_memory))
        
        self.memory_states = memory_states

        out = self.process_outputs(base_model_outputs, output_attentions, output_hidden_states)
        return out
    
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


from modeling_rmt.conditional_generation import *
from torch.nn import CrossEntropyLoss
class RMTEncoderDecoderFullMemoryLastSeg(RMTEncoderDecoderMemoryLayers):
    def forward(self, input_ids, attention_mask=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        kwargs = {'attention_mask': attention_mask,
                #   'position_ids': position_ids, 
                  'inputs_embeds': inputs_embeds,
                  'labels': labels, 'output_attentions': output_attentions,
                  'output_hidden_states': output_hidden_states, 'return_dict': return_dict,
                  }

        memory = self.set_memory(input_ids.shape)
        segmented = self.pad_and_segment(input_ids)

        memories = []
        base_model_outputs = []
        for seg_num, segment_input_ids in enumerate(segmented):                
            if self.rmt_config['bptt_depth'] != -1:
                raise NotImplementedError

            seg_kwargs, non_empty_mask = self.prepare_kwargs(segment_input_ids, kwargs)
            if sum(non_empty_mask) == 0:
                continue
            
            seg_kwargs['inputs_embeds'][:, self.memory_position] = memory[non_empty_mask]
            out = self.model(**seg_kwargs)
            base_model_outputs.append(out)
            
            memory[non_empty_mask] = out.encoder_hidden_states[-1][:, self.memory_position]
            memories.append(torch.clone(memory))

        hidden_states = torch.cat(memories[:-1] + [out.encoder_hidden_states[-1]], dim=1)
        decoder_input_ids = self.model._shift_right(labels)
        decoder_outputs = self.model.decoder(input_ids=decoder_input_ids, 
                                             encoder_hidden_states=hidden_states, 
                                             output_hidden_states=output_hidden_states, 
                                             output_attentions=output_attentions)
        # base_model_outputs.append(decoder_outputs)

        out = self.process_outputs(base_model_outputs, output_attentions, output_hidden_states)

        sequence_output = decoder_outputs[0]
        # Set device for model parallelism
        if self.model.model_parallel:
            torch.cuda.set_device(self.model.encoder.first_device)
            self.model.lm_head = self.model.lm_head.to(self.model.encoder.first_device)
            sequence_output = sequence_output.to(self.model.lm_head.weight.device)

        if self.model.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model.model_dim**-0.5)

        lm_logits = self.model.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        
        out['loss'] = loss

        return out

    def generate(self, input_ids, attention_mask=None, position_ids=None, head_mask=None,
                inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None,
                min_length=None, max_length=None):
        kwargs = {'attention_mask': attention_mask,
                  'inputs_embeds': inputs_embeds,
                  'output_attentions': output_attentions,
                  'output_hidden_states': output_hidden_states, 'return_dict': return_dict,
                  'min_length': min_length, 'max_length': max_length
                  }

        memory = self.set_memory(input_ids.shape)
        segmented = self.pad_and_segment(input_ids)

        memories = []
        for seg_num, segment_input_ids in enumerate(segmented):                
            if self.rmt_config['bptt_depth'] != -1:
                raise NotImplementedError

            seg_kwargs, non_empty_mask = self.prepare_kwargs(segment_input_ids, kwargs)
            if sum(non_empty_mask) == 0:
                continue
            seg_kwargs['inputs_embeds'][:, self.memory_position] = memory[non_empty_mask]

            for param in ['min_length', 'max_length']:
                if param in seg_kwargs:
                    seg_kwargs.pop(param)
                    
            out = self.model.encoder(**seg_kwargs)
            memory[non_empty_mask] = out.last_hidden_state[:, self.memory_position]
            memories.append(torch.clone(memory))

        hidden_states = torch.cat(memories[:-1] + [out.last_hidden_state], dim=1)
        out = self.model.generate(**seg_kwargs, encoder_hidden_states=hidden_states)

        return out