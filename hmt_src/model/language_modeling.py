import math
import torch
import copy
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import OPTConfig
from .long_mem_cross_attn_vanilla import CrossAttentionMemory
from .memory_cell import SummaryCell, MemoryCell
from .utils import SegmentIterator
from accelerate.logging import get_logger
from torch.profiler import profile, record_function, ProfilerActivity
import random
import evaluate
from huggingface_hub import PyTorchModelHubMixin

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

IGNORE_INDEX = -100

class Summary_Memory_RecurrentWrapper(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, base_model, num_mem_embed, num_prepend, mem_hidden_dim=4096, mem_window_size=64, rmt_only=False, baseline_only=False, **rmt_kwargs):
        super().__init__()

        if isinstance(base_model.config, OPTConfig):
            mem_emb_dim = base_model.config.word_embed_proj_dim
        else:
            mem_emb_dim = base_model.config.hidden_size

        self.num_mem_embed = num_mem_embed
        self.num_prepend = num_prepend

        self.memory_cell = MemoryCell(base_model, num_mem_embed=num_mem_embed, num_prepend=num_prepend)
        if rmt_only or baseline_only:
            self.cross_attn = None
            self.summary_cell = None
        else:
            self.cross_attn = CrossAttentionMemory(mem_emb_dim, mem_hidden_dim)
            self.summary_cell = SummaryCell(base_model, num_mem_embed)
            
        self.rmt_config = rmt_kwargs
        self.mem_window_size = mem_window_size
        self.logger = get_logger('')
        
        self.rouge = evaluate.load('rouge')
        self.f1 = evaluate.load("f1")

    def forward(self, 
            input_ids, 
            labels=None, 
            labels_mask=None, 
            inputs_embeds=None, 
            attention_mask=None, 
            mask_size=None,  # Size of the attention mask used to compute the loss, it should be the length of the labels. If it's None, then self.mask_size is used. 
            output_attentions=None, 
            output_hidden_states=None, 
            sum_fraction=0.5,
            segment_size=1022, 
            mode='train', 
            prof=False,
            pos_mask=None,
            **kwargs
        ):

        mask_size = self.rmt_config.get('mask_size') if mask_size is None else mask_size

        memory_state = None
        prepend_state = None
        segment = None
        seg_iter = SegmentIterator(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        seg_num = 0

        cell_outputs = []
        n_cell_out = self.rmt_config.get('n_cell_out')
        memory_seq = None

        total_hist = []

        while True:

            prepend_state = segment['input_ids'][:,-self.num_prepend:].cuda() if segment is not None and self.num_prepend > 0 else None
            segment = seg_iter.next(segment_size)
            if segment is None:
                break

            if self.cross_attn is not None:
                seg = copy.deepcopy(segment)
                seg['input_ids'] = seg['input_ids'][:,:int(round(segment_size * sum_fraction))]
                seg['attention_mask'] = seg['attention_mask'][:,:int(round(segment_size * sum_fraction))]
                # summary current segment to get summrization
                summary_prompt = self.summary_cell(input_ids.shape[0]) if self.num_mem_embed > 0 else None
                _, summary_state = self.memory_cell(**seg, pre_memory_state=summary_prompt)
                
                # attend to long-term memory
                memory_prompt, hist = self.cross_attn(
                    memory_seq,
                    summary_state,
                    mode,
                    seg_num if seg_num < self.mem_window_size else self.mem_window_size,
                )
                if hist is not None:
                    total_hist.extend(hist)

            # process the whole segement and get new long-term memory
            if prof:
                with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=True) as prof_m:
                    with record_function("model_inference"):
                        cell_out, memory_state = self.memory_cell(
                            **segment,
                            pre_memory_state=memory_prompt,
                            prepend_state=prepend_state,
                            output_hidden_states=True,
                        )
                
                with open('model_profile_dump.txt', 'w') as file:
                    file.write(prof_m.key_averages().table(sort_by="cuda_time_total"))
                
                prof_m.export_chrome_trace("model_trace.json")
                exit(0)
            else:
                cell_out, memory_state = self.memory_cell(
                    **segment,
                    pre_memory_state=memory_prompt,
                    prepend_state=prepend_state,
                    output_hidden_states=True,
                )

            # if prof:
            #     torch.cuda.synchronize()
            #     self.logger.info('segment ' + str(seg_num) + ' elapsed time: ' + str(start.elapsed_time(end)) + ' ms')

            cell_outputs.append(cell_out)
            if len(cell_outputs) > n_cell_out:
                cell_outputs.pop(0)
            
            if self.cross_attn is not None:
                if memory_seq is None:
                    memory_seq = memory_state.cpu()
                else:
                    memory_seq = torch.cat([memory_seq, memory_state.cpu()], dim=1)
                    if memory_seq.shape[1] > self.mem_window_size:
                        memory_seq = memory_seq[:,-self.mem_window_size:,:]

            if memory_state is not None:
                self.manage_gradients(memory_state, seg_num)

            seg_num+=1
        
        out, metrics = self.process_outputs(cell_outputs, labels=labels, 
                                   labels_mask=labels_mask,
                                   output_attentions=output_attentions, 
                                   output_hidden_states=output_hidden_states,
                                   mask_size=mask_size)
        return out, total_hist, metrics
    
    def generate(self, input_ids, attention_mask, segment_size, mem_seq=None, sum_fraction=0.5, **generate_kwargs):
        """Generate tokens by processing segments sequentially and rolling memory forward."""
        memory_prompt = None
        memory_state = None
        prepend_state = None
        prev_input_ids = None
        segmented = self.segment(segment_size, input_ids=input_ids, attention_mask=attention_mask)
        memory_seq = mem_seq

        # Walk through all but the last segment to build up the memory sequence.
        for seg_num, segment in enumerate(segmented[:-1]):
            if prev_input_ids is not None and self.num_prepend > 0:
                prepend_state = prev_input_ids[:, -self.num_prepend:].cuda()

            for k, v in segment.items():
                segment[k] = v.cuda()

            if self.cross_attn is not None:
                seg = copy.deepcopy(segment)
                cut = int(round(segment_size * sum_fraction))
                seg['input_ids'] = seg['input_ids'][:, :cut]
                seg['attention_mask'] = seg['attention_mask'][:, :cut]
                summary_prompt = self.summary_cell(segment['input_ids'].shape[0]) if self.num_mem_embed > 0 else None
                _, summary_state = self.memory_cell(**seg, pre_memory_state=summary_prompt)
                memory_prompt, _ = self.cross_attn(
                    memory_seq,
                    summary_state,
                    'generate',
                    seg_num if seg_num < self.mem_window_size else self.mem_window_size,
                )
            else:
                memory_prompt = None

            with torch.no_grad():
                _, memory_state = self.memory_cell(
                    **segment,
                    pre_memory_state=memory_prompt,
                    prepend_state=prepend_state,
                    output_hidden_states=True,
                )

            if self.cross_attn is not None:
                if memory_seq is None:
                    memory_seq = memory_state.cpu()
                else:
                    memory_seq = torch.cat([memory_seq, memory_state.cpu()], dim=1)
                    if memory_seq.shape[1] > self.mem_window_size:
                        memory_seq = memory_seq[:, -self.mem_window_size:, :]

            prev_input_ids = segment['input_ids'].cpu()
            for k, v in segment.items():
                segment[k] = v.cpu()

        # Prepare the final segment for generation.
        final_segment = segmented[-1]
        if prev_input_ids is not None and self.num_prepend > 0:
            prepend_state = prev_input_ids[:, -self.num_prepend:].cuda()
        for k, v in final_segment.items():
            final_segment[k] = v.cuda()

        seg_num = len(segmented) - 1
        output_len = generate_kwargs.get('max_new_tokens', None)

        if self.cross_attn is not None:
            seg = copy.deepcopy(final_segment)
            seg['input_ids'] = seg['input_ids'][:, :(segment_size // 2)]
            seg['attention_mask'] = seg['attention_mask'][:, :(segment_size // 2)]
            summary_prompt = self.summary_cell(final_segment['input_ids'].shape[0]) if self.num_mem_embed > 0 else None
            _, summary_state = self.memory_cell(**seg, pre_memory_state=summary_prompt)
            memory_prompt, _ = self.cross_attn(
                memory_seq,
                summary_state,
                'generate',
                seg_num if seg_num < self.mem_window_size else self.mem_window_size,
            )
        else:
            memory_prompt = None

        if output_len is None or output_len <= segment_size:
            out = self.memory_cell.generate(**final_segment, pre_memory_state=memory_prompt, prepend_state=prepend_state, **generate_kwargs)
        else:
            generate_kwargs = dict(generate_kwargs)
            generate_kwargs['max_new_tokens'] = segment_size
            final_out = []
            for _ in range(math.ceil(output_len / segment_size)):
                out = self.memory_cell.generate(**final_segment, pre_memory_state=memory_prompt, prepend_state=prepend_state, **generate_kwargs)
                final_out.append(out)
                # if out is shorter than max_new_tokens, then stop, otherwise we will continue generation
                if out.shape[1] < segment_size:
                    return torch.cat(final_out, dim=1)
                else:
                    if self.cross_attn is not None:
                        summary_prompt = self.summary_cell(out.shape[0]) if self.num_mem_embed > 0 else None
                        seg = {
                            'input_ids': out[:, :(segment_size // 2)],
                            'attention_mask': torch.ones_like(out[:, :(segment_size // 2)]),
                        }
                        _, summary_state = self.memory_cell(**seg, pre_memory_state=summary_prompt)
                        memory_prompt, _ = self.cross_attn(
                            memory_seq,
                            summary_state,
                            'generate',
                            seg_num if seg_num < self.mem_window_size else self.mem_window_size,
                        )

                    with torch.no_grad():
                        _, memory_state = self.memory_cell(
                            input_ids=out,
                            attention_mask=torch.ones_like(out),
                            pre_memory_state=memory_prompt,
                            prepend_state=prepend_state,
                            output_hidden_states=True,
                        )

                    if self.cross_attn is not None:
                        if memory_seq is None:
                            memory_seq = memory_state.cpu()
                        else:
                            memory_seq = torch.cat([memory_seq, memory_state.cpu()], dim=1)
                            if memory_seq.shape[1] > self.mem_window_size:
                                memory_seq = memory_seq[:, -self.mem_window_size:, :]

                    if self.num_prepend > 0:
                        prepend_state = out[:, -self.num_prepend:].cuda()
                    seg_num += 1

            return torch.cat(final_out, dim=1)

        return out

    def segment(self, segment_size, **kwargs):
        segments = []
        for k, tensor in kwargs.items():
            if tensor is None:
                continue
            for s, start in enumerate(range(0, tensor.shape[1], segment_size)):
                if s == len(segments):
                    segments.append({})
                segments[s][k] = tensor[:, start:start + segment_size]

        return segments

    def process_outputs(self, cell_outputs, **kwargs):
        out = CausalLMOutputWithCrossAttentions()
        full_logits = torch.cat([o.logits for o in cell_outputs], dim=1)
        full_hidden_states = tuple([torch.cat(layer_hs, dim=1) for layer_hs in zip(*[o.hidden_states for o in cell_outputs])])
        
        mask_size = kwargs.get('mask_size')
        metrics = {
            'loss': None,
            'ppl': None,
            'precision': None,
            'recall': None,
            'f1': None,
            'accuracy': None
        }

        labels = kwargs.get('labels')
        if labels.shape[1] <= mask_size:
            mask_size = labels.shape[1]-1
        
        if labels is not None:
            shift_labels = labels[..., -mask_size:].contiguous()
            shift_logits = full_logits[..., -(mask_size+1):-1, :].contiguous()
            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            
            loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            gen_loss = loss_fct(flat_logits.cuda(), flat_labels.cuda())
            out['loss'] = gen_loss
            metrics['loss'] = out['loss'].detach().item()
            metrics['ppl'] = torch.exp(gen_loss.detach()).item()

            # filter ignore_index before computing metrics
            flat_labels_cpu = flat_labels.detach().cpu()
            mask = flat_labels_cpu != IGNORE_INDEX
            labels_valid = flat_labels_cpu[mask]

            predictions = flat_logits.argmax(dim=-1).detach().cpu()
            preds_valid = predictions[mask]

            # add zero_division=0 to avoild warning
            if labels_valid.numel() > 0:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels_valid,
                    preds_valid,
                    average='weighted',
                    zero_division=0
                )
                accuracy = accuracy_score(labels_valid, preds_valid)
                metrics['precision'] = float(precision)
                metrics['recall'] = float(recall)
                metrics['f1'] = float(f1)
                metrics['accuracy'] = float(accuracy)

        else:
            zero = torch.tensor(0.0, device=full_logits.device)
            out['loss'] = zero
            metrics['loss'] = 0.0
            metrics['ppl'] = None
            metrics['f1'] = None
            metrics['accuracy'] = None

        out['logits'] = full_logits
        segment_keys = ['loss', 'logits']
        if kwargs.get('output_attentions'):
            segment_keys.append('attentions')
        if kwargs.get('output_hidden_states'):
            segment_keys.append('hidden_states')
            out['hidden_states'] = full_hidden_states

        return out, metrics
        
    def manage_gradients(self, memory_state, seg_num):
        if seg_num == 0:
            return True
        memory_state = memory_state.detach()
        return False



class Memory_Only_RecurrentWrapper(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, base_model, num_mem_embed, num_prepend, mem_hidden_dim=4096, mem_window_size=64, rmt_only=False, baseline_only=False, **rmt_kwargs):
        super().__init__()

        if isinstance(base_model.config, OPTConfig):
            mem_emb_dim = base_model.config.word_embed_proj_dim
        else:
            mem_emb_dim = base_model.config.hidden_size

        self.num_mem_embed = num_mem_embed
        self.num_prepend = num_prepend

        self.memory_cell = MemoryCell(base_model, num_mem_embed=num_mem_embed, num_prepend=num_prepend, same_pre_suf_memory=False)
        if rmt_only or baseline_only:
            self.cross_attn = None
        else:
            self.cross_attn = CrossAttentionMemory(mem_emb_dim, mem_hidden_dim)

            
        self.rmt_config = rmt_kwargs
        self.mem_window_size = mem_window_size
        self.logger = get_logger('')
        
        self.rouge = evaluate.load('rouge')
        self.f1 = evaluate.load("f1")

    def forward(self, 
            input_ids, 
            labels=None, 
            labels_mask=None, 
            inputs_embeds=None, 
            attention_mask=None, 
            mask_size=None,  # Size of the attention mask used to compute the loss, it should be the length of the labels. If it's None, then self.mask_size is used. 
            output_attentions=None, 
            output_hidden_states=None, 
            sum_fraction=0.5,
            segment_size=1022, 
            mode='train', 
            prof=False,
            pos_mask=None,
            **kwargs
        ):

        mask_size = self.rmt_config.get('mask_size') if mask_size is None else mask_size

        memory_state = None
        prepend_state = None
        segment = None
        seg_iter = SegmentIterator(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        seg_num = 0

        cell_outputs = []
        n_cell_out = self.rmt_config.get('n_cell_out')
        memory_seq = self.memory_cell.get_init_suf_memory_states(input_ids.shape[0]).cpu()

        total_hist = []

        while True:

            prepend_state = segment['input_ids'][:,-self.num_prepend:].cuda() if segment is not None and self.num_prepend > 0 else None
            segment = seg_iter.next(segment_size)
            if segment is None:
                break
            
            cell_out, memory_prompt = self.memory_cell(
                **segment,
                pre_memory_state=memory_state,
                prepend_state=prepend_state,
                output_hidden_states=True,
            )

            cell_outputs.append(cell_out)
            if len(cell_outputs) > n_cell_out:
                cell_outputs.pop(0)
            
            if self.cross_attn is not None:
                # attend to long-term memory
                memory_state, hist = self.cross_attn(
                    memory_seq,
                    memory_prompt,
                    mode,
                    seg_num if seg_num < self.mem_window_size else self.mem_window_size,
                )
                if hist is not None:
                    total_hist.extend(hist)

                if memory_seq is None:
                    memory_seq = memory_state.cpu()
                else:
                    memory_seq = torch.cat([memory_seq, memory_state.cpu()], dim=1)
                    if memory_seq.shape[1] > self.mem_window_size:
                        memory_seq = memory_seq[:,-self.mem_window_size:,:]

            if memory_state is not None:
                self.manage_gradients(memory_state, seg_num)

            seg_num+=1
        
        out, metrics = self.process_outputs(cell_outputs, labels=labels, 
                                   labels_mask=labels_mask,
                                   output_attentions=output_attentions, 
                                   output_hidden_states=output_hidden_states,
                                   mask_size=mask_size)
        return out, total_hist, metrics
    
    def generate(self, input_ids, attention_mask, segment_size, mem_seq=None, sum_fraction=0.5, **generate_kwargs):
        """Generate tokens by processing segments sequentially and rolling memory forward."""
        memory_state = None
        prepend_state = None
        prev_input_ids = None
        segmented = self.segment(segment_size, input_ids=input_ids, attention_mask=attention_mask)
        memory_seq = mem_seq

        # Walk through all but the last segment to build up the memory sequence.
        for seg_num, segment in enumerate(segmented[:-1]):
            if prev_input_ids is not None and self.num_prepend > 0:
                prepend_state = prev_input_ids[:, -self.num_prepend:].cuda()

            for k, v in segment.items():
                segment[k] = v.cuda()

            with torch.no_grad():
                _, memory_prompt = self.memory_cell(
                    **segment,
                    pre_memory_state=memory_state,
                    prepend_state=prepend_state,
                    output_hidden_states=True,
                )

            if self.cross_attn is not None:
                memory_state, _ = self.cross_attn(
                    memory_seq,
                    memory_prompt,
                    'generate',
                    seg_num if seg_num < self.mem_window_size else self.mem_window_size,
                )
            
                if memory_seq is None:
                    memory_seq = memory_state.cpu()
                else:
                    memory_seq = torch.cat([memory_seq, memory_state.cpu()], dim=1)
                    if memory_seq.shape[1] > self.mem_window_size:
                        memory_seq = memory_seq[:, -self.mem_window_size:, :]

            prev_input_ids = segment['input_ids'].cpu()
            for k, v in segment.items():
                segment[k] = v.cpu()

        # Prepare the final segment for generation.
        final_segment = segmented[-1]
        current_segment_size = final_segment['input_ids'].shape[1]
        
        if prev_input_ids is not None and self.num_prepend > 0:
            prepend_state = prev_input_ids[:, -self.num_prepend:].cuda()
        for k, v in final_segment.items():
            final_segment[k] = v.cuda()

        seg_num = len(segmented) - 1
        output_len = generate_kwargs.get('max_new_tokens', None)

        if output_len is None or output_len <= segment_size - current_segment_size:
            out = self.memory_cell.generate(**final_segment, pre_memory_state=memory_state, prepend_state=prepend_state, **generate_kwargs)
        else:
            generate_kwargs = dict(generate_kwargs)
            generate_len = 0
            final_out = []
            while generate_len < output_len:
                if(current_segment_size < segment_size):
                    generate_kwargs['max_new_tokens'] = segment_size - current_segment_size
                    out = self.memory_cell.generate(
                        **final_segment, 
                        pre_memory_state=memory_state, 
                        prepend_state=prepend_state, 
                        **generate_kwargs
                    )
                    final_out.append(out)
                    # if out is shorter than max_new_tokens, then stop, otherwise we will continue generation
                    if out.shape[1] < segment_size:
                        return torch.cat(final_out, dim=1)
                
                with torch.no_grad():
                    _, memory_prompt = self.memory_cell(
                        input_ids=out,
                        attention_mask=torch.ones_like(out),
                        pre_memory_state=memory_state,
                        prepend_state=prepend_state,
                        output_hidden_states=True,
                    )

                if self.cross_attn is not None:
                    memory_state, _ = self.cross_attn(
                        memory_seq,
                        memory_prompt,
                        'generate',
                        seg_num if seg_num < self.mem_window_size else self.mem_window_size,
                    )

                    if memory_seq is None:
                        memory_seq = memory_state.cpu()
                    else:
                        memory_seq = torch.cat([memory_seq, memory_state.cpu()], dim=1)
                        if memory_seq.shape[1] > self.mem_window_size:
                            memory_seq = memory_seq[:, -self.mem_window_size:, :]

                if self.num_prepend > 0:
                    prepend_state = out[:, -self.num_prepend:].cuda()
                
                seg_num += 1
                generate_len += out.shape[1]
                final_segment = {}
                current_segment_size = 0
        
            return torch.cat(final_out, dim=1)

        return out

    def segment(self, segment_size, **kwargs):
        segments = []
        for k, tensor in kwargs.items():
            if tensor is None:
                continue
            for s, start in enumerate(range(0, tensor.shape[1], segment_size)):
                if s == len(segments):
                    segments.append({})
                segments[s][k] = tensor[:, start:start + segment_size]

        return segments

    def process_outputs(self, cell_outputs, **kwargs):
        out = CausalLMOutputWithCrossAttentions()
        full_logits = torch.cat([o.logits for o in cell_outputs], dim=1)
        full_hidden_states = tuple([torch.cat(layer_hs, dim=1) for layer_hs in zip(*[o.hidden_states for o in cell_outputs])])
        
        mask_size = kwargs.get('mask_size')
        metrics = {
            'loss': None,
            'ppl': None,
            'precision': None,
            'recall': None,
            'f1': None,
            'accuracy': None
        }

        labels = kwargs.get('labels')
        if labels.shape[1] <= mask_size:
            mask_size = labels.shape[1]-1
        
        if labels is not None:
            shift_labels = labels[..., -mask_size:].contiguous()
            shift_logits = full_logits[..., -(mask_size+1):-1, :].contiguous()
            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            
            loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            gen_loss = loss_fct(flat_logits.cuda(), flat_labels.cuda())
            out['loss'] = gen_loss
            metrics['loss'] = out['loss'].detach().item()
            metrics['ppl'] = torch.exp(gen_loss.detach()).item()

            # filter ignore_index before computing metrics
            flat_labels_cpu = flat_labels.detach().cpu()
            mask = flat_labels_cpu != IGNORE_INDEX
            labels_valid = flat_labels_cpu[mask]

            predictions = flat_logits.argmax(dim=-1).detach().cpu()
            preds_valid = predictions[mask]

            # add zero_division=0 to avoild warning
            if labels_valid.numel() > 0:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels_valid,
                    preds_valid,
                    average='weighted',
                    zero_division=0
                )
                accuracy = accuracy_score(labels_valid, preds_valid)
                metrics['precision'] = float(precision)
                metrics['recall'] = float(recall)
                metrics['f1'] = float(f1)
                metrics['accuracy'] = float(accuracy)

        else:
            zero = torch.tensor(0.0, device=full_logits.device)
            out['loss'] = zero
            metrics['loss'] = 0.0
            metrics['ppl'] = None
            metrics['f1'] = None
            metrics['accuracy'] = None

        out['logits'] = full_logits
        segment_keys = ['loss', 'logits']
        if kwargs.get('output_attentions'):
            segment_keys.append('attentions')
        if kwargs.get('output_hidden_states'):
            segment_keys.append('hidden_states')
            out['hidden_states'] = full_hidden_states

        return out, metrics
        
    def manage_gradients(self, memory_state, seg_num):
        if seg_num == 0:
            return True
        memory_state = memory_state.detach()
        return False
