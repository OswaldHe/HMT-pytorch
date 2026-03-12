import math
import torch
import copy
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import OPTConfig
from .long_mem_cross_attn_vanilla import CrossAttentionMemory
from .memory_cell import SummaryCell, MemoryCell
from .segment_iter import SegmentIterator, Bert_SegmentIterator
from accelerate.logging import get_logger
from torch.profiler import profile, record_function, ProfilerActivity
import random
import evaluate
from huggingface_hub import PyTorchModelHubMixin

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

IGNORE_INDEX = -100

class Summary_Memory_RecurrentWrapper(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, base_model, num_mem_embed, num_prepend, mem_hidden_dim=4096, mem_window_size=64, rmt_only=False, baseline_only=False, **hmt_kwargs):
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
            
        self.hmt_config = hmt_kwargs
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
            segment_size=1024, 
            mode='train', 
            pos_mask=None,
            **kwargs
        ):

        mask_size = self.hmt_config.get('mask_size') if mask_size is None else mask_size

        memory_state = None
        prepend_state = None
        segment = None
        seg_iter = SegmentIterator(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        seg_num = 0

        cell_outputs = []
        n_cell_out = self.hmt_config.get('n_cell_out')
        memory_seq = None

        total_hist = []

        while True:

            prepend_state = segment['input_ids'][:,-self.num_prepend:].cuda() if segment is not None and self.num_prepend > 0 else None
            segment = seg_iter.next(segment_size)
            if segment is None:
                break
            current_segment_size = segment_size

            memory_prompt = None
            if self.cross_attn is not None:
                seg = copy.deepcopy(segment)
                cut_size = int(round(current_segment_size * sum_fraction))
                cut_size = max(1, min(cut_size, seg['input_ids'].shape[1]))
                seg['input_ids'] = seg['input_ids'][:, :cut_size]
                seg['attention_mask'] = seg['attention_mask'][:, :cut_size]
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

            cell_out, memory_state = self.memory_cell(
                **segment,
                pre_memory_state=memory_prompt,
                prepend_state=prepend_state,
                output_hidden_states=True,
            )

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

            seg_num+=1
        
        out, metrics = self.process_outputs(cell_outputs, labels=labels, 
                                   labels_mask=labels_mask,
                                   output_attentions=output_attentions, 
                                   output_hidden_states=output_hidden_states,
                                   mask_size=mask_size)
        return out, total_hist, metrics
    
    def generate(self, input_ids, attention_mask, segment_size=1024, mem_seq=None, sum_fraction=0.5, **generate_kwargs):
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
            cut = int(round(segment_size * sum_fraction))
            seg['input_ids'] = seg['input_ids'][:, :cut]
            seg['attention_mask'] = seg['attention_mask'][:, :cut]
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
                            'input_ids': out[:, :cut],
                            'attention_mask': torch.ones_like(out[:, :cut]),
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
        full_logits = torch.cat([o.logits for o in cell_outputs], dim=1)
        full_hidden_states = tuple([torch.cat(layer_hs, dim=1) for layer_hs in zip(*[o.hidden_states for o in cell_outputs])])
        return self._build_output_and_metrics(
            full_logits=full_logits,
            full_hidden_states=full_hidden_states,
            full_valid_mask=None,
            **kwargs,
        )

    def _build_output_and_metrics(self, full_logits, full_hidden_states, full_valid_mask=None, **kwargs):
        out = CausalLMOutputWithCrossAttentions()
        labels = kwargs.get('labels')
        mask_size = kwargs.get('mask_size')
        metrics = {
            'loss': None,
            'ppl': None,
            'precision': None,
            'recall': None,
            'f1': None,
            'accuracy': None
        }
        
        if labels is not None:
            labels_len = labels.shape[1]
            logits_len = full_logits.shape[1]
            if mask_size is None:
                mask_size = labels_len
            effective_mask_size = min(mask_size, labels_len, logits_len - 1)

            if effective_mask_size <= 0:
                gen_loss = full_logits.new_zeros(())
                out['loss'] = gen_loss
                metrics['loss'] = 0.0
                metrics['ppl'] = None
                metrics['f1'] = None
                metrics['accuracy'] = None
                out['logits'] = full_logits
                if kwargs.get('output_hidden_states'):
                    out['hidden_states'] = full_hidden_states
                return out, metrics

            shift_labels = labels[..., -effective_mask_size:].contiguous()
            shift_logits = full_logits[..., -(effective_mask_size+1):-1, :].contiguous()
            if full_valid_mask is not None:
                shift_valid_mask = full_valid_mask[..., -(effective_mask_size+1):-1].to(shift_labels.device).contiguous()
                shift_labels = shift_labels.masked_fill(~shift_valid_mask, IGNORE_INDEX)
            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            
            if (flat_labels != IGNORE_INDEX).any():
                loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
                gen_loss = loss_fct(flat_logits.to(full_logits.device), flat_labels.to(full_logits.device))
                metrics['ppl'] = torch.exp(gen_loss.detach()).item()
            else:
                gen_loss = flat_logits.new_zeros(())
                metrics['ppl'] = None
            out['loss'] = gen_loss
            metrics['loss'] = out['loss'].detach().item()

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
        if kwargs.get('output_hidden_states'):
            out['hidden_states'] = full_hidden_states

        return out, metrics

    def _apply_token_mask_and_pad(self, tensor, token_mask, target_len):
        target_len = max(int(target_len), 1)
        if tensor.dim() < 2:
            return tensor
        batch_size = tensor.shape[0]
        output_shape = (batch_size, target_len) + tuple(tensor.shape[2:])
        padded = tensor.new_zeros(output_shape)
        for bidx in range(batch_size):
            valid = tensor[bidx][token_mask[bidx]]
            copy_len = min(valid.shape[0], target_len)
            if copy_len > 0:
                padded[bidx, :copy_len] = valid[:copy_len]
        return padded


class Summary_Memory_RecurrentWrapper_Dynamic(Summary_Memory_RecurrentWrapper):
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
            segment_size=1024, 
            mode='train', 
            pos_mask=None,
            **kwargs
        ):

        mask_size = self.hmt_config.get('mask_size') if mask_size is None else mask_size
        if input_ids is None or attention_mask is None:
            raise ValueError("Summary_Memory_RecurrentWrapper_Dynamic requires input_ids and attention_mask")
        if self.hmt_config.get('dynamic_seg_checkpoint') is None:
            raise ValueError("dynamic_seg_checkpoint is required for Summary_Memory_RecurrentWrapper_Dynamic")
        if self.hmt_config.get('lm_tokenizer') is None:
            raise ValueError("lm_tokenizer is required for Summary_Memory_RecurrentWrapper_Dynamic")

        memory_state = None
        prepend_state = None
        segment = None
        seg_iter = Bert_SegmentIterator(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            segment_length=segment_size,
            lm_tokenizer=self.hmt_config.get('lm_tokenizer'),
            seg_checkpoint=self.hmt_config.get('dynamic_seg_checkpoint'),
            debug=bool(self.hmt_config.get('dynamic_seg_debug', False)),
        )
        seg_num = 0

        cell_outputs = []
        n_cell_out = self.hmt_config.get('n_cell_out')
        memory_seq = None
        token_masks = []

        total_hist = []

        while True:

            prepend_state = segment['input_ids'][:,-self.num_prepend:].cuda() if segment is not None and self.num_prepend > 0 else None
            iter_out = seg_iter.next(segment_size)
            if iter_out is None:
                break
            segment, real_segment_len, real_token_mask = iter_out
            token_masks.append(real_token_mask)
            current_segment_size = real_segment_len

            memory_prompt = None
            if self.cross_attn is not None:
                seg = copy.deepcopy(segment)
                cut_size = int(round(current_segment_size * sum_fraction))
                cut_size = max(1, min(cut_size, seg['input_ids'].shape[1]))
                seg['input_ids'] = seg['input_ids'][:, :cut_size]
                seg['attention_mask'] = seg['attention_mask'][:, :cut_size]
                summary_prompt = self.summary_cell(input_ids.shape[0]) if self.num_mem_embed > 0 else None
                _, summary_state = self.memory_cell(**seg, pre_memory_state=summary_prompt)
                memory_prompt, hist = self.cross_attn(
                    memory_seq,
                    summary_state,
                    mode,
                    seg_num if seg_num < self.mem_window_size else self.mem_window_size,
                )
                if hist is not None:
                    total_hist.extend(hist)

            cell_out, memory_state = self.memory_cell(
                **segment,
                pre_memory_state=memory_prompt,
                prepend_state=prepend_state,
                output_hidden_states=True,
            )

            cell_outputs.append(cell_out)
            if len(cell_outputs) > n_cell_out:
                cell_outputs.pop(0)
                if len(token_masks) > 0:
                    token_masks.pop(0)
            
            if self.cross_attn is not None:
                if memory_seq is None:
                    memory_seq = memory_state.cpu()
                else:
                    memory_seq = torch.cat([memory_seq, memory_state.cpu()], dim=1)
                    if memory_seq.shape[1] > self.mem_window_size:
                        memory_seq = memory_seq[:,-self.mem_window_size:,:]

            seg_num += 1
        
        out, metrics = self.process_outputs(cell_outputs, labels=labels, 
                                   labels_mask=labels_mask,
                                   output_attentions=output_attentions, 
                                   output_hidden_states=output_hidden_states,
                                   mask_size=mask_size,
                                   token_masks=token_masks)
        return out, total_hist, metrics

    def process_outputs(self, cell_outputs, **kwargs):
        full_logits = torch.cat([o.logits for o in cell_outputs], dim=1)
        full_hidden_states = tuple([torch.cat(layer_hs, dim=1) for layer_hs in zip(*[o.hidden_states for o in cell_outputs])])
        token_masks = kwargs.get('token_masks')
        full_valid_mask = None
        if token_masks:
            full_token_mask = torch.cat(token_masks, dim=1).to(full_logits.device).bool()
            max_valid_len = max(int(full_token_mask.sum(dim=1).max().item()), 1)
            full_logits = self._apply_token_mask_and_pad(full_logits, full_token_mask, max_valid_len)
            full_hidden_states = tuple(
                self._apply_token_mask_and_pad(layer_hs, full_token_mask, max_valid_len)
                for layer_hs in full_hidden_states
            )
            full_valid_mask = torch.zeros(
                full_logits.shape[:2],
                dtype=torch.bool,
                device=full_logits.device,
            )
            valid_counts = full_token_mask.sum(dim=1)
            for bidx in range(full_valid_mask.shape[0]):
                keep_len = min(int(valid_counts[bidx].item()), full_valid_mask.shape[1])
                if keep_len > 0:
                    full_valid_mask[bidx, :keep_len] = True

        return self._build_output_and_metrics(
            full_logits=full_logits,
            full_hidden_states=full_hidden_states,
            full_valid_mask=full_valid_mask,
            **kwargs,
        )



class Memory_Only_RecurrentWrapper(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(
        self, 
        base_model, 
        num_mem_embed, 
        num_prepend, 
        mem_hidden_dim=4096, 
        mem_window_size=64, 
        rmt_only=False, 
        baseline_only=False,
        mem_mlp=False,
        mem_mlp_hidden_dim=None, 
        **hmt_kwargs
    ):
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

        self.mem_window_size = mem_window_size
        
        self.hmt_config = hmt_kwargs
        self.mem_mlp = None
        if mem_mlp:
            print("Using mem_mlp to process memory_state before adding to memory_seq.")
            if mem_mlp_hidden_dim is None:
                mem_mlp_hidden_dim = mem_emb_dim
            self.mem_mlp = torch.nn.Sequential(
                torch.nn.Linear(mem_emb_dim, mem_mlp_hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(mem_mlp_hidden_dim, mem_emb_dim),
            )
        
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
            segment_size=1024, 
            mode='train', 
            pos_mask=None,
            **kwargs
        ):

        mask_size = self.hmt_config.get('mask_size') if mask_size is None else mask_size

        memory_state = None
        prepend_state = None
        memory_seq = None
        segment = None
        seg_iter = SegmentIterator(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        seg_num = 0

        cell_outputs = []
        n_cell_out = self.hmt_config.get('n_cell_out')
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
                if memory_seq is None:
                    memory_seq = self.memory_cell.get_init_pre_memory_states(input_ids.shape[0]).cpu()

                memory_state, hist = self.cross_attn(
                    memory_seq,
                    memory_prompt,
                    mode,
                    seg_num if seg_num < self.mem_window_size else self.mem_window_size,
                )
                

                memory_seq = torch.cat([memory_seq, memory_state.cpu()], dim=1)
                if memory_seq.shape[1] > self.mem_window_size:
                    memory_seq = memory_seq[:,-self.mem_window_size:,:]

                if hist is not None:
                    total_hist.extend(hist)
                
                # to-do: if args.mem_mlp, add a mlp layer here to process memory_state
                if self.mem_mlp is not None:
                    memory_state = self.mem_mlp(memory_state)

            seg_num+=1
        
        out, metrics = self.process_outputs(cell_outputs, labels=labels, 
                                   labels_mask=labels_mask,
                                   output_attentions=output_attentions, 
                                   output_hidden_states=output_hidden_states,
                                   mask_size=mask_size)
        return out, total_hist, metrics
    
    def generate(self, input_ids, attention_mask, segment_size=1024, mem_seq=None, **generate_kwargs):
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
                if memory_seq is None:
                    memory_seq = self.memory_cell.get_init_pre_memory_states(input_ids.shape[0]).cpu()
    
                memory_state, _ = self.cross_attn(
                    memory_seq,
                    memory_prompt,
                    'generate',
                    seg_num if seg_num < self.mem_window_size else self.mem_window_size,
                )
                
                memory_seq = torch.cat([memory_seq, memory_state.cpu()], dim=1)
                if memory_seq.shape[1] > self.mem_window_size:
                    memory_seq = memory_seq[:, -self.mem_window_size:, :]

                if self.mem_mlp is not None:
                    memory_state = self.mem_mlp(memory_state)

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
                    if memory_seq is None:
                        memory_seq = self.memory_cell.get_init_pre_memory_states(input_ids.shape[0]).cpu()

                    memory_state, _ = self.cross_attn(
                        memory_seq,
                        memory_prompt,
                        'generate',
                        seg_num if seg_num < self.mem_window_size else self.mem_window_size,
                    )
                    if self.mem_mlp is not None:
                        memory_state = self.mem_mlp(memory_state)

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
