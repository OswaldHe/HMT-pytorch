import torch
from transformers import OPTConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class SummaryCell(torch.nn.Module):
    def __init__(self, base_model, num_mem_embed):
        super().__init__()
        self.num_mem_embed = num_mem_embed
        embeddings = base_model.get_input_embeddings()
        if num_mem_embed > 0:
            if isinstance(base_model.config, OPTConfig):
                mem_emb_dim = getattr(base_model.config, "n_embd", base_model.config.word_embed_proj_dim)
            else:
                mem_emb_dim = getattr(base_model.config, "n_embd", base_model.config.hidden_size)
            # sum_prompt_embeds serve as prompt tokens for summarizing current segment's topic
            sum_prompt_emb_weights = torch.randn((num_mem_embed, mem_emb_dim)) * embeddings.weight.data.std()
            self.register_parameter("sum_prompt_embeds", torch.nn.Parameter(sum_prompt_emb_weights, requires_grad=True))
        else:
            self.register_parameter("sum_prompt_embeds", None)

    def forward(self, batch_size):
        if self.sum_prompt_embeds is None:
            return None
        return self.sum_prompt_embeds.repeat(batch_size, 1, 1)


class MemoryCell(torch.nn.Module):
    def __init__(self, base_model, num_mem_embed, num_prepend, same_pre_suf_memory=True):
        super().__init__()
        self.model = base_model
        self.n_prepend = num_prepend
        self.num_mem_embed = num_mem_embed
        self.same_pre_suf_memory = same_pre_suf_memory

        embeddings = self.model.get_input_embeddings()
        if num_mem_embed > 0:
            if isinstance(self.model.config, OPTConfig):
                memory_dim = getattr(self.model.config, "n_embd", self.model.config.word_embed_proj_dim)
            else:
                memory_dim = getattr(self.model.config, "n_embd", self.model.config.hidden_size)
            
            pre_memory_weights = torch.randn((num_mem_embed, memory_dim)) * embeddings.weight.data.std()
            self.register_parameter("init_pre_memory_state", torch.nn.Parameter(pre_memory_weights, requires_grad=True))
            
            if same_pre_suf_memory:
                self.register_parameter("init_suf_memory_state", None)
            else:
                suf_memory_weights = torch.randn((num_mem_embed, memory_dim)) * embeddings.weight.data.std()
                self.register_parameter("init_suf_memory_state", torch.nn.Parameter(suf_memory_weights, requires_grad=True))

    def get_init_pre_memory_states(self, batch_size):
        if self.num_mem_embed > 0:
            return self.init_pre_memory_state.repeat(batch_size, 1, 1)
        else:
            return None
    
    def get_init_suf_memory_states(self, batch_size):
        if self.num_mem_embed > 0:
            if self.same_pre_suf_memory:
                return self.init_pre_memory_state.repeat(batch_size, 1, 1)
            else:
                return self.init_suf_memory_state.repeat(batch_size, 1, 1)
        else:
            return None

    def forward(self, input_ids, pre_memory_state=None, prepend_state=None, suf_memory_state=None, **kwargs):
        input_ids = input_ids.cuda()
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                kwargs[k] = v.cuda()

        if pre_memory_state is None:
            pre_memory_state = self.get_init_pre_memory_states(input_ids.shape[0])
        if suf_memory_state is None:
            suf_memory_state = self.get_init_suf_memory_states(input_ids.shape[0])
        
        seg_kwargs = self.process_input(input_ids, pre_memory_state, prepend_state, suf_memory_state, **kwargs)
        out = self.model(**seg_kwargs)
        n_prepend = self.n_prepend
        out, new_memory_state = self.process_output(out, 0 if prepend_state is None else n_prepend, **kwargs)
        input_ids = input_ids.cpu()
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                kwargs[k] = v.cpu()
        return out, new_memory_state

    def generate(self, input_ids, pre_memory_state=None, prepend_state=None, attention_mask=None, **generate_kwargs):
        if self.num_mem_embed > 0:
            if pre_memory_state is None:
                pre_memory_state = self.init_pre_memory_state.repeat(input_ids.shape[0], 1, 1)
        seg_kwargs = self.process_input(input_ids, pre_memory_state, prepend_state, generate=True, attention_mask=attention_mask)
        out = self.model.generate(
            inputs_embeds=seg_kwargs["inputs_embeds"],
            attention_mask=seg_kwargs["attention_mask"],
            **generate_kwargs,
        )
        return out

    def process_input(self, input_ids, pre_memory_state=None, prepend_state=None, suf_memory_state=None, generate=False, **kwargs):
        seg_kwargs = dict(**kwargs)

        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        if prepend_state is not None:
            prepend_embeds = self.model.get_input_embeddings()(prepend_state)
            inputs_embeds = torch.cat([prepend_embeds, inputs_embeds], dim=1)
        if pre_memory_state is not None:
            if generate:
                inputs_embeds = torch.cat([pre_memory_state, inputs_embeds], dim=1)
            else:
                if suf_memory_state is not None:
                    inputs_embeds = torch.cat([pre_memory_state, inputs_embeds, suf_memory_state], dim=1)
                else:
                    logger.info("Warning: suf_memory_state is None while pre_memory_state is not None during training. Using pre_memory_state for both prefix and suffix.")
                    inputs_embeds = torch.cat([pre_memory_state, inputs_embeds, pre_memory_state], dim=1)

        seg_kwargs["input_ids"] = None
        seg_kwargs["inputs_embeds"] = inputs_embeds
        if kwargs.get("attention_mask") is not None:
            # Do not reserve memory slots in the mask when no memory is attached.
            mem_tokens = pre_memory_state.shape[1] if pre_memory_state is not None else 0
            seg_kwargs["attention_mask"] = self.pad_attention_mask(
                kwargs["attention_mask"],
                inputs_embeds.shape,
                0 if prepend_state is None else self.n_prepend,
                generate,
                mem_tokens=mem_tokens,
            )
        seg_kwargs["output_hidden_states"] = True
        return seg_kwargs

    def pad_attention_mask(self, attention_mask, shape, n_prepend, generate=False, mem_tokens=None):
        # mem_tokens reflects how many memory embeddings were actually concatenated.
        mem_tokens = self.num_mem_embed if mem_tokens is None else mem_tokens
        if mem_tokens in {0, None}:
            mask = torch.ones(*shape[:2], dtype=torch.int64).to(attention_mask.device)
            mask[:, (n_prepend):] = attention_mask
            return mask
        else:
            mask = torch.ones(*shape[:2], dtype=torch.int64).to(attention_mask.device)
            if generate:
                mask[:, (mem_tokens + n_prepend):] = attention_mask
            else:
                mask[:, (mem_tokens + n_prepend) : -mem_tokens] = attention_mask
            return mask

    def process_output(self, model_outputs, n_prepend, **kwargs):
        if self.num_mem_embed not in {0, None}:
            out = CausalLMOutputWithCrossAttentions()
            out_memory_state = model_outputs.hidden_states[-1][:, -self.num_mem_embed :]
            out["logits"] = model_outputs.logits[:, (self.num_mem_embed + n_prepend) : -self.num_mem_embed]
            out["logits"] = out["logits"].cpu()

            if kwargs.get("output_hidden_states"):
                out["hidden_states"] = [
                    lh[:, (self.num_mem_embed + n_prepend) : -self.num_mem_embed] for lh in model_outputs.hidden_states
                ]
            if kwargs.get("output_attentions"):
                out["attentions"] = model_outputs["attentions"]
        else:
            out = CausalLMOutputWithCrossAttentions()
            out_memory_state = None
            out["logits"] = model_outputs.logits[:, (n_prepend):]

            if kwargs.get("output_hidden_states"):
                out["hidden_states"] = [lh[:, (n_prepend):] for lh in model_outputs.hidden_states]
            if kwargs.get("output_attentions"):
                out["attentions"] = model_outputs["attentions"]

        return out, out_memory_state
