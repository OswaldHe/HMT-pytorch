import math
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class CrossAttentionMemory(torch.nn.Module):
    def __init__(self, emb, memory_size, dim, hidden_dim):
        super().__init__()
        self.emb = emb
        self.memory_size = memory_size
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        self.wq = torch.nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.wk = torch.nn.Linear(self.dim, self.hidden_dim, bias=False)

    def forward(self, memory, inputs):
        if memory is None:
            return None
        inputs_embeds = self.emb.cpu()(inputs)
        xq = self.wq.cpu()(inputs_embeds) # (batch, seq_len, hidden_dim)
        mk = self.wk.cpu()(memory) # (batch, mem_len, hidden_dim)

        scores = torch.matmul(xq, mk.transpose(1,2)) / math.sqrt(self.hidden_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq) # (seq_len, mem_len)
        scores = F.softmax(scores.float().sum(dim=1, keepdim=True), dim=-1).type_as(xq) # (1, mem_len)
        output = torch.matmul(scores, memory) # (1, dim)
        output = output.cuda()

        return output
