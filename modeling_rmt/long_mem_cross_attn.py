import math
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class CrossAttentionMemory(torch.nn.Module):
    def __init__(self, memory_size, dim, hidden_dim):
        super().__init__()
        self.memory_size = memory_size
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        self.wq = torch.nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.wk = torch.nn.Linear(self.dim, self.hidden_dim, bias=False)

    def forward(self, memory, inputs):
        if memory is None:
            return None
        inputs = inputs.cuda()
        memory = memory.cuda()
        xq = self.wq(inputs) # (batch, 1, hidden_dim)
        mk = self.wk(memory) # (batch, mem_len, hidden_dim)

        scores = torch.matmul(xq, mk.transpose(1,2)) / math.sqrt(self.hidden_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq) # (batch, 1, mem_len)
        output = torch.matmul(scores, memory) # (batch, 1, dim)
        inputs = inputs.cpu()
        memory = memory.cpu()

        return output
