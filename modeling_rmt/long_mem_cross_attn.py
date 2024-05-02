import math
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class CrossAttentionMemory(torch.nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        self.wq = torch.nn.Linear(self.dim, self.hidden_dim, bias=False, dtype=torch.bfloat16)
        self.wk = torch.nn.Linear(self.dim, self.hidden_dim, bias=False, dtype=torch.bfloat16)

    def forward(self, memory, inputs, mode="train", seg_num=0, browse_thres=4):
        if memory is None:
            return None, None, False
        inputs = inputs.cuda().bfloat16()
        memory = memory.cuda().bfloat16()
        batch_size, _, _ = inputs.shape
        xq = self.wq(inputs) # (batch, 1, hidden_dim)
        mk = self.wk(memory) # (batch, mem_len, hidden_dim)

        scores = torch.matmul(xq, mk.transpose(1,2)) / math.sqrt(self.hidden_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq) # (batch, 1, mem_len)
        hist = torch.flatten(torch.sub(torch.full((batch_size, 1), seg_num).cpu(), torch.argmax(scores, dim=2).cpu())).tolist()
        browse = False
        if hist[0] < browse_thres:
            browse = True
        output = torch.matmul(scores, memory) # (batch, 1, dim)
        inputs = inputs.cpu()
        memory = memory.cpu()

        return output, hist, browse
