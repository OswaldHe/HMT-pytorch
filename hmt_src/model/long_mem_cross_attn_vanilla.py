import math
import torch
import torch.nn.functional as F

class CrossAttentionMemory(torch.nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        self.wq = torch.nn.Linear(self.dim, self.hidden_dim, bias=False, dtype=torch.bfloat16)
        self.wk = torch.nn.Linear(self.dim, self.hidden_dim, bias=False, dtype=torch.bfloat16)

    def forward(self, memory_seq, inputs, mode="train", seg_num=0):
        if memory_seq is None:
            # No memory to attend to for the first segment; skip adding a bogus prompt.
            # Returning None avoids injecting meaningless tokens that can destabilize training.
            return None, None
        inputs = inputs.cuda().bfloat16()
        memory_seq = memory_seq.cuda().bfloat16()
        batch_size, _, _ = inputs.shape
        xq = self.wq(inputs) # (batch, 1, hidden_dim)
        mk = self.wk(memory_seq) # (batch, mem_len, hidden_dim)

        scores = torch.matmul(xq, mk.transpose(1,2)) / math.sqrt(self.hidden_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq) # (batch, 1, mem_len)
        hist = torch.flatten(torch.sub(torch.full((batch_size, 1), seg_num).cpu(), torch.argmax(scores, dim=2).cpu())).tolist()
        output = torch.matmul(scores, memory_seq) # (batch, 1, dim)
        inputs = inputs.cpu()
        memory_seq = memory_seq.cpu()

        return output, hist
