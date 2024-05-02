import torch
from transformers import LlamaForCausalLM
from scipy.fftpack import dct
import numpy as np

class EmbedAutoEncoder(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dct_map = torch.from_numpy(dct(np.eye(hidden_dim), norm='ortho', axis=0))[1024:-1024,:].bfloat16().cuda()
        self.rec_loss = 0

    def forward(self, inputs):
        if inputs.shape[1] > 0:
            x = torch.transpose(inputs, 1, 2)
            x = torch.einsum('ij,bjk->bik', self.dct_map, x)
            x = torch.einsum('ij,bjk->bik', self.dct_map.t(), x)
            x = torch.transpose(x, 1, 2)
            self.rec_loss = torch.sum(torch.mean(torch.pow(torch.sub(inputs, x), 2), axis=2))
            return x
        else:
            return inputs

def inject_eae(model, hidden_dim, kernel_size, stride):
    if isinstance(model, LlamaForCausalLM):
        for i in range(len(model.model.layers)):
            layer = model.model.layers[i]
            mlp = torch.nn.Sequential(
                EmbedAutoEncoder(hidden_dim),
                layer.mlp
            )
            
            model.model.layers[i].mlp = mlp
        return model
    else:
        print("Not Implement Yet. Return the original model.")
        return model
