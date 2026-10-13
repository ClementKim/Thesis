import torch
import torch.nn as nn

class IPMoE(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        ## Auxiliary Token
        self.aux = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, question, image):
        pass