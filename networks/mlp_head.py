import numpy as np
import torch
import torch.nn as nn

from .vit import Transformer

class MLPHead(nn.Module):
    def __init__(self, num_ch = [768, 768, 768, 768], out_ch = 1024, device="cuda"):
        super(MLPHead, self).__init__()
        self.mlps = nn.ModuleList()
        for n in num_ch:
            self.mlps.append(nn.Sequential(*[nn.Linear(n, out_ch), nn.ReLU(), nn.Linear(out_ch, n)]))
    def forward(self, input_feature, index):
        output_features = self.mlps[index](input_feature)     
        return output_features