import torch
from torch import nn
from einops import repeat

from .dpt_utils import *

class UncertDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.refinenet1 = make_fusion_block(256, False)
        self.refinenet2 = make_fusion_block(256, False)
        self.refinenet3 = make_fusion_block(256, False)
        self.refinenet4 = make_fusion_block(256, False)

        self.output_conv = head = nn.Sequential(
            nn.Conv2d(256, 256 // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256 // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

    def forward(self, features):
        layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn = features[0], features[1], features[2], features[3]
        
        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)

        pred_uncert = self.output_conv(path_1)
        
        return pred_uncert