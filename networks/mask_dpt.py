import torch
from torch import nn
from einops import repeat

from .vit import Transformer
from .dpt_utils import *
import random

class Masked_DPT(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        max_depth,
        features=[96, 192, 384, 768],
        hooks=[2, 5, 8, 11],
        vit_features=768,
        use_readout="ignore",
        start_index=1,
    ):
        super().__init__()
        
        # ViT
        self.encoder = encoder
        self.encoder.transformer.set_hooks(hooks)

        #read out processing (ignore / add / project[dpt use this process])
        readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)

        # 32, 48, 136, 384
        self.act_postprocess1 = nn.Sequential(
            # readout_oper[0],
            Transpose(1, 2),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=features[0],
                out_channels=features[0],
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )
        self.act_postprocess2 = nn.Sequential(
            # readout_oper[1],
            Transpose(1, 2),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=features[1],
                out_channels=features[1],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )
        self.act_postprocess3 = nn.Sequential(
            # readout_oper[2],
            Transpose(1, 2),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[2],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )
        self.act_postprocess4 = nn.Sequential(
            # readout_oper[3],
            Transpose(1, 2),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[3],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Conv2d(
                in_channels=features[3],
                out_channels=features[3],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )
        
        self.unflatten = nn.Sequential(
            nn.Unflatten(
                2,
                torch.Size(
                    [
                        encoder.get_image_size()[0] // encoder.get_patch_size()[0],
                        encoder.get_image_size()[1] // encoder.get_patch_size()[1],
                    ]
                ),
            )
        )

        self.scratch = make_scratch(features, 256)
        self.scratch.refinenet1 = make_fusion_block(256, False)
        self.scratch.refinenet2 = make_fusion_block(256, False)
        self.scratch.refinenet3 = make_fusion_block(256, False)
        self.scratch.refinenet4 = make_fusion_block(256, False)

        self.scratch.output_conv = head = nn.Sequential(
            nn.Conv2d(256, 256 // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256 // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
            nn.Identity(),
        )

        self.max_depth = max_depth
        self.target_size = encoder.get_image_size()

    def forward(self, img, K = 1):
        # assert mask_ratio >= 0 and mask_ratio < 1, 'masking ratio must be kept between 0 and 1'

        x = self.encoder.to_patch_embedding(img)
        b, n, _ = x.shape
        x += self.encoder.pos_embedding[:, 1:(n + 1)]
        x = self.encoder.dropout(x)
        
        if K == 1:
            glob = self.encoder.transformer(x)

            layer_1 = self.encoder.transformer.features[0] 
            layer_2 = self.encoder.transformer.features[1] 
            layer_3 = self.encoder.transformer.features[2] 
            layer_4 = self.encoder.transformer.features[3] 
            
            layer_1 = self.act_postprocess1[0](layer_1)
            layer_2 = self.act_postprocess2[0](layer_2)
            layer_3 = self.act_postprocess3[0](layer_3)
            layer_4 = self.act_postprocess4[0](layer_4)

        else:
            num_patches = (self.target_size[0] // self.encoder.get_patch_size()[0]) * \
                          (self.target_size[1] // self.encoder.get_patch_size()[1])
            batch_range = torch.arange(b, device = x.device)[:, None]
            rand_indices = torch.rand(b, num_patches, device = x.device).argsort(dim = -1)

            ### random shuffle the patches
            x = x[batch_range,rand_indices]
           
            ### assign mask
            v = sorted([random.randint(1,num_patches-1) for i in range(int(K-1))] + [0, num_patches])
            mask_v = torch.zeros(len(v[:-1]), num_patches).to(x.device)
            for i in range(len(v[:-1])):
                mask_v[i, v[i]:v[i+1]] = 1.0

            ### K-way augmented attention
            partial_token = self.encoder.transformer(x, (mask_v.transpose(0,1) @ mask_v))
            reform_indices = torch.argsort(rand_indices, dim=1)

            
            #no class
            layer_1 = self.act_postprocess1[0](self.encoder.transformer.features[0][batch_range, reform_indices])
            layer_2 = self.act_postprocess2[0](self.encoder.transformer.features[1][batch_range, reform_indices])
            layer_3 = self.act_postprocess3[0](self.encoder.transformer.features[2][batch_range, reform_indices])
            layer_4 = self.act_postprocess4[0](self.encoder.transformer.features[3][batch_range, reform_indices])

        features= [layer_1, layer_2, layer_3, layer_4]

        if layer_1.ndim == 3:
            layer_1 = self.unflatten(layer_1)
        if layer_2.ndim == 3:
            layer_2 = self.unflatten(layer_2)
        if layer_3.ndim == 3:
            layer_3 = self.unflatten(layer_3)
        if layer_4.ndim == 3:
            layer_4 = self.unflatten(layer_4)

        layer_1 = self.act_postprocess1[1:](layer_1)
        layer_2 = self.act_postprocess2[1:](layer_2)
        layer_3 = self.act_postprocess3[1:](layer_3)
        layer_4 = self.act_postprocess4[1:](layer_4)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        fusion_features = [layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn]

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        pred_depth = self.scratch.output_conv(path_1) * self.max_depth
        
        return pred_depth, features, fusion_features
    
    
    def resize_image_size(self, h, w, start_index=1):
        self.encoder.resize_pos_embed(h, w, start_index)
        self.unflatten = nn.Sequential( 
                            nn.Unflatten(
                                2,
                                torch.Size(
                                    [
                                        self.encoder.get_image_size()[0] // self.encoder.get_patch_size()[0],
                                        self.encoder.get_image_size()[1] // self.encoder.get_patch_size()[1],
                                    ]
                                ),
                            )   
                        )
    def target_out_size(self,h, w):
        self.unflatten = nn.Sequential(
            nn.Unflatten(
                2,
                torch.Size(
                    [
                        h // self.encoder.get_patch_size()[0],
                        w // self.encoder.get_patch_size()[1],
                    ]
                ),
            )
        )
        self.target_size = (h,w)
        
