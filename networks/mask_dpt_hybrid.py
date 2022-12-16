import torch
from torch import nn
from einops import repeat

from .vit import Transformer
from .dpt_utils import *

class Masked_DPT_hybrid(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        features=[96, 192, 384, 768],
        hooks=[2, 5, 8, 11],
        vit_features=768,
        max_depth= 10.0,
        use_readout="ignore",
        start_index=1,
    ):
        super().__init__()
        
        # ViT
        self.encoder = encoder
        
        
        self.encoder.transformer.set_hooks(hooks)
        self.max_depth = max_depth
        readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)

        # 32, 48, 136, 384
        self.act_postprocess1 = nn.Sequential(
            nn.Identity(), nn.Identity(), nn.Identity()
        )
        self.act_postprocess2 = nn.Sequential(
            nn.Identity(), nn.Identity(), nn.Identity()
        )                                                    #### in hybrid

        self.act_postprocess3 = nn.Sequential(
            readout_oper[2],
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
            readout_oper[3],
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
        self.scratch.refinenet4 = make_fusion_block(256, False)
        self.scratch.refinenet3 = make_fusion_block(256, False)
        self.scratch.refinenet2 = make_fusion_block(256, False)
        self.scratch.refinenet1 = make_fusion_block(256, False)
    
        self.scratch.output_conv = head = nn.Sequential(
            nn.Conv2d(256, 256 // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256 // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            #nn.ReLU(True),
            nn.Sigmoid(),
            nn.Identity(),
        )   

    def forward(self, img, mask_ratio = 0.0):
        assert mask_ratio >= 0 and mask_ratio < 1, 'masking ratio must be kept between 0 and 1'

        if mask_ratio == 0.0:
            glob = self.encoder(img)

            layer_1 = self.encoder.to_patch_embedding.backbone.stages[0].features 
            layer_2 = self.encoder.to_patch_embedding.backbone.stages[1].features 
            layer_3 = self.encoder.transformer.features[2] 
            layer_4 = self.encoder.transformer.features[3] 

            layer_1 = self.act_postprocess1[0:2](layer_1)
            layer_2 = self.act_postprocess2[0:2](layer_2)
            layer_3 = self.act_postprocess3[0:2](layer_3)
            layer_4 = self.act_postprocess4[0:2](layer_4)
        else:
            x = self.encoder.to_patch_embedding(img)
            b, n, _ = x.shape
            num_patches = (self.encoder.get_image_size()[0] // self.encoder.get_patch_size()[0]) * \
                          (self.encoder.get_image_size()[1] // self.encoder.get_patch_size()[1])

            cls_tokens = repeat(self.encoder.cls_token, '1 n d -> b n d', b = b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.encoder.pos_embedding[:, :(n + 1)]
            x = self.encoder.dropout(x)

            rand_indices = torch.rand(b, num_patches, device = x.device).argsort(dim = -1)
            mask_indices = rand_indices.clone()

            num_masked = int(num_patches*mask_ratio)
            batch_range = torch.arange(b, device = x.device)[:, None]
            layer_1 = []
            layer_2 = []
            layer_3 = []
            layer_4 = []

            for i in range(int(1/mask_ratio)-1):
                unmasked_indices = mask_indices[:, :num_masked]
                mask_indices    = mask_indices[:, num_masked:]

                partial_token = self.encoder.transformer(torch.cat((x[:,0,:].unsqueeze(1),(x[:,1:,:])[batch_range,unmasked_indices]), dim=1))
                
                layer_3.append(self.act_postprocess3[0](self.encoder.transformer.features[2]))
                layer_4.append(self.act_postprocess4[0](self.encoder.transformer.features[3]))

            partial_token = self.encoder.transformer(torch.cat((x[:,0,:].unsqueeze(1),(x[:,1:,:])[batch_range,mask_indices]), dim=1))
            layer_1.append(self.encoder.to_patch_embedding.backbone.stages[0].features)
            layer_2.append(self.encoder.to_patch_embedding.backbone.stages[1].features)
            layer_3.append(self.act_postprocess3[0](self.encoder.transformer.features[2]))
            layer_4.append(self.act_postprocess4[0](self.encoder.transformer.features[3]))
            
            reform_indices = torch.argsort(rand_indices, dim=1)
            layer_1 = self.act_postprocess1[1](torch.cat(layer_1, dim=1))
            layer_2 = self.act_postprocess2[1](torch.cat(layer_2, dim=1))
            layer_3 = self.act_postprocess3[1](torch.cat(layer_3, dim=1)[batch_range, reform_indices])
            layer_4 = self.act_postprocess4[1](torch.cat(layer_4, dim=1)[batch_range, reform_indices])
            
        features= [layer_3, layer_4]

        if layer_1.ndim == 3:
            layer_1 = self.unflatten(layer_1)
        if layer_2.ndim == 3:
            layer_2 = self.unflatten(layer_2)
        if layer_3.ndim == 3:
            layer_3 = self.unflatten(layer_3)
        if layer_4.ndim == 3:
            layer_4 = self.unflatten(layer_4)

        layer_1 = self.act_postprocess1[2:](layer_1)
        layer_2 = self.act_postprocess2[2:](layer_2)
        layer_3 = self.act_postprocess3[2:](layer_3)
        layer_4 = self.act_postprocess4[2:](layer_4)

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

    def load_pretrained(self, path):
        source_dict = torch.load(path)
        target_dict = self.state_dict()
    
        for idx, (s_key, t_key) in enumerate(zip(source_dict.keys(),target_dict.keys())):
            if 'scratch' in s_key:
                pass
            else:
                target_dict[t_key] = source_dict[s_key]
                if 'norm' in t_key:
                    target_dict[t_key].eps = 1e-5
            
        self.load_state_dict(target_dict)