from torch import nn
import torch
from .depth_decoder import DepthDecoder
from .resnet_encoder import ResnetEncoder

class Monodepth(nn.Module):
    def __init__(self, resnet_encoder, depth_decoder, seg_decoder=None, max_depth=10.0):
        super(Monodepth, self).__init__()
        self.encoder = resnet_encoder
        self.decoder = depth_decoder
        self.seg_decoder = seg_decoder
        self.max_depth = max_depth

    def forward(self, input_image, crop=None, mask_ratio=0.0):
        fusion_features = None
        
        features, mask = self.encoder(input_image, mask_ratio=mask_ratio)
        pred_depth = self.decoder(features)[('disp',0)] * self.max_depth
            
        return pred_depth, features, fusion_features

    def load_pretrained(self, path):
        source_dict = torch.load(path)
        target_dict = self.state_dict()
    
        for idx, (s_key, t_key) in enumerate(zip(source_dict.keys(),target_dict.keys())):
            if 'seg_decoder.decoder' in t_key:
                if t_key.split(".")[-3] in ['10','11','12','13']:
                    pass
                else:
                    target_dict[t_key] = source_dict[s_key.replace("seg_decoder.decoder","decoder.decoder")]
            else:
                target_dict[t_key] = source_dict[s_key]
            
        self.load_state_dict(target_dict)