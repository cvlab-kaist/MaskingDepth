import math

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from timm.models.vision_transformer_hybrid import HybridEmbed
from timm.models.resnetv2 import create_resnetv2_stem, make_div, Bottleneck, DownsampleConv, DownsampleAvg, \
    PreActBottleneck
from timm.models.layers import GroupNormAct, ClassifierHead, StdConv2d, create_conv2d, StdConv2dSame
from functools import partial
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim,eps=1e-06)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = True)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None, drop_True=False):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        #### make attention score from masked region to zero
        if not(mask == None):
            dots[:,:, mask==0.0] = -10.0
        
        attn = self.attend(dots)
        if drop_True:
            attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        self.hooks = []
        self.features = None

    def set_hooks(self, hooks):
        self.hooks = hooks
    
    def forward(self, x, mask=None, drop_prob=0.0):
        i = 0
        ll = []
        drop_True = torch.rand(1) < drop_prob
        for attn, ff in self.layers:
            # if mask == None or i in self.hooks[-1]:
            if mask == None:
                x = attn(x, drop_True=drop_True) + x
            else:
                x = attn.fn(attn.norm(x),mask, drop_True=drop_True) + x

            x = ff(x) + x
            if i in self.hooks:
                ll.append(x)
            i += 1
        self.features = tuple(ll)
    
        return x
   
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, hybrid=False, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = image_size #pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.__image_size = image_size
        self.__patch_size = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        if hybrid:
            backbone = ResNetV2(
                layers=(3,4,9), num_classes=0, global_pool='', in_chans=channels,
                preact=False, stem_type='same')
            self.to_patch_embedding = HybridEmbed(
                    backbone, img_size=image_size, in_chans=channels, embed_dim=dim)
        else:
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                nn.Linear(patch_dim, dim),
            )


        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim,eps=1e-06),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

    def get_image_size(self):
        return self.__image_size
    def get_patch_size(self):
        return self.__patch_size

    def resize_pos_embed(self, h, w, start_index=1):
        self.__image_size = (h,w)
        pw, ph = self.get_patch_size()
        gs_w = w//pw
        gs_h = h//ph

        posemb_tok, posemb_grid = (
            self.pos_embedding[:, : start_index],
            self.pos_embedding[0, start_index :],
        )

        gs_old = int(math.sqrt(len(posemb_grid)))

        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

        self.pos_embedding = nn.Parameter(torch.cat([posemb_tok, posemb_grid], dim=1))


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode.
    """

    def __init__(
            self, layers, channels=(256, 512, 1024, 2048),
            num_classes=1000, in_chans=3, global_pool='avg', output_stride=32,
            width_factor=1, stem_chs=64, stem_type='', avg_down=False, preact=True,
            act_layer=nn.ReLU, conv_layer=StdConv2dSame, norm_layer=partial(GroupNormAct, num_groups=32),
            drop_rate=0., drop_path_rate=0., zero_init_last=False):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        wf = width_factor

        self.feature_info = []
        stem_chs = make_div(stem_chs * wf)
        self.stem = create_resnetv2_stem(
            in_chans, stem_chs, stem_type, preact, conv_layer=conv_layer, norm_layer=norm_layer)
        stem_feat = ('stem.conv3' if is_stem_deep(stem_type) else 'stem.conv') if preact else 'stem.norm'
        self.feature_info.append(dict(num_chs=stem_chs, reduction=2, module=stem_feat))

        prev_chs = stem_chs
        curr_stride = 4
        dilation = 1
        block_dprs = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(layers)).split(layers)]
        block_fn = PreActBottleneck if preact else Bottleneck
        self.stages = nn.Sequential()
        for stage_idx, (d, c, bdpr) in enumerate(zip(layers, channels, block_dprs)):
            out_chs = make_div(c * wf)
            stride = 1 if stage_idx == 0 else 2
            if curr_stride >= output_stride:
                dilation *= stride
                stride = 1
            stage = ResNetStage(
                prev_chs, out_chs, stride=stride, dilation=dilation, depth=d, avg_down=avg_down,
                act_layer=act_layer, conv_layer=conv_layer, norm_layer=norm_layer, block_dpr=bdpr, block_fn=block_fn)
            prev_chs = out_chs
            curr_stride *= stride
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{stage_idx}')]
            self.stages.add_module(str(stage_idx), stage)

        self.num_features = prev_chs
        self.norm = norm_layer(self.num_features) if preact else nn.Identity()
        self.head = ClassifierHead(
            self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate, use_conv=True)

        #self.init_weights(zero_init_last=zero_init_last)
        self.grad_checkpointing = False

    @torch.jit.ignore
    def init_weights(self, zero_init_last=True):
        named_apply(partial(_init_weights, zero_init_last=zero_init_last), self)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix='resnet/'):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
                (r'^norm', (99999,))
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.head = ClassifierHead(
            self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate, use_conv=True)

    def forward_features(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x, flatten=True)
        else:
            x = self.stages(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class ResNetStage(nn.Module):
    """ResNet Stage."""
    def __init__(
            self, in_chs, out_chs, stride, dilation, depth, bottle_ratio=0.25, groups=1,
            avg_down=False, block_dpr=None, block_fn=PreActBottleneck,
            act_layer=None, conv_layer=None, norm_layer=None, **block_kwargs):
        super(ResNetStage, self).__init__()
        first_dilation = 1 if dilation in (1, 2) else 2
        layer_kwargs = dict(act_layer=act_layer, conv_layer=conv_layer, norm_layer=norm_layer)
        proj_layer = DownsampleAvg if avg_down else DownsampleConv
        prev_chs = in_chs
        self.blocks = nn.Sequential()
        for block_idx in range(depth):
            drop_path_rate = block_dpr[block_idx] if block_dpr else 0.
            stride = stride if block_idx == 0 else 1
            self.blocks.add_module(str(block_idx), block_fn(
                prev_chs, out_chs, stride=stride, dilation=dilation, bottle_ratio=bottle_ratio, groups=groups,
                first_dilation=first_dilation, proj_layer=proj_layer, drop_path_rate=drop_path_rate,
                **layer_kwargs, **block_kwargs))
            prev_chs = out_chs
            first_dilation = dilation
            proj_layer = None

    def forward(self, x):
        x = self.blocks(x)
        self.features = x
        return x