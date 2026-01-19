"""
Most of this code comes from the timm  library.
We tried to disentangle from the timm library version.

Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

"""
import logging
import math
import warnings
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepaudiox.modules.backbones.base_backbone import BaseBackbone
from deepaudiox.modules.backbones.passt.modules import Block, PatchEmbed
from deepaudiox.modules.backbones.passt.preprocess import AugmentMelSTFT
from deepaudiox.modules.backbones.passt.vit_helpers import (
    init_vit_weights,
    load_pretrained_weights,
    trunc_normal_,
)

# Global variables
_logger = logging.getLogger()

class PaSSTConfig:
    def __init__(self, cfg=None):
        self.url = 'https://github.com/kkoutini/PaSST/releases/download/v.0.0.9/passt-s-kd-ap.486.pt'
        self.num_classes = 527, 
        self.pool_size = None
        self.crop_pct = 1.0
        self.interpolation = 'bicubic'
        self.fixed_input_size = True
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.first_conv = 'patch_embed.proj' 
        self.classifiers = ('head.1', 'head_dist')
        self.u_patchout = 0
        self.s_patchout_t = 0
        self.s_patchout_f = 0
        self.embed_dim = 768
        self.distilled = True
        self.pretrained = True
        self.img_size = (128, 998)
        self.patch_size = 16
        self.fstride = 10
        self.tstride = 10
        self.in_chans = 1
        self.depth = 12
        self.num_heads = 12
        self.mlp_ratio = 4.
        self.qkv_bias = True
        self.representation_size = None
        self.drop_rate = 0.
        self.attn_drop_rate = 0.
        self.drop_path_rate = 0.
        self.norm_layer = None
        self.act_layer = None
        self.weight_init = ''

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)

class PaSST(BaseBackbone):
    def __init__(
        self, 
        cfg: PaSSTConfig = PaSSTConfig(),
        sample_rate: int = 16_000
    ):
        super().__init__(out_dim=768, sample_rate=sample_rate)
        self.cfg = cfg
        self.feature_extractor = AugmentMelSTFT()

        self.num_classes = self.cfg.num_classes
        self.u_patchout = self.cfg.u_patchout
        self.s_patchout_t = self.cfg.s_patchout_t
        self.s_patchout_f = self.cfg.s_patchout_f
        self.num_features = self.embed_dim = self.cfg.embed_dim 
        self.num_tokens = 2 if self.cfg.distilled else 1
        self.embed_dim = self.cfg.embed_dim
        self.classifiers = self.cfg.classifiers
        self.patch_embed = PatchEmbed(
            img_size=self.cfg.img_size, 
            patch_size=self.cfg.patch_size, 
            stride=(self.cfg.fstride, self.cfg.tstride), 
            in_chans=self.cfg.in_chans, 
            embed_dim=self.embed_dim,
            flatten=False
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) if self.cfg.distilled else None

        self.new_pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, self.embed_dim))  # for C and D tokens
        self.freq_new_pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, self.patch_embed.grid_size[0], 1))  # | f
        self.time_new_pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, 1, self.patch_embed.grid_size[1]))  # __ t
        self.pos_drop = nn.Dropout(p=self.cfg.drop_rate)

        norm_layer = self.cfg.norm_layer or partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, self.cfg.drop_path_rate, self.cfg.depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=self.embed_dim, 
                num_heads=self.cfg.num_heads, 
                mlp_ratio=self.cfg.mlp_ratio, 
                qkv_bias=self.cfg.qkv_bias, 
                drop=self.cfg.drop_rate,
                attn_drop=self.cfg.attn_drop_rate, 
                drop_path=dpr[i], 
                norm_layer=norm_layer, 
                act_layer=self.cfg.act_layer or nn.GELU
            )
            for i in range(self.cfg.depth)])
        self.norm = norm_layer(self.embed_dim)

        # Representation layer
        if self.cfg.representation_size and not self.distilled:
            self.num_features = self.cfg.representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict([
                    ('fc', nn.Linear(self.embed_dim, self.cfg.representation_size)),
                    ('act', nn.Tanh())
                ])
            )
        else:
            self.pre_logits = nn.Identity()        
        
        # Initialize weights
        if self.cfg.weight_init not in ['nlhb', '']:
            raise ValueError(f"Unsuported weight initialization mode: {self.cfg.weight_init}")

        trunc_normal_(self.new_pos_embed, std=.02)
        trunc_normal_(self.freq_new_pos_embed, std=.02)
        trunc_normal_(self.time_new_pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(init_vit_weights)

        # Load pre-trained weights
        if self.cfg.url is not None:
            load_pretrained_weights(
                model = self,
                pretrained_url = self.cfg.url,
                num_classes = self.num_classes,
                in_chans = self.cfg.in_chans,
                filter_fn = checkpoint_filter_fn,
                strict = True,
                first_conv = self.cfg.first_conv,
                classifiers = self.classifiers
            )

    def extract_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(waveforms)
        features = features.unsqueeze(1)
        return features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.patch_embed(features)
        B_dim, E_dim, F_dim, T_dim = features.shape 
        time_new_pos_embed = self.time_new_pos_embed
        if features.shape[-1] != time_new_pos_embed.shape[-1]:
            time_new_pos_embed = time_new_pos_embed[:, :, :, :features.shape[-1]]    
        features = features + time_new_pos_embed
        features = features + self.freq_new_pos_embed

        if self.training and self.s_patchout_t:
            # ([1, 768, 1, 82])
            random_indices = torch.randperm(T_dim)[:T_dim - self.s_patchout_t].sort().values
            features = features[:, :, :, random_indices]
        if self.training and self.s_patchout_f:
            # [1, 768, 12, 1]
            random_indices = torch.randperm(F_dim)[:F_dim - self.s_patchout_f].sort().values
            features = features[:, :, random_indices, :]

        # Flatten the sequence
        features = features.flatten(2).transpose(1, 2)
        if self.training and self.u_patchout:
            seq_len = features.shape[1]
            random_indices = torch.randperm(seq_len)[:seq_len - self.u_patchout].sort().values
            features = features[:, random_indices, :]

        cls_tokens = self.cls_token.expand(B_dim, -1, -1) + self.new_pos_embed[:, :1, :]
        if self.dist_token is None:
            features = torch.cat((cls_tokens, features), dim=1)
        else:
            dist_token = self.dist_token.expand(B_dim, -1, -1) + self.new_pos_embed[:, 1:, :]
            features = torch.cat((cls_tokens, dist_token, features), dim=1)

        features = self.pos_drop(features)
        features = self.blocks(features)

        features = self.norm(features)

        # if self.dist_token is None:
        #     return self.pre_logits(features[:, 0])
        # else:
        #     return features[:, 0], features[:, 1]
        if self.dist_token is None:
            return self.pre_logits(features)
        else:
            return features

def adapt_image_pos_embed_to_passt(
    posemb, 
    num_tokens=1, 
    gs_new=(), 
    mode='bicubic'
):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s with %s cls/dis tokens', posemb.shape, gs_new, num_tokens)
    
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
        
    gs_old = int(math.sqrt(len(posemb_grid)))

    assert len(gs_new) >= 2

    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode=mode, align_corners=False)
    freq_new_pos_embed = posemb_grid.mean(dim=3, keepdim=True)
    time_new_pos_embed = posemb_grid.mean(dim=2, keepdim=True)

    _logger.info('New Position cls/dstl embedding %s', posemb_tok.shape)
    _logger.info('New FREQ Position embedding %s', freq_new_pos_embed.shape)
    _logger.info('New TIME Position embedding %s', time_new_pos_embed.shape)
    
    return posemb_tok, freq_new_pos_embed, time_new_pos_embed

def resize_pos_embed(
    posemb, 
    posemb_new, 
    num_tokens=1, 
    gs_new=(), 
    mode='bicubic'
):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s with %s cls/dis tokens', posemb.shape, posemb_new.shape, num_tokens)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode=mode, align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

def checkpoint_filter_fn(state_dict, model):
    """
    Convert patch embedding weights from manual patchify + linear projection
    to convolution-based patch embedding and adapt positional embeddings.
    """
    out_dict = {}

    # Handle checkpoints saved as {"model": state_dict}
    if "model" in state_dict:
        state_dict = state_dict["model"]

    state_dict = {k: v for k, v in state_dict.items()}

    # --------------------------------------------------
    # Adapt ImageNet positional embeddings to PaSST
    # --------------------------------------------------
    if "time_new_pos_embed" not in state_dict:
        _logger.info(
            "Adapting pos embedding from ImageNet pretrained model to PaSST."
        )

        pos_embed = state_dict.pop("pos_embed")
        (
            new_pos_embed,
            freq_new_pos_embed,
            time_new_pos_embed,
        ) = adapt_image_pos_embed_to_passt(
            pos_embed,
            getattr(model, "num_tokens", 1),
            model.patch_embed.grid_size,
        )

        state_dict["new_pos_embed"] = new_pos_embed
        state_dict["freq_new_pos_embed"] = freq_new_pos_embed
        state_dict["time_new_pos_embed"] = time_new_pos_embed

    # --------------------------------------------------
    # Process remaining parameters
    # --------------------------------------------------
    for key, value in state_dict.items():
        if "patch_embed.proj.weight" in key and value.ndim < 4:
            # Old models: linear patch embedding → conv patch embedding
            (
                out_channels,
                in_channels,
                kernel_height,
                kernel_width,
            ) = model.patch_embed.proj.weight.shape

            value = value.reshape(
                out_channels,
                -1,
                kernel_height,
                kernel_width,
            )

        elif key == "pos_embed" and value.shape != model.new_pos_embed.shape:
            # Defensive resize (should rarely occur)
            value = resize_pos_embed(
                value,
                model.new_pos_embed,
                getattr(model, "num_tokens", 1),
                model.patch_embed.grid_size,
            )

        out_dict[key] = value

    return out_dict