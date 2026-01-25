"""
Most of this code comes from the timm  library.
We tried to disentangle from the timm library version.

Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

"""

import math
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepaudiox.modules.backbones.passt.modules import Block, PatchEmbed
from deepaudiox.modules.backbones.passt.preprocess import AugmentMelSTFT
from deepaudiox.modules.backbones.passt.vit_helpers import init_vit_weights, trunc_normal_
from deepaudiox.modules.baseclasses import BaseBackbone


class PaSSTConfig:
    """
    Configuration class for PaSST (Patchout Spectrogram Transformer).

    Stores all hyperparameters and settings for the PaSST model,
    including model architecture, patch sizes, positional embeddings,
    dropout rates, and other training-related options.

    Args:
        cfg (dict, optional): A dictionary of config overrides. Defaults to None.

    Attributes:
        num_classes (int): Number of output classes.
        pool_size (tuple | None): Pooling size for feature maps.
        crop_pct (float): Fraction of the image to crop when resizing.
        interpolation (str): Interpolation method for resizing ('bicubic', 'bilinear', etc.).
        fixed_input_size (bool): Whether the input size is fixed.
        mean (tuple): Input normalization mean.
        std (tuple): Input normalization std.
        first_conv (str): Name of the first convolutional layer.
        classifiers (tuple): Names of classifier layers.
        u_patchout, s_patchout_t, s_patchout_f (int): Patchout parameters for data augmentation.
        embed_dim (int): Embedding dimension.
        distilled (bool): Whether to use knowledge distillation token.
        pretrained (bool): Whether pretrained weights are used.
        img_size (tuple): Input image/spectrogram size (frequency x time).
        patch_size (int): Patch size.
        fstride, tstride (int): Frequency and time stride for patching.
        in_chans (int): Number of input channels.
        depth (int): Number of transformer blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of hidden dimension in MLP relative to embedding.
        qkv_bias (bool): If True, include bias in Q/K/V projections.
        representation_size (int | None): Size of the representation layer, if used.
        drop_rate, attn_drop_rate, drop_path_rate (float): Dropout rates.
        norm_layer, act_layer (nn.Module | None): Normalization and activation layers.
        weight_init (str): Weight initialization method.
    """

    def __init__(self, cfg=None):
        self.num_classes = (527,)
        self.pool_size = None
        self.crop_pct = 1.0
        self.interpolation = "bicubic"
        self.fixed_input_size = True
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.first_conv = "patch_embed.proj"
        self.classifiers = ("head.1", "head_dist")
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
        self.mlp_ratio = 4.0
        self.qkv_bias = True
        self.representation_size = None
        self.drop_rate = 0.0
        self.attn_drop_rate = 0.0
        self.drop_path_rate = 0.0
        self.norm_layer = None
        self.act_layer = None
        self.weight_init = ""

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        """
        Update the configuration using a dictionary.

        Args:
            cfg (dict): Dictionary containing configuration keys and values to update.
        """
        self.__dict__.update(cfg)


class PaSST(BaseBackbone):
    """
    Patchout Spectrogram Transformer (PaSST) backbone.

    A transformer-based model for audio classification that operates
    on spectrogram inputs and uses patchout regularization. Supports
    distillation tokens, positional embeddings in both time and frequency
    dimensions, and multiple patchout strategies.

    Pretrained checkpoint: https://github.com/kkoutini/PaSST/releases/download/v.0.0.9/passt-s-kd-ap.486.pt

    Args:
        cfg (PaSSTConfig | None): Model configuration. Defaults to None.
        sample_rate (int): Input audio sample rate. Defaults to 16_000 Hz.

    Attributes:
        feature_extractor (nn.Module): Converts waveforms to spectrograms.
        patch_embed (PatchEmbed): Converts spectrograms to patch embeddings.
        cls_token (nn.Parameter): Class token embedding.
        dist_token (nn.Parameter | None): Distillation token embedding.
        new_pos_embed, freq_new_pos_embed, time_new_pos_embed (nn.Parameter): Positional embeddings.
        blocks (nn.Sequential): Transformer blocks.
        norm (nn.Module): Layer normalization applied to final features.
        pre_logits (nn.Module): Optional representation layer before classifier.
    """

    def __init__(self, cfg: PaSSTConfig | None = None, sample_rate: int = 16_000):
        """_summary_

        Args:
            cfg (PaSSTConfig | None, optional): _description_. Defaults to None.
            sample_rate (int, optional): _description_. Defaults to 16_000.

        Raises:
            ValueError: _description_
        """
        super().__init__(out_dim=768, sample_rate=sample_rate)
        if cfg is None:
            self.cfg = PaSSTConfig()
        else:
            self.cfg = cfg
        self.feature_extractor = AugmentMelSTFT(sr=sample_rate)

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
            flatten=False,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) if self.cfg.distilled else None

        self.new_pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, self.embed_dim))  # for C and D tokens
        self.freq_new_pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, self.patch_embed.grid_size[0], 1))  # | f
        self.time_new_pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, 1, self.patch_embed.grid_size[1]))  # __ t
        self.pos_drop = nn.Dropout(p=self.cfg.drop_rate)

        norm_layer = self.cfg.norm_layer or partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, self.cfg.drop_path_rate, self.cfg.depth)]
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=self.embed_dim,
                    num_heads=self.cfg.num_heads,
                    mlp_ratio=self.cfg.mlp_ratio,
                    qkv_bias=self.cfg.qkv_bias,
                    drop=self.cfg.drop_rate,
                    attn_drop=self.cfg.attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=self.cfg.act_layer or nn.GELU,
                )
                for i in range(self.cfg.depth)
            ]
        )
        self.norm = norm_layer(self.embed_dim)

        # Representation layer
        if self.cfg.representation_size and not self.distilled:
            self.num_features = self.cfg.representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict([("fc", nn.Linear(self.embed_dim, self.cfg.representation_size)), ("act", nn.Tanh())])
            )
        else:
            self.pre_logits = nn.Identity()

        # Initialize weights
        if self.cfg.weight_init not in ["nlhb", ""]:
            raise ValueError(f"Unsuported weight initialization mode: {self.cfg.weight_init}")

        trunc_normal_(self.new_pos_embed, std=0.02)
        trunc_normal_(self.freq_new_pos_embed, std=0.02)
        trunc_normal_(self.time_new_pos_embed, std=0.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(init_vit_weights)

    def extract_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Convert raw audio waveforms into spectrogram features.

        Args:
            waveforms (torch.Tensor): Audio tensor of shape (batch, time).

        Returns:
            torch.Tensor: Spectrogram features of shape (batch, 1, freq, time).
        """
        features = self.feature_extractor(waveforms)
        features = features.unsqueeze(1)
        return features

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass of the PaSST backbone.

        Args:
            features (torch.Tensor): Input spectrogram features (batch, channels, freq, time).

        Returns:
            torch.Tensor: Transformer features after patch embedding, positional encoding,
                          transformer blocks, and normalization. Returns features excluding
                          class/distillation tokens.
        """
        x = self.patch_embed(x)
        B_dim, E_dim, F_dim, T_dim = x.shape
        time_new_pos_embed = self.time_new_pos_embed
        if x.shape[-1] != time_new_pos_embed.shape[-1]:
            time_new_pos_embed = time_new_pos_embed[:, :, :, : x.shape[-1]]
        x = x + time_new_pos_embed
        x = x + self.freq_new_pos_embed

        if self.training and self.s_patchout_t:
            # ([1, 768, 1, 82])
            random_indices = torch.randperm(T_dim)[: T_dim - self.s_patchout_t].sort().values
            x = x[:, :, :, random_indices]
        if self.training and self.s_patchout_f:
            # [1, 768, 12, 1]
            random_indices = torch.randperm(F_dim)[: F_dim - self.s_patchout_f].sort().values
            x = x[:, :, random_indices, :]

        # Flatten the sequence
        x = x.flatten(2).transpose(1, 2)
        if self.training and self.u_patchout:
            seq_len = x.shape[1]
            random_indices = torch.randperm(seq_len)[: seq_len - self.u_patchout].sort().values
            x = x[:, random_indices, :]

        cls_tokens = self.cls_token.expand(B_dim, -1, -1) + self.new_pos_embed[:, :1, :]
        if self.dist_token is None:
            x = torch.cat((cls_tokens, x), dim=1)
        else:
            dist_token = self.dist_token.expand(B_dim, -1, -1) + self.new_pos_embed[:, 1:, :]
            x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = self.pos_drop(x)
        x = self.blocks(x)

        x = self.norm(x)

        # if self.dist_token is None:
        #     return self.pre_logits(features[:, 0])
        # else:
        #     return features[:, 0], features[:, 1]
        if self.dist_token is None:
            return self.pre_logits(x[:, 1:])
        else:
            return self.pre_logits(x[:, 2:])


def adapt_image_pos_embed_to_passt(posemb, num_tokens=1, gs_new=(), mode="bicubic"):
    """
    Adapt a 2D positional embedding from an image model to the PaSST spectrogram.

    Args:
        posemb (torch.Tensor): Original positional embeddings (1, num_patches + num_tokens, dim).
        num_tokens (int): Number of special tokens (class/distillation).
        gs_new (tuple): New grid size (freq, time) for the spectrogram.
        mode (str): Interpolation mode for resizing ('bicubic', 'bilinear', etc.).

    Returns:
        tuple: (posemb_tok, freq_new_pos_embed, time_new_pos_embed)
            - posemb_tok: Token embeddings.
            - freq_new_pos_embed: Frequency-wise positional embedding.
            - time_new_pos_embed: Time-wise positional embedding.
    """

    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

    gs_old = int(math.sqrt(len(posemb_grid)))

    if len(gs_new) < 2:
        raise ValueError(f"Variable gs_old can not be smaller than 2, given value: {gs_new}")

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode=mode, align_corners=False)
    freq_new_pos_embed = posemb_grid.mean(dim=3, keepdim=True)
    time_new_pos_embed = posemb_grid.mean(dim=2, keepdim=True)

    return posemb_tok, freq_new_pos_embed, time_new_pos_embed


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=(), mode="bicubic"):
    """
    Resize the positional embeddings to match a new model configuration.

    Used when loading pretrained weights from a different input resolution.

    Args:
        posemb (torch.Tensor): Original positional embeddings.
        posemb_new (torch.Tensor): Target positional embeddings shape.
        num_tokens (int): Number of special tokens.
        gs_new (tuple): New grid size (freq, time). If empty, inferred from target.
        mode (str): Interpolation mode for resizing.

    Returns:
        torch.Tensor: Resized positional embeddings.
    """
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode=mode, align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

    return posemb
