# --------------------------------------------------------
# BEATs: Audio Pre-Training with Acoustic Tokenizers (https://arxiv.org/abs/2212.09058)
# Github source: https://github.com/microsoft/unilm/tree/master/beats
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------

import torch
import torch.nn as nn
import torchaudio.compliance.kaldi as ta_kaldi
from torch.nn import LayerNorm

from deepaudiox.modules.backbones.beats.beats_modules.backbone import (
    TransformerEncoder,
)
from deepaudiox.modules.baseclasses import BaseBackbone


class BEATsConfig:
    """Configuration class for BEATs audio backbone.

    Holds model hyperparameters, architecture settings, and preprocessing options.

    Attributes:
        input_patch_size (int): Patch size for patch embedding.
        embed_dim (int): Dimension of patch embedding.
        conv_bias (bool): Whether to include bias in convolutional encoder.
        encoder_layers (int): Number of transformer encoder layers.
        encoder_embed_dim (int): Transformer embedding dimension.
        encoder_ffn_embed_dim (int): Transformer feed-forward dimension.
        encoder_attention_heads (int): Number of attention heads in the transformer.
        activation_fn (str): Activation function to use in the transformer.
        layer_wise_gradient_decay_ratio (float): Decay ratio for layer-wise gradient decay.
        layer_norm_first (bool): Whether to apply LayerNorm first in transformer.
        deep_norm (bool): Whether to apply DeepNorm.
        dropout (float): Transformer dropout probability.
        attention_dropout (float): Dropout for attention weights.
        activation_dropout (float): Dropout after activation in FFN.
        encoder_layerdrop (float): Probability of dropping a transformer layer.
        dropout_input (float): Dropout applied to input features.
        conv_pos (int): Number of filters for convolutional positional embeddings.
        conv_pos_groups (int): Number of groups for convolutional positional embedding.
        relative_position_embedding (bool): Apply relative positional embeddings.
        num_buckets (int): Number of buckets for relative positional embedding.
        max_distance (int): Maximum distance for relative positional embedding.
        gru_rel_pos (bool): Apply gated relative positional embedding.
        finetuned_model (bool): Whether the model is fine-tuned.
        predictor_dropout (float): Dropout probability for the predictor.
        predictor_class (int): Number of target classes for predictor.
    """

    def __init__(self, cfg=None):
        self.input_patch_size: int = 16
        self.embed_dim: int = 512
        self.conv_bias: bool = False
        self.encoder_layers: int = 12
        self.encoder_embed_dim: int = 768
        self.encoder_ffn_embed_dim: int = 3072
        self.encoder_attention_heads: int = 12
        self.activation_fn: str = "gelu"
        self.layer_wise_gradient_decay_ratio: float = 0.6
        self.layer_norm_first: bool = False
        self.deep_norm: bool = True
        self.dropout: float = 0.1
        self.attention_dropout: float = 0.1
        self.activation_dropout: float = 0.0
        self.encoder_layerdrop: float = 0.05
        self.dropout_input: float = 0.0
        self.conv_pos: int = 128
        self.conv_pos_groups: int = 16
        self.relative_position_embedding: bool = True
        self.num_buckets: int = 320
        self.max_distance: int = 800
        self.gru_rel_pos: bool = True
        self.finetuned_model: bool = False
        self.predictor_dropout: float = 0.1
        self.predictor_class: int = 527

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        """Update configuration with a dictionary of values.

        Args:
            cfg (dict): Dictionary containing configuration overrides.
        """
        self.__dict__.update(cfg)


class BEATs(BaseBackbone):
    """BEATs audio backbone.

    Implements the BEATs model for audio representation learning, including
    patch embedding, transformer encoder, and optional classifier.

    Args:
        cfg (BEATsConfig | None): Configuration object for model parameters.
            If None, a default BEATsConfig is used.
        preprocess_flag (bool): Whether to perform preprocessing on input waveforms.
        sample_rate (int): Sampling rate of input audio.
    """

    def __init__(self, cfg: BEATsConfig | None = None, preprocess_flag: bool = True, sample_rate: int = 16_000) -> None:
        super().__init__(out_dim=768, sample_rate=sample_rate)

        if cfg is None:
            self.cfg = BEATsConfig()
        else:
            self.cfg = cfg
        self.preprocess_flag: bool = preprocess_flag

        self.fbank_mean, self.fbank_std = 15.41663, 6.55582

        self.embed = self.cfg.embed_dim
        self.post_extract_proj = (
            nn.Linear(self.embed, self.cfg.encoder_embed_dim) if self.embed != self.cfg.encoder_embed_dim else None
        )

        self.input_patch_size = self.cfg.input_patch_size
        self.patch_embedding = nn.Conv2d(
            1, self.embed, kernel_size=self.input_patch_size, stride=self.input_patch_size, bias=self.cfg.conv_bias
        )

        self.dropout_input = nn.Dropout(self.cfg.dropout_input)

        if self.cfg.deep_norm and self.cfg.layer_norm_first:
            raise ValueError("deep_norm and layer_norm_first cannot both be True at the same time")

        self.encoder = TransformerEncoder(self.cfg)
        self.layer_norm = LayerNorm(self.embed)

        if self.cfg.finetuned_model:
            self.predictor_dropout = nn.Dropout(self.cfg.predictor_dropout)
            self.predictor = nn.Linear(self.cfg.encoder_embed_dim, self.cfg.predictor_class)
        else:
            self.predictor = None

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a padding mask aligned with feature sequence length.

        Args:
            features (torch.Tensor): Feature tensor.
            padding_mask (torch.Tensor): Input padding mask.

        Returns:
            torch.Tensor: Adjusted padding mask,
                          True for valid tokens, False for padded positions.
        """
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def extract_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Convert raw waveforms to normalized log-Mel filterbank features.

        Args:
            waveforms (torch.Tensor): Input audio tensor.

        Returns:
            torch.Tensor: Filterbank features normalized by mean and std.
        """
        fbanks = []
        for waveform in waveforms:
            waveform = waveform.unsqueeze(0) * 2**15
            fbank = ta_kaldi.fbank(
                waveform, num_mel_bins=128, sample_frequency=self.sample_rate, frame_length=25, frame_shift=10
            )
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        fbank = (fbank - self.fbank_mean) / (2 * self.fbank_std)
        return fbank.unsqueeze(1)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None):
        """Forward pass through BEATs backbone.

        Args:
            x (torch.Tensor): Input tensor.
            padding_mask (torch.Tensor | None): Optional padding mask.

        Returns:
            torch.Tensor: Output feature tensor after transformer encoding.
        """
        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(x, padding_mask)

        features = self.patch_embedding(x)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        x = self.dropout_input(features)

        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
        )

        return x
