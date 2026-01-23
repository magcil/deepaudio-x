# --------------------------------------------------------
# BEATs: Audio Pre-Training with Acoustic Tokenizers (https://arxiv.org/abs/2212.09058)
# Github source: https://github.com/microsoft/unilm/tree/master/beats
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import LayerNorm, Parameter

from deepaudiox.modules.backbones.beats.beats_modules.modules import (
    GLU_Linear,
    GradMultiply,
    SamePad,
    get_activation_fn,
    quant_noise,
)


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder module for BEATs.

    Applies convolutional positional encoding followed by multiple
    TransformerSentenceEncoderLayer layers.

    Args:
        args: Namespace or object containing model hyperparameters such as:
            - encoder_embed_dim
            - encoder_ffn_embed_dim
            - encoder_attention_heads
            - encoder_layers
            - dropout
            - attention_dropout
            - activation_dropout
            - activation_fn
            - layer_norm_first
            - deep_norm
            - relative_position_embedding
            - num_buckets
            - max_distance
            - gru_rel_pos
            - encoder_layerdrop
    """

    def __init__(self, args):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        if hasattr(args, "relative_position_embedding"):
            self.relative_position_embedding = args.relative_position_embedding
            self.num_buckets = args.num_buckets
            self.max_distance = args.max_distance
        else:
            self.relative_position_embedding = False
            self.num_buckets = 0
            self.max_distance = 0

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                    deep_norm=args.deep_norm,
                    has_relative_attention_bias=self.relative_position_embedding,
                    num_buckets=self.num_buckets,
                    max_distance=self.max_distance,
                    gru_rel_pos=args.gru_rel_pos,
                    encoder_layers=args.encoder_layers,
                )
                for i in range(args.encoder_layers)
            ]
        )
        if self.relative_position_embedding:
            for i in range(1, args.encoder_layers):
                del self.layers[i].self_attn.relative_attention_bias
                self.layers[i].self_attn.relative_attention_bias = self.layers[0].self_attn.relative_attention_bias

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

        if args.deep_norm:
            deep_norm_beta = math.pow(8 * args.encoder_layers, -1 / 4)
            for i in range(args.encoder_layers):
                nn.init.xavier_normal_(self.layers[i].self_attn.k_proj.weight, gain=1)
                nn.init.xavier_normal_(self.layers[i].self_attn.v_proj.weight, gain=deep_norm_beta)
                nn.init.xavier_normal_(self.layers[i].self_attn.q_proj.weight, gain=1)
                nn.init.xavier_normal_(self.layers[i].self_attn.out_proj.weight, gain=deep_norm_beta)
                nn.init.xavier_normal_(self.layers[i].fc1.weight, gain=deep_norm_beta)
                nn.init.xavier_normal_(self.layers[i].fc2.weight, gain=deep_norm_beta)

        self.layer_wise_gradient_decay_ratio = getattr(args, "layer_wise_gradient_decay_ratio", 1)

    def forward(self, x, padding_mask=None, layer=None):
        """
        Forward pass through the encoder.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            padding_mask (Tensor, optional): Boolean mask of shape (batch_size, seq_len), where True
                values indicate padding positions to ignore.
            layer (int, optional): If specified, returns features at this specific layer.

        Returns:
            Tuple:
                - Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
                - List of tuples: Intermediate layer outputs as (x, pos_bias) for each layer.
        """
        x, layer_results = self.extract_features(x, padding_mask, layer)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(self, x, padding_mask=None, tgt_layer=None):
        """
        Extract features from input through convolutional positional encoding and Transformer layers.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            padding_mask (Tensor, optional): Boolean mask for padding tokens.
            tgt_layer (int, optional): If specified, stops computation at this layer.

        Returns:
            Tuple:
                - Tensor: Output features of shape (batch_size, seq_len, embed_dim).
                - List of tuples: Intermediate outputs (x, pos_bias) for each layer.
        """
        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        z = None
        if tgt_layer is not None:
            layer_results.append((x, z))
        r = None
        pos_bias = None
        for i, layer in enumerate(self.layers):
            if self.layer_wise_gradient_decay_ratio != 1.0:
                x = GradMultiply.apply(x, self.layer_wise_gradient_decay_ratio)
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z, pos_bias = layer(x, self_attn_padding_mask=padding_mask, need_weights=False, pos_bias=pos_bias)
            if tgt_layer is not None:
                layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer with optional GLU feedforward network.

    Consists of multi-headed self-attention, optional relative positional bias,
    feedforward network (FC + activation + FC), dropout, and layer normalization.

    Args:
        embedding_dim (int): Dimension of input embeddings.
        ffn_embedding_dim (int): Hidden dimension of feedforward network.
        num_attention_heads (int): Number of attention heads.
        dropout (float): Dropout probability for attention output.
        attention_dropout (float): Dropout probability within attention softmax.
        activation_dropout (float): Dropout probability for feedforward activations.
        activation_fn (str): Activation function for feedforward network. Options: "relu", "gelu", "tanh", "glu".
        layer_norm_first (bool): If True, applies layer norm before self-attention.
        deep_norm (bool): If True, applies DeepNorm scaling.
        has_relative_attention_bias (bool): Whether to use relative positional bias.
        num_buckets (int): Number of buckets for relative positional bias.
        max_distance (int): Maximum relative distance for bias.
        rescale_init (bool): If True, disables bias in Q/K linear layers.
        gru_rel_pos (bool): If True, uses GRU-based relative positional encoding.
        encoder_layers (int): Total number of encoder layers for DeepNorm.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        deep_norm: bool = False,
        has_relative_attention_bias: bool = False,
        num_buckets: int = 0,
        max_distance: int = 0,
        rescale_init: bool = False,
        gru_rel_pos: bool = False,
        encoder_layers: int = 0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        self.activation_name = activation_fn
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            has_relative_attention_bias=has_relative_attention_bias,
            num_buckets=num_buckets,
            max_distance=max_distance,
            rescale_init=rescale_init,
            gru_rel_pos=gru_rel_pos,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)

        if self.activation_name == "glu":
            self.fc1 = GLU_Linear(self.embedding_dim, ffn_embedding_dim, "swish")
        else:
            self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        self.final_layer_norm = LayerNorm(self.embedding_dim)

        self.deep_norm = deep_norm
        if self.deep_norm:
            self.deep_norm_alpha = math.pow(2 * encoder_layers, 1 / 4)
        else:
            self.deep_norm_alpha = 1

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        pos_bias=None,
    ):
        """
        Forward pass through one Transformer encoder layer.

        Args:
            x (Tensor): Input tensor of shape (seq_len, batch_size, embed_dim).
            self_attn_mask (Tensor, optional): Optional attention mask.
            self_attn_padding_mask (Tensor, optional): Optional padding mask of shape (batch_size, seq_len).
            need_weights (bool): If True, return attention weights.
            pos_bias (Tensor, optional): Precomputed relative positional bias.

        Returns:
            Tuple:
                - Tensor: Output tensor of same shape as input (seq_len, batch_size, embed_dim).
                - Tensor | None: Attention weights (if `need_weights=True`), else None.
                - Tensor | None: Position bias after attention computation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
                position_bias=pos_bias,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.fc1(x) if self.activation_name == "glu" else self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                attn_mask=self_attn_mask,
                position_bias=pos_bias,
            )

            x = self.dropout1(x)
            x = residual * self.deep_norm_alpha + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.fc1(x) if self.activation_name == "glu" else self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual * self.deep_norm_alpha + x
            x = self.final_layer_norm(x)

        return x, attn, pos_bias


class MultiheadAttention(nn.Module):
    """
    Multi-headed attention module with optional relative positional encoding and incremental state.

    Supports self-attention and encoder-decoder attention. Can integrate
    relative positional bias and GRU-based relative position embeddings.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        has_relative_attention_bias=False,
        num_buckets=32,
        max_distance=128,
        gru_rel_pos=False,
        rescale_init=False,
    ):
        """
        Args:
            embed_dim (int): Dimension of input embeddings.
            num_heads (int): Number of attention heads.
            kdim (int, optional): Dimension of key embeddings. Defaults to embed_dim.
            vdim (int, optional): Dimension of value embeddings. Defaults to embed_dim.
            dropout (float): Dropout probability for attention output.
            bias (bool): If True, add bias to linear projections.
            add_bias_kv (bool): If True, adds learnable bias to key and value.
            add_zero_attn (bool): If True, adds a zero attention vector.
            self_attention (bool): Whether this is self-attention.
            encoder_decoder_attention (bool): Whether this is encoder-decoder attention.
            q_noise (float): Probability of quantization noise for linear projections.
            qn_block_size (int): Block size for quantization noise.
            has_relative_attention_bias (bool): If True, uses relative positional bias.
            num_buckets (int): Number of buckets for relative positional bias.
            max_distance (int): Maximum relative distance for bias.
            gru_rel_pos (bool): If True, use GRU-based relative position encoding.
            rescale_init (bool): If True, disables bias in Q/K linear layers.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)

        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

        self.head_dim = embed_dim // num_heads
        self.q_head_dim = self.head_dim
        self.k_head_dim = self.head_dim

        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f"embed_dim={self.embed_dim} must be divisible by num_heads={num_heads}")

        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        if self.self_attention and not self.qkv_same_dim:
            raise ValueError("Self-attention requires query, key and value to be of the same size")

        k_bias = True
        if rescale_init:
            k_bias = False

        k_embed_dim = embed_dim
        q_embed_dim = embed_dim

        self.k_proj = quant_noise(nn.Linear(self.kdim, k_embed_dim, bias=k_bias), q_noise, qn_block_size)
        self.v_proj = quant_noise(nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.q_proj = quant_noise(nn.Linear(embed_dim, q_embed_dim, bias=bias), q_noise, qn_block_size)

        self.out_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.gru_rel_pos = gru_rel_pos
        if self.gru_rel_pos:
            self.grep_linear = nn.Linear(self.q_head_dim, 8)
            self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters of the attention module.

        Uses Xavier uniform initialization for linear layers.
        If Q/K/V dimensions are the same, scaled initialization is applied.
        Relative positional embeddings, bias vectors, and output projections
        are initialized appropriately.
        """
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
        if self.has_relative_attention_bias:
            nn.init.xavier_normal_(self.relative_attention_bias.weight)

    def _relative_positions_bucket(self, relative_positions, bidirectional=True):
        """
        Convert relative positions to bucket indices for relative positional bias.

        Args:
            relative_positions (Tensor): Tensor of relative positions (key_pos - query_pos).
            bidirectional (bool, optional): Whether attention is bidirectional. Defaults to True.

        Returns:
            Tensor: Bucket indices for each relative position, shape same as `relative_positions`.
        """
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        relative_buckets = 0

        if bidirectional:
            num_buckets = num_buckets // 2
            relative_buckets += (relative_positions > 0).to(torch.long) * num_buckets
            relative_positions = torch.abs(relative_positions)
        else:
            relative_positions = -torch.min(relative_positions, torch.zeros_like(relative_positions))

        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact

        relative_postion_if_large = max_exact + (
            torch.log(relative_positions.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_positions, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """
        Compute the relative positional bias tensor for attention.

        Args:
            query_length (int): Length of the query sequence.
            key_length (int): Length of the key sequence.

        Returns:
            Tensor: Positional bias of shape (num_heads, query_length, key_length).
        """
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(relative_position, bidirectional=True)
        relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1])
        return values

    def forward(
        self,
        query,
        key: Tensor | None,
        value: Tensor | None,
        key_padding_mask: Tensor | None = None,
        incremental_state: dict[str, dict[str, Tensor | None]] | None = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Tensor | None = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        position_bias: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """
        Compute multi-head attention.

        Args:
            query (Tensor): Query tensor of shape (tgt_len, batch_size, embed_dim).
            key (Tensor, optional): Key tensor of shape (src_len, batch_size, kdim). Defaults to None.
            value (Tensor, optional): Value tensor of shape (src_len, batch_size, vdim). Defaults to None.
            key_padding_mask (Tensor, optional): Mask for padding positions (batch, src_len). Defaults to None.
            incremental_state (dict, optional): Cached state for incremental decoding. Defaults to None.
            need_weights (bool, optional): Return attention weights averaged over heads. Defaults to True.
            static_kv (bool, optional): Use static key/value for incremental decoding. Defaults to False.
            attn_mask (Tensor, optional): Mask for causal attention (tgt_len, src_len). Defaults to None.
            before_softmax (bool, optional): Return raw attention scores. Defaults to False.
            need_head_weights (bool, optional): Return attention weights for each head. Defaults to False.
            position_bias (Tensor, optional): Precomputed positional bias tensor. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor | None, Tensor | None]:
                - Attention output (tgt_len, batch_size, embed_dim)
                - Attention weights (batch_size, tgt_len, src_len) if `need_weights=True`, else None
                - Positional bias tensor used in attention
        """
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        if embed_dim != self.embed_dim:
            raise ValueError(f"embed_dim={embed_dim} does not match self.embed_dim={self.embed_dim}")
        if list(query.size()) != [tgt_len, bsz, embed_dim]:
            raise ValueError(f"Expected query size {[tgt_len, bsz, embed_dim]}, but got {list(query.size())}")
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                if key_bsz != bsz:
                    raise ValueError(f"key_bsz={key_bsz} does not match batch size bsz={bsz}")

                if value is None:
                    raise ValueError("Value tensor must not be None")

                if (src_len, bsz) != tuple(value.shape[:2]):
                    raise ValueError(
                        f"Expected value shape first two dimensions to be ({src_len}, {bsz}), "
                        f"but got {tuple(value.shape[:2])}"
                    )

        if self.has_relative_attention_bias and position_bias is None:
            position_bias = self.compute_bias(tgt_len, src_len)
            position_bias = position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).view(bsz * self.num_heads, tgt_len, src_len)

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state and static_kv:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if self.encoder_decoder_attention and self.self_attention:
                    raise ValueError(
                        "encoder_decoder_attention and self_attention cannot both be True at the same time"
                    )
                key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                if value is not None:
                    raise ValueError("Expected value to be None")
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            if key is None or value is None:
                raise ValueError("Both key and value tensors must not be None")
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling
        alpha = 32
        q *= 1 / alpha

        if self.bias_k is not None:
            if self.bias_v is None:
                raise ValueError("bias_v must not be None")
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.q_head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.k_head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                if _prev_key is None:
                    raise ValueError("_prev_key must not be None")

                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    if k is None:
                        raise ValueError("k must not be None")
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                if _prev_value is None:
                    raise ValueError("_prev_value must not be None")

                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    if v is None:
                        raise ValueError("v must not be None")
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Tensor | None = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            if k is None or v is None:
                raise ValueError("Both k and v must not be None")
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            if incremental_state is None:
                raise ValueError("incremental_state must not be None")
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        if k is None:
            raise ValueError("k must not be None")

        if k.size(1) != src_len:
            raise ValueError(f"Expected k.size(1) to be {src_len}, but got {k.size(1)}")

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            if key_padding_mask.size(0) != bsz:
                raise ValueError(f"Expected key_padding_mask.size(0) to be {bsz}, but got {key_padding_mask.size(0)}")

            if key_padding_mask.size(1) != src_len:
                raise ValueError(
                    f"Expected key_padding_mask.size(1) to be {src_len}, but got {key_padding_mask.size(1)}"
                )

        if self.add_zero_attn:
            if v is None:
                raise ValueError("v must not be None")
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask),
                    ],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = (attn_weights - attn_weights.max(dim=-1, keepdim=True)[0]) * alpha
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        expected_shape = [bsz * self.num_heads, tgt_len, src_len]
        if list(attn_weights.size()) != expected_shape:
            raise ValueError(f"Expected attn_weights shape {expected_shape}, but got {list(attn_weights.size())}")

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v, position_bias

        if position_bias is not None:
            attn_mask_rel_pos = position_bias
            if self.gru_rel_pos == 1:
                query_layer = q.view(bsz, self.num_heads, tgt_len, self.q_head_dim) * alpha / self.scaling
                _B, _H, _L, __ = query_layer.size()
                gate_a, gate_b = torch.sigmoid(
                    self.grep_linear(query_layer).view(_B, _H, _L, 2, 4).sum(-1, keepdim=False)
                ).chunk(2, dim=-1)
                gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                attn_mask_rel_pos = gate_a_1.view(bsz * self.num_heads, tgt_len, 1) * position_bias

            attn_mask_rel_pos = attn_mask_rel_pos.view(attn_weights.size())

            attn_weights = attn_weights + attn_mask_rel_pos

        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        if v is None:
            raise ValueError("v must not be None")

        attn = torch.bmm(attn_probs, v)
        expected_shape = [bsz * self.num_heads, tgt_len, self.head_dim]
        if list(attn.size()) != expected_shape:
            raise ValueError(f"Expected attn shape {expected_shape}, but got {list(attn.size())}")
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Tensor | None = None
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights, position_bias

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Tensor | None,
        prev_key_padding_mask: Tensor | None,
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Tensor | None:
        """
        Combine current and previous key padding masks for incremental attention.

        Args:
            key_padding_mask (Tensor | None): Current key padding mask.
            prev_key_padding_mask (Tensor | None): Previous key padding mask.
            batch_size (int): Batch size.
            src_len (int): Total sequence length of keys.
            static_kv (bool): Whether keys are static (don't change during incremental decoding).

        Returns:
            Tensor | None: Combined key padding mask of shape (batch_size, src_len) or None.
        """
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), key_padding_mask.float()], dim=1)
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - prev_key_padding_mask.size(1)),
                    device=prev_key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), filler.float()], dim=1)
            else:
                new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - key_padding_mask.size(1)),
                    device=key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat([filler.float(), key_padding_mask.float()], dim=1)
            else:
                new_key_padding_mask = key_padding_mask.float()
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def _get_input_buffer(
        self, incremental_state: dict[str, dict[str, Tensor | None]] | None
    ) -> dict[str, Tensor | None]:
        """
        Retrieve the attention cache from incremental_state.

        Args:
            incremental_state (dict, optional): Dictionary holding cached states.

        Returns:
            dict: Cached attention state. Empty dict if no state exists.
        """
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: dict[str, Tensor | None] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: dict[str, dict[str, Tensor | None]],
        buffer: dict[str, Tensor | None],
    ):
        """
        Store the attention cache into incremental_state.

        Args:
            incremental_state (dict): Dictionary holding cached states.
            buffer (dict): Attention cache to store.

        Returns:
            dict: Updated incremental_state dictionary.
        """
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        """
        Placeholder for applying sparse attention masks. Currently a no-op.

        Args:
            attn_weights (Tensor): Attention weights before masking.
            tgt_len (int): Length of target sequence.
            src_len (int): Length of source sequence.
            bsz (int): Batch size.

        Returns:
            Tensor: Potentially masked attention weights.
        """
        return attn_weights


def init_bert_params(module):
    """
    Initialize the parameters of BERT-related modules.

    This function customizes the initialization of Linear, Embedding, and
    MultiheadAttention layers for BERT-like models. It overrides PyTorch's
    default initialization to follow standard BERT practices.

    Behavior:
        1. Linear layers: Weights are initialized from a normal distribution
           (mean=0, std=0.02), and biases are set to zero.
        2. Embedding layers: Weights are initialized from a normal distribution
           (mean=0, std=0.02). If `padding_idx` is specified, the corresponding
           embedding is zeroed.
        3. MultiheadAttention layers: The `q_proj`, `k_proj`, and `v_proj`
           linear projections are initialized from a normal distribution
           (mean=0, std=0.02).

    Args:
        module (nn.Module): PyTorch module to initialize. Can be an instance of
            nn.Linear, nn.Embedding, or MultiheadAttention.

    Notes:
        - This function is often used with `model.apply(init_bert_params)`
          to recursively initialize all submodules of a model.
        - The nested helper `normal_` ensures reproducibility on CPU and CUDA
          by temporarily moving tensors to CPU for random initialization and
          then restoring the original device.
    """

    def normal_(data):
        """
        Initialize a tensor with values sampled from a normal distribution
        (mean=0, std=0.02) while preserving device placement.

        Args:
            data (Tensor): Tensor to be initialized.
        """
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)
