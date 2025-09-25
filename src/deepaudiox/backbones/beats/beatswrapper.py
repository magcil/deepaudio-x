
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .beats_modules.BEATs import BEATs, BEATsConfig

warnings.filterwarnings(
    "ignore",
    message="`torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`",
    category=FutureWarning,
)

MODEL_CONFIG = {
    "encoder_layers": 12,
    "encoder_embed_dim": 768,
    "encoder_ffn_embed_dim": 3072,
    "encoder_attention_heads": 12,
    "activation_fn": "gelu",
    "dropout": 0.0,
    "attention_dropout": 0.0,
    "activation_dropout": 0.0,
    "encoder_layerdrop": 0.05,
    "dropout_input": 0.0,
    "layer_norm_first": False,
    "conv_bias": False,
    "conv_pos": 128,
    "conv_pos_groups": 16,
    "relative_position_embedding": True,
    "num_buckets": 320,
    "max_distance": 800,
    "gru_rel_pos": True,
    "deep_norm": True,
    "input_patch_size": 16,
    "layer_wise_gradient_decay_ratio": 0.6,
    "embed_dim": 512,
    "finetuned_model": False,
}


class DivEncLayer(nn.Module):
    # Divided Encoder Layer for dimensionality reduction
    def __init__(self, q: int, v: int, unit_dim: list[int] | None = None) -> None:
        """Divided Encoder Layer for dimensionality reduction.
        Args:
            q (int): Number of splits.
            v (int): Dimension of each split.
            unit_dim (list): List containing the dimensions of the two linear layers.
        """
        if unit_dim is None:
            unit_dim = [32, 1]
        super().__init__()
        self.split_fc_layers: nn.ModuleList = nn.ModuleList()
        self.q: int = q
        self.unit_dim: list[int] = unit_dim
        self.v: int = v
        self._construct_layers()

    def _construct_layers(self) -> None:
        for _i in range(self.q):
            seq = nn.Sequential()
            seq.append(nn.Linear(self.v, self.unit_dim[0]))
            seq.append(nn.ELU())
            seq.append(nn.LayerNorm(self.unit_dim[0]))
            seq.append(nn.Linear(self.unit_dim[0], self.unit_dim[1]))
            self.split_fc_layers.append(seq)

    def _split_encoding(self, x_slices: torch.Tensor) -> torch.Tensor:
        out: list[torch.Tensor] = []
        for i in range(self.q):
            out.append(self.split_fc_layers[i](x_slices[:, i, :]))
        return torch.concat(out, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: BxD, D=1024
        x = torch.reshape(x, (x.shape[0], self.q, -1))
        return self._split_encoding(x)


class BEATsBackbone(nn.Module):
    # Initialize BEATs Model
    def __init__(
        self,
        backbone_config: dict = MODEL_CONFIG,
        div_encoder_layer: bool = True,
        sample_frequency: int = 16000,
    ) -> None:
        super().__init__()
        """A wrapper for BEATs model to be used as a backbone in other models.
        Args:
            backbone_config (Dict): Configuration dictionary for BEATs model.
            div_encoder_layer (bool): Whether to use DivEncLayer for dimensionality reduction.
            sample_frequency (int): Sample frequency for audio input.
        """
        # Initialize BEATs Encoder
        cfg = BEATsConfig(cfg=backbone_config)
        self.sample_frequency: int = sample_frequency
        self.encoder: BEATs = BEATs(cfg=cfg, preprocess_flag=True)
        self.div_encoder_layer: bool = div_encoder_layer

        if div_encoder_layer:
            self.projection_head: DivEncLayer = DivEncLayer(q=128, v=int(768 / 128))

    def load_pretrained_encoder(self, weights: str) -> None:
        self.encoder.load_state_dict(torch.load(weights, weights_only=True))

    def freeze_encoder_weights(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder.preprocess(x, sample_frequency=self.sample_frequency)
        x = x.unsqueeze(1)
        # x: B x 1 x T x F
        x = self.encoder(x)
        # x: B x N x 768
        x = x.mean(1)
        # x: B x 768
        if self.div_encoder_layer:
            return F.normalize(self.projection_head(x), p=2.0)
        else:
            return F.normalize(x, p=2.0)
