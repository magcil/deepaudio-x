import warnings

import torch

from deepaudiox.modules.backbones.base_backbone import BaseBackbone

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


class BEATsBackbone(BaseBackbone):
    # Initialize BEATs Model
    def __init__(
        self,
        backbone_config: dict = MODEL_CONFIG,
        sample_frequency: int = 16000,
    ) -> None:
        super().__init__(out_dim=768, sample_frequency=sample_frequency)
        """A wrapper for BEATs model to be used as a backbone in other models.
        Args:
            backbone_config (Dict): Configuration dictionary for BEATs model.
            div_encoder_layer (bool): Whether to use DivEncLayer for dimensionality reduction.
            sample_frequency (int): Sample frequency for audio input.
        """
        # Initialize BEATs Encoder
        cfg = BEATsConfig(cfg=backbone_config)
        self.encoder: BEATs = BEATs(cfg=cfg, preprocess_flag=True)

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

        return x
