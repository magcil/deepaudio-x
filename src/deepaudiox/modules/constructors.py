from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepaudiox.modules.backbones import BACKBONES
from deepaudiox.modules.baseclasses import BaseAudioClassifier, BaseBackbone, BasePooling
from deepaudiox.modules.classifier.classifier import MLPHead
from deepaudiox.modules.pooling import GAP, POOLING
from deepaudiox.utils.downloader import Downloader
from deepaudiox.utils.file_utils import load_checkpoint


class BackbonePoolingResolverMixin:
    """Mixin providing helper methods to resolve backbones and pooling modules.

    This mixin centralizes the logic for:
      - Instantiating a backbone by name or using a provided BaseBackbone instance.
      - Optionally loading pretrained weights.
      - Resolving a pooling module by name, instance, or defaulting to GAP.

    Notes:
        - Valid backbone names are those registered in `deepaudiox.modules.backbones`.
        - Pooling is resolved using `deepaudiox.modules.pooling.POOLING`.
    """

    def _resolve_backbone(
        self,
        backbone: Literal["beats", "passt"] | BaseBackbone,
        pretrained: bool,
        sample_rate: int,
    ) -> BaseBackbone:
        """Resolve backbone from literal or BaseBackbone instance.

        Args:
            backbone (Literal["beats", "passt"] | BaseBackbone): Backbone name or instance.
            pretrained (bool): Whether to load pretrained weights.
            sample_rate (int): Sample rate for the backbone.

        Returns:
            BaseBackbone: Initialized backbone model.
        """
        if isinstance(backbone, str):
            model = BACKBONES[backbone]()
            if pretrained:
                downloader = Downloader()
                ckpt_path = downloader.download_checkpoint(backbone)
                ckpt = load_checkpoint(ckpt_path)
                model.load_state_dict(ckpt)
        else:
            model = backbone

        model.sample_rate = sample_rate
        return model

    def _resolve_pooling(
        self,
        pooling: Literal["gap", "simpool", "ep"] | BasePooling,
        out_dim: int,
    ) -> BasePooling:
        """Resolve pooling layer from literal, BasePooling instance, or None.

        Args:
            pooling (Literal["gap", "simpool", "ep"] | BasePooling | None): Pooling layer name, instance, or None.
            out_dim (int): Backbone output dimension used to configure pooling modules.

        Returns:
            BasePooling: Initialized pooling layer.
        """
        if isinstance(pooling, str):
            return POOLING[pooling](dim=out_dim)
        return pooling


class BackboneConstructor(nn.Module, BackbonePoolingResolverMixin):
    """Backbone model wrapper with optional pooling and normalization.

    Attributes:
        backbone (BaseBackbone): Backbone model for feature extraction.
        pooling (BasePooling): Pooling layer applied to the backbone feature map.
        norm_p (float or None): Optional Lp normalization applied after pooling.
        out_dim (int): Dimension of the backbone model feature map
    """

    def __init__(
        self,
        backbone: Literal["beats", "passt"] | BaseBackbone,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        pooling: Literal["gap", "simpool", "ep"] | BasePooling | None = None,
        sample_rate: int = 16_000,
        norm_p: float | None = None,
    ):
        """Initialize the BackboneConstructor.

        Args:
            backbone (Literal["beats", "passt"] | BaseBackbone): Backbone name or instance.
            pretrained (bool): Whether to load pretrained weights for the backbone.
            freeze_backbone (bool): Whether to freeze the backbone weights during training.
            pooling (Literal["gap", "simpool", "ep"] | BasePooling | None): Optional pooling layer for aggregation.
            sample_rate (int): Sample frequency for audio input.
            norm_p (float or None): Optional Lp norm applied after pooling. If pooling is None, GAP is used.

        Example:
            >>> from deepaudiox import Backbone
            >>> backbone = Backbone(
            ...     backbone="beats",
            ...     pretrained=True,
            ...     freeze_backbone=True,
            ...     pooling="gap",
            ...     sample_rate=16000,
            ...     norm_p=2.0,
            ... )
        """
        super().__init__()
        self.backbone = self._resolve_backbone(
            backbone=backbone, pretrained=pretrained, sample_rate=sample_rate
        )  # Resolve backbone
        if pooling is not None:
            self.pooling = self._resolve_pooling(pooling=pooling, out_dim=self.backbone.out_dim)  # Resolve Pooling
        else:
            self.pooling = GAP()
        self.norm_p = norm_p  # Parameter to apply Lp norm after pooling
        self.out_dim = self.backbone.out_dim  # Out dim of backbone

        # Freeze backbone's weights
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def extract_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Extract backbone-specific features from raw waveforms.

        Args:
            waveforms (torch.Tensor): Input waveforms of shape (B, T).

        Returns:
            torch.Tensor: Model-specific feature map, either (B, N, D) or (B, D, H, W).
        """
        return self.backbone.extract_features(waveforms)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through the backbone.

        Args:
            x (torch.Tensor): Input waveforms of shape (B, T).
            padding_mask (torch.Tensor or None): Optional padding mask (unused).

        Returns:
            torch.Tensor: Backbone feature map of shape (B, N, D) or (B, D, H, W).

        Example:
            >>> import torch
            >>> from deepaudiox import Backbone
            >>> backbone = Backbone(backbone="beats", pretrained=True, sample_rate=16_000)
            >>> waveforms = torch.randn(2, 5 * 16_000)
            >>> features = backbone.forward(waveforms)
            >>> # features shape: (B, N, D) for Transformer or (B, D, H, W) for CNN backbones
        """
        return self.backbone.forward_pipeline(x)

    def forward_with_pooling(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through backbone and pooling (with optional normalization).

        Args:
            x (torch.Tensor): Input waveforms of shape (B, T).
            padding_mask (torch.Tensor or None): Optional padding mask (unused).

        Returns:
            torch.Tensor: Pooled tensor of shape (B, D).

        Example:
            >>> import torch
            >>> from deepaudiox import Backbone
            >>> backbone = Backbone(backbone="beats", pretrained=True, pooling="gap", sample_rate=16_000)
            >>> waveforms = torch.randn(2, 5 * 16_000)
            >>> embeddings = backbone.forward_with_pooling(waveforms)
            >>> # embeddings shape: (B, D)
        """
        x = self.forward(x)
        if self.norm_p:
            return F.normalize(self.pooling(x), p=self.norm_p)
        else:
            return self.pooling(x)


class AudioClassifierConstructor(BaseAudioClassifier, BackbonePoolingResolverMixin):
    """Classifier model using a backbone for feature extraction.

    Attributes:
        backbone (BaseBackboneConstructor): Backbone model with optional pooling method.
        classifier (MLPHead): Classifier head for final predictions.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: Literal["beats", "passt"] | BaseBackbone,
        pooling: Literal["gap", "simpool", "ep"] | BasePooling | None = None,
        freeze_backbone: bool = False,
        sample_rate: int = 16000,
        classifier_hidden_layers: list[int] | None = None,
        activation: Literal["relu", "gelu", "tanh", "leakyrelu"] = "relu",
        apply_batch_norm: bool = True,
        pretrained: bool = False,
    ):
        """Initialize the AudioClassifierConstructor.

        Args:
            num_classes (int): Number of output classes.
            backbone (Literal["beats", "passt"] | BaseBackbone): Backbone model to use for feature extraction.
            pooling (Literal["gap", "simpool", "ep"] | BasePooling | None): Optional pooling layer to aggregate
                features.
            freeze_backbone (bool): Whether to freeze the backbone weights during training.
            sample_rate (int): Sample frequency for audio input.
            classifier_hidden_layers (list[int] or None): Hidden layer sizes for the classifier head.
            activation (Literal["relu", "gelu", "tanh", "leakyrelu"]): Activation function for the classifier head.
            apply_batch_norm (bool): Whether to apply batch normalization in the classifier head.
            pretrained (bool): Whether to load pretrained weights for the backbone.
                If pooling is None, GAP is used by default.

        Example:
            >>> from deepaudiox import AudioClassifier
            >>> model = AudioClassifier(
            ...     num_classes=10,
            ...     backbone="beats",
            ...     pooling=None,
            ...     freeze_backbone=True,
            ...     sample_rate=16000,
            ...     classifier_hidden_layers=[512, 256],
            ...     activation="relu",
            ...     apply_batch_norm=True,
            ...     pretrained=True,
            ... )
        """
        super().__init__()

        self.backbone_constructor = BackboneConstructor(
            backbone=backbone,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            pooling=pooling,
            sample_rate=sample_rate,
        )

        self.classifier = MLPHead(
            num_classes=num_classes,
            in_dim=self.backbone_constructor.out_dim,
            hidden_layers=classifier_hidden_layers,
            activation=activation,
            apply_batch_norm=apply_batch_norm,
        )

    def forward(self, x) -> torch.Tensor:
        """Forward pass through the classifier.

        Args:
            x (torch.Tensor): Input waveforms of shape (B, T)

        Returns:
            torch.Tensor: Logits of shape (B, num_classes)

        Example:
            >>> import torch
            >>> from deepaudiox import AudioClassifier
            >>> model = AudioClassifier(num_classes=10, backbone="beats", sample_rate=16_000, pretrained=True)
            >>> waveforms = torch.randn(2, 5 * 16_000)
            >>> logits = model.forward(waveforms)
            >>> # logits shape: (B, num_classes)
        """
        x = self.forward_with_pooling(x)
        x = self.classifier(x)
        return x

    def forward_backbone(self, x) -> torch.Tensor:
        """Extract feature map from the backbone.

        Args:
            x (torch.Tensor): Input waveforms of shape (B, T).

        Returns:
            torch.Tensor: Returns the feature map of the backbone model (B, T, D) or (B, D, H, W).

        Example:
            >>> import torch
            >>> from deepaudiox import AudioClassifier
            >>> model = AudioClassifier(num_classes=10, backbone="beats", sample_rate=16_000, pretrained=True)
            >>> waveforms = torch.randn(2, 5 * 16_000)
            >>> features = model.forward_backbone(waveforms)
            >>> # features shape: (B, N, D) for Transformer or (B, D, H, W) for CNN backbones
        """

        return self.backbone_constructor(x)

    def forward_with_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and pooling.

        Args:
            x (torch.Tensor): x (torch.Tensor): Input waveforms of shape (B, T).

        Returns:
            torch.Tensor: Pooled tensor of shape (B, D).

        Example:
            >>> import torch
            >>> from deepaudiox import AudioClassifier
            >>> model = AudioClassifier(num_classes=10, backbone="beats", sample_rate=16_000, pretrained=True)
            >>> waveforms = torch.randn(2, 5 * 16_000)
            >>> embeddings = model.forward_with_pooling(waveforms)
            >>> # embeddings shape: (B, D)
        """
        return self.backbone_constructor.forward_with_pooling(x)
