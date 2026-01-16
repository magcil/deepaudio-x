from typing import Literal

import torch

from deepaudiox.modules.backbones import BACKBONES
from deepaudiox.modules.backbones.base_backbone import BaseBackbone
from deepaudiox.modules.base_audio_classifier import BaseAudioClassifier
from deepaudiox.modules.classifier.classifier import MLPHead
from deepaudiox.modules.pooling import POOLING
from deepaudiox.modules.pooling.base_pooling import BasePooling
from deepaudiox.modules.pooling.gap import GAP
from deepaudiox.utils.downloader import Downloader
from deepaudiox.utils.file_utils import load_checkpoint


class AudioClassifierConstructor(BaseAudioClassifier):
    """Classifier model using a backbone for feature extraction.

    Attributes:
        num_classes (int): Number of output classes.
        backbone_model (BaseBackbone): Backbone model for feature extraction.
        pooling (BasePooling or None): Optional pooling layer to aggregate features. If None, GAP is used.
        emb_dim (int): Dimension of the embeddings after projection (if any).
        classifier (MLPHead): Classifier head for final predictions.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: Literal["beats"] | BaseBackbone,
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
            backbone (Literal["beats"] | BaseBackbone): Backbone model to use for feature extraction.
            pooling (Literal["gap", "simpool"] | BasePooling | None): Optional pooling layer to aggregate features.
            freeze_backbone (bool): Whether to freeze the backbone weights during training.
            sample_rate (int): Sample frequency for audio input.
            classifier_hidden_layers (list[int] or None): Hidden layer sizes for the classifier head.
            activation (Literal["relu", "gelu", "tanh", "leakyrelu"]): Activation function for the classifier head.
            apply_batch_norm (bool): Whether to apply batch normalization in the classifier head.
            pretrained (bool): Whether to load pretrained weights for the backbone.

        Example:
            >>> from deepaudiox.modules.audio_classifier_constructor import AudioClassifierConstructor
            >>> model = AudioClassifierConstructor(
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

        self.backbone_model = self._resolve_backbone(backbone, pretrained, sample_rate)

        # Freeze backbone's weights
        if freeze_backbone:
            for p in self.backbone_model.parameters():
                p.requires_grad = False

        self.pooling = self._resolve_pooling(pooling)

        self.classifier = MLPHead(
            num_classes=num_classes,
            in_dim=self.backbone_model.out_dim,
            hidden_layers=classifier_hidden_layers,
            activation=activation,
            apply_batch_norm=apply_batch_norm,
        )

    def _resolve_backbone(
        self, backbone: Literal["beats"] | BaseBackbone, pretrained: bool, sample_rate: int
    ) -> BaseBackbone:
        """Resolve backbone from literal or BaseBackbone instance.

        Args:
            backbone (Literal["beats"] | BaseBackbone): Backbone name or instance.
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

    def _resolve_pooling(self, pooling: Literal["gap", "simpool", "ep"] | BasePooling | None) -> BasePooling:
        """Resolve pooling layer from literal, BasePooling instance, or None.

        Args:
            pooling (Literal["gap", "simpool"] | BasePooling | None): Pooling layer name, instance, or None.

        Returns:
            BasePooling: Initialized pooling layer.
        """
        if pooling is None:
            return GAP()
        elif isinstance(pooling, str):
            return POOLING[pooling](dim=self.backbone_model.out_dim)
        else:
            return pooling

    def forward(self, x) -> torch.Tensor:
        """Forward pass through the classifier.

        Args:
            x (torch.Tensor): Input waveforms of shape (B, T)

        Returns:
            torch.Tensor: Logits of shape (B, num_classes)
        """
        embedding = self.get_embeddings(x)
        x = self.apply_pooling(embedding)
        x = self.classifier(x)

        return x

    def get_embeddings(self, x) -> torch.Tensor:
        """Extract embeddings from the backbone (with optional projection).

        Args:
            x (torch.Tensor): Input waveforms of shape (B, T).

        Returns:
            torch.Tensor: Returns the feature map of the backbone model.
        """

        return self.backbone_model.forward_pipeline(x)

    def apply_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pooling to the input feature map.

        Args:
            x (torch.Tensor): Input feature map of shape (B, D, H, W) for CNNs or (B, T, D) for Transformers.

        Returns:
            torch.Tensor: Pooled tensor of shape (B, D).
        """
        return self.pooling(x)
