from typing import Literal

import torch

from deepaudiox.modules.backbones import BACKBONES
from deepaudiox.modules.base_audio_classifier import BaseAudioClassifier
from deepaudiox.modules.classifier.classifier import MLPHead
from deepaudiox.modules.projection.base_projection import BaseProjection
from deepaudiox.utils.downloader import Downloader
from deepaudiox.utils.file_utils import load_checkpoint


class AudioClassifierConstructor(BaseAudioClassifier):
    """Classifier model using a backbone for feature extraction.

    Attributes:
        num_classes (int): Number of output classes.
        backbone_model (BaseBackbone): Backbone model for feature extraction.
        projection (BaseProjection or None): Optional projection layer to adjust embedding dimensions.
        emb_dim (int): Dimension of the embeddings after projection (if any).
        classifier (MLPHead): Classifier head for final predictions.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: Literal["beats"],
        projection: BaseProjection | None = None,
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
            backbone (Literal["beats"]): Backbone model to use for feature extraction.
            projection (BaseProjection or None): Optional projection layer to adjust embedding dimensions.
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
            ...     projection=None,
            ...     freeze_backbone=True,
            ...     sample_rate=16000,
            ...     classifier_hidden_layers=[512, 256],
            ...     activation="relu",
            ...     apply_batch_norm=True,
            ...     pretrained=True,
            ... )
        """
        super().__init__()

        self.backbone_model = BACKBONES[backbone]()
        # Set sample frequency for backbone feature extraction
        self.backbone_model.sample_rate = sample_rate

        if pretrained:
            downloader = Downloader()
            ckpt_path = downloader.download_checkpoint(backbone)
            ckpt = load_checkpoint(ckpt_path)
            self.backbone_model.load_state_dict(ckpt)

        # Freeze backbone's weights
        if freeze_backbone:
            for p in self.backbone_model.parameters():
                p.requires_grad = False

        self.projection: BaseProjection | None = None

        if projection is not None:
            self.projection = projection
            self.emb_dim = self.projection.out_dim
        else:
            self.emb_dim = self.backbone_model.out_dim

        self.classifier = MLPHead(
            num_classes=num_classes,
            in_dim=self.emb_dim,
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
        """
        embedding = self.get_embeddings(x)
        x = self.classifier(embedding)

        return x

    def get_embeddings(self, x) -> torch.Tensor:
        """Extract embeddings from the backbone (with optional projection).

        Args:
            x (torch.Tensor): Input waveforms of shape (B, T).

        Returns:
            torch.Tensor: Extracted embeddings of shape (B, emb_dim).
        """

        return (
            self.projection(self.backbone_model.forward_pipeline(x))
            if self.projection
            else self.backbone_model.forward_pipeline(x)
        )
