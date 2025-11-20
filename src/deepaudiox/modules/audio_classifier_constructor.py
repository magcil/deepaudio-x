from typing import Literal

import torch.nn as nn

from deepaudiox.modules.backbones import BACKBONES
from deepaudiox.modules.base_audio_classifier import BaseAudioClassifier
from deepaudiox.modules.classifier.classifier import MLPHead
from deepaudiox.modules.projection.base_projection import BaseProjection


class AudioClassifierConstructor(BaseAudioClassifier):
    def __init__(
        self,
        num_classes: int,
        backbone: Literal["beats"],
        projection: BaseProjection | None = None,
        freeze_backbone: bool = False,
        sample_frequency: int = 16000,
        classifier_hidden_layers: list[int] | None = None,
        activation: Literal["relu", "gelu", "tanh", "leakyrelu"] = "relu",
        apply_batch_norm: bool = True,
    ):
        """Classifier model using a backbone for feature extraction.
        Attributes:
            num_classes (int): Number of output classes.
            backbone (str): Backbone model name.
            freeze_backbone (bool): Whether to freeze backbone weights during training.
            sample_frequency (int): Sample frequency for audio input.
            classifier_hidden_layers (list[int] or None): List of hidden layer sizes for classifier head.
            activation (str): Activation function name for classifier head.
            apply_batch_norm (bool): Whether to use BatchNorm1d in classifier head.
        """
        super().__init__()

        self.backbone_model = BACKBONES[backbone]()
        # Set sample frequency for backbone feature extraction
        self.backbone_model.sample_frequency = sample_frequency

        if freeze_backbone:
            self.backbone_model.freeze_encoder_weights()

        if projection is not None:
            self.backbone_model = nn.Sequential(self.backbone_model, projection)
            self.emb_dim = projection.out_dim
        else:
            self.emb_dim = self.backbone_model.out_dim

        self.classifier = MLPHead(
            num_classes=num_classes,
            in_dim=self.emb_dim,
            hidden_layers=classifier_hidden_layers,
            activation=activation,
            apply_batch_norm=apply_batch_norm,
        )

    def forward(self, x):
        embedding = self.get_embeddings(x)
        x = self.classifier(embedding)

        return x

    def get_embeddings(self, x):
        return self.backbone_model(x)
