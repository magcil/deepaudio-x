import torch
import torch.nn as nn
import torchaudio


class Wav2VecClassifier(nn.Module):
    """
    Lightweight audio classification model built on top of a pretrained Wav2Vec2 backbone.
    """

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        bundle_name: str = "WAV2VEC2_BASE"
    ):
        """
        Args:
            num_classes: Number of target classes.
            pretrained: Whether to load pretrained weights.
            freeze_backbone: If True, backbone weights are frozen.
            bundle_name: torchaudio pipeline name (e.g., 'WAV2VEC2_BASE', 'WAV2VEC2_ASR_BASE_960H').
        """
        super().__init__()

        # Load a pretrained Wav2Vec2 model
        bundle = getattr(torchaudio.pipelines, bundle_name)
        self.sample_rate = bundle.sample_rate
        self.feature_extractor = bundle.get_model()

        # Optionally freeze the backbone
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # Infer feature dimension from backbone
        self.feature_dim = 768

        # Simple classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor | None = None):
        # Forward through Wav2Vec2
        features, _ = self.feature_extractor.extract_features(waveforms)

        # Take last layer output [B, T', D]
        features = features[-1]

        # Optional: masked mean pooling
        if lengths is not None:
            # downsample factor for Wav2Vec2 BASE ~320
            downsample_factor = 320
            valid_lengths = torch.div(lengths, downsample_factor, rounding_mode='floor')
            mask = torch.arange(features.size(1), device=features.device)[None, :] < valid_lengths[:, None]
            pooled = (features * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            pooled = features.mean(dim=1)

        # Classification head
        logits = self.classifier(pooled)
        return logits


    def get_backbone(self):
        """Return the pretrained feature extractor for downstream reuse."""
        return self.feature_extractor
