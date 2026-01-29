# deepaudiox/modules/baseclasses.py

"""
BaseClasses for abstracting nn modules (e.g., backbones, pooling layers, classifiers)
"""

from abc import ABC, abstractmethod
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepaudiox.dtos.dataset_items import AudioPrediction


class BaseAudioClassifier(nn.Module, ABC):
    """Base class for creating custom audio classifiers.

    This class defines the standard interface for audio classification models.
    Subclasses must implement the core initialization and forward pass. The
    built-in `predict` method provides a convenience wrapper to obtain predicted
    labels, posterior probabilities, and raw logits.

    Methods:
        __init__: Initialize the classifier and its components.
        forward: Process input waveforms and return logits.
        predict: Compute predicted classes, posterior probabilities, and logits.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialize the audio classifier."""
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Pass the input through the model and return logits.

        Args:
            x (torch.Tensor): The input tensor.
        """
        raise NotImplementedError

    def predict(self, x: torch.Tensor) -> dict[str, np.ndarray]:
        """Compute predicted class and posterior probabilities.

        Args:
            x (torch.Tensor): Input Waveforms of shape B x N*SR, B Batch size, N length, SR sample rate

        Returns:
            dict[str, np.ndarray]: y_preds, posteriors, logits.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        logits = self.forward(x)
        posteriors = F.softmax(logits, dim=1)
        max_posteriors = posteriors.max(dim=1)

        return {
            "y_preds": max_posteriors.indices.numpy(force=True),
            "posteriors": max_posteriors.values.numpy(force=True),
            "logits": logits.numpy(force=True),
        }
    
    def inference_on_waveform(
            self,
            x: torch.Tensor,
            sample_rate: int,
            class_mapping: dict[str, int],
            segment_duration: int | None
        ) -> AudioPrediction:
        """Get prediction on a waveform.
        
        Args:
        x (torch.Tensor): Input waveform to be used for inference.
        sample_rate (int): Sampling rate of audio sample.
        class_mapping (dict[str, int]): Class-to-index mapping that is used by the model.
        segment_duration (int | None): None for inference on whole waveform, or int segment duration in sec for inference on segments.
        
        Returns:
        AudioPrediction: An object containing the final label and posterior, and optionally segment-level results.
        """
        index_to_class = {idx: cl for cl, idx in class_mapping.items()}

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        if x.dtype != torch.float32:
            x = x.float()

        device = next(self.parameters()).device

        if segment_duration:
            total_duration = len(x) / sample_rate

            if total_duration < segment_duration:
                x = x.to(device)
                inference = self.model.predict(x)
                prediction = AudioPrediction(
                final_label = index_to_class[inference["y_preds"][0]],
                final_posterior = inference["posteriors"][0])

                return prediction

            else:
                segments = []
                num_segments = total_duration // self.segment_duration
                segment_length = self.segment_duration * self.sample_rate

                for i in range(num_segments):
                    segment = x[i * segment_length : i * segment_length + segment_length]
                    segments.append(segment)

                batch = torch.stack([torch.tensor(seg) for seg in segments])
                batch = batch.to(device)
                inference = self.model.predict(batch)

                segment_labels = []
                for pred in inference["y_preds"]:
                    segment_labels.append(index_to_class[pred])

                # Majority vote to get final prediction
                preds, counts = np.unique((inference["y_preds"]), return_counts=True)
                top_preds = preds[counts == counts.max()]
                final_pred = None

                if len(top_preds) == 1:
                    # No tie: use only majority vote
                    final_pred = top_preds[0]
                    class_posteriors = []
                    for pred, posterior in zip(inference["y_preds"], inference["posteriors"]):
                        if pred == final_pred:
                            class_posteriors.append(posterior)
                    mean_posterior = np.mean(class_posteriors)
                else:
                    # Tie: use majority vote and choose the y_pred with the highest mean posterior
                    max_mean_posterior = 0
                    for y_pred in top_preds:
                        class_posteriors = []
                        for pred, posterior in zip(inference["y_preds"], inference["posteriors"]):
                            if pred == y_pred:
                                class_posteriors.append(posterior)
                        mean_posterior = np.mean(class_posteriors)
                        if mean_posterior > max_mean_posterior:
                                max_mean_posterior = mean_posterior
                                final_pred = y_pred

                prediction = AudioPrediction(
                final_label = index_to_class[final_pred],
                final_posterior = mean_posterior,

                segment_labels = segment_labels,
                segment_posteriors = list(inference["posteriors"]))

                return prediction
        
        else:
            x = x.to(device)
            inference = self.model.predict(x)
            prediction = AudioPrediction(
            final_label = index_to_class[inference["y_preds"][0]],
            final_posterior = inference["posteriors"][0])

            return prediction
    
    def inference_on_file(
            self,
            path: str | Path,
            sample_rate: int,
            class_mapping: dict[str, int],
            segment_duration: int | None
        ) -> AudioPrediction:
        """Get prediction for an audio sample from a file path.
        
        Args:
        path (str): Path to the audio file (MP3 or WAV) to be used for inference.
        sample_rate (int): Sampling rate of audio sample.
        class_mapping (dict[str, int]): Class-to-index mapping as it is used by the model.
        segment_duration (int | None): None for inference on whole waveform, or int segment duration in sec for inference on segments.

        Returns:
        AudioPrediction: An object containing the final label and posterior, and optionally segment-level results.
        """
        y, _ = librosa.load(path, sr=sample_rate)
        prediction = self.inference_on_waveform(
            self,
            x = y,
            sample_rate = sample_rate,
            class_mapping = class_mapping,
            segment_duration = segment_duration
        )

        return prediction


class BasePooling(nn.Module, ABC):
    """Abstract base class for all pooling modules.

    This class defines the interface for pooling that operate an input
    feature map obtained from a CNN or a Transformer BaseBackbone. Subclasses must
    implement the forward-processing logic. The input is expected to be a feature map of shape (B, D, H, W) for CNNs
    or (B, T, D) for Transformers.

    Methods:
        __init__: Store input dimensionality.
        forward: Apply the pooling module to an input tensor and return the result.
    """

    def __init__(self, in_dim: int | None = None) -> None:
        """Initialize the BaseProjection.

        Args:
            in_dim (int): Input dimension. This is D for both CNNs and Transformers.
        """
        super().__init__()
        self.in_dim = in_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass returning a projected tensor."""
        pass


class BaseBackbone(nn.Module, ABC):
    """Abstract base class for all audio backbone models.

    This class defines the common interface for backbone architectures that
    convert raw waveforms into fixed-dimensional embeddings. Subclasses must
    implement the core feature extraction and forward-processing logic.

    Methods:
        __init__: Initializes the embedding dimension and the sample_rate of the audios.
        forward: Computes embeddings from pre-extracted audio features.
        extract_features: Converts raw waveforms into model-specific features.
        forward_pipeline: Extracts features and then applies forward().
    """

    def __init__(self, out_dim: int, sample_rate: int) -> None:
        """Initialize the BaseBackbone.

        Args:
            out_dim (int): Output dim of the backbone feature map. For CNNs the embeddings are of shape (B, C, H, W)
            and for Transformers of shape (B, T, D), where out_dim is either C or D respectively. The output embeddings
            could be of shape (B, out_dim) in case of pooling backbones.
            sample_rate (int): Sample rate for audio input.
        """
        super().__init__()
        self.out_dim = out_dim
        self.sample_rate = sample_rate

    @abstractmethod
    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Computes of the embeddings of the input features.

        Args:
            x: (torch.Tensor) Input audio-specific features of shape (B, 1, F, T) or (B, 1, T, F)
            padding_mask: (torch.Tensor) Optional padding mask.

        Returns:
            torch.Tensor: Embeddings of shape (B, N, D) or (B, D, H, W) where D is the embedding dimension.
        """
        pass

    @abstractmethod
    def extract_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Convert raw waveforms into internal acoustic features.

        Args:
            waveforms (torch.Tensor): Tensor of shape (B, T).

        Returns:
            torch.Tensor: Model-specific feature representation before final forward().
        """
        pass

    def forward_pipeline(self, x: torch.Tensor) -> torch.Tensor:
        """Standard processing pipeline:

            1. Extract features from raw audio
            2. Pass features through forward()

        Args:
            x (torch.Tensor): Input waveforms of shape (B, T), where T is the length of waveforms.

        Returns:
            torch.Tensor: Final model output of shape (B, out_dim, H, W) for CNNs or (B, T, out_dim) for Transformers.
        """
        x = self.extract_features(x)
        return self.forward(x)
