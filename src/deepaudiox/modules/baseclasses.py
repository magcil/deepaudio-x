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
from deepaudiox.utils.decorators import eval_mode


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

        This is a low-level method that does not manage model mode or gradient
        context. The caller is responsible for calling ``model.eval()`` and
        wrapping with ``torch.no_grad()`` or ``torch.inference_mode()`` as needed.
        For end-to-end inference with automatic mode management, use
        ``inference_on_waveform`` or ``inference_on_file`` instead.

        Args:
            x (torch.Tensor): Input waveforms of shape (B, T), where T is the number of audio samples.

        Returns:
            dict[str, np.ndarray]: y_preds, posteriors, logits.

        Example:
            >>> import torch
            >>> from deepaudiox import AudioClassifier
            >>> model = AudioClassifier(num_classes=10, backbone="beats", sample_rate=16_000, pretrained=True)
            >>> model.eval()
            >>> waveforms = torch.randn(2, 5 * 16_000)
            >>> with torch.no_grad():
            ...     outputs = model.predict(waveforms)
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

    @torch.inference_mode()
    @eval_mode
    def inference_on_waveform(
        self,
        x: torch.Tensor | np.ndarray,
        sample_rate: int,
        class_mapping: dict[str, int],
        segment_duration: float | None = None,
        batch_size: int = 4,
    ) -> dict:
        """Get prediction on a waveform.

        Args:
            x (torch.Tensor | np.ndarray): Input waveform to be used for inference. Accepts shape (T,).
            sample_rate (int): Sampling rate of audio sample.
            class_mapping (dict[str, int]): Class-to-index mapping that is used by the model.
            segment_duration (float | None): Optional segment duration in seconds for segment-level inference.
                If provided, the last remainder is right-padded to a full segment.
            batch_size (int): Optional batch size for segment-level inference. Default is 4.

        Returns:
            dict: A dictionary with keys:
                - ``final_label`` (str): Predicted class label.
                - ``final_posterior`` (float): Posterior probability for the predicted class.
                - ``segment_labels`` (list[str] | None): Per-segment labels when segmenting is used.
                - ``segment_posteriors`` (list[float] | None): Per-segment posteriors aligned with
                  ``segment_labels`` when segmenting is used.

        Example:
            >>> import torch
            >>> from deepaudiox import AudioClassifier
            >>> class_mapping = {"speech": 0, "music": 1}
            >>> model = AudioClassifier(num_classes=len(class_mapping), backbone="beats", sample_rate=16_000)
            >>> waveform = torch.randn(5 * 16_000)
            >>> prediction = model.inference_on_waveform(
            ...     waveform,
            ...     sample_rate=16_000,
            ...     class_mapping=class_mapping,
            ...     segment_duration=1.0,
            ...     batch_size=4,
            ... )
        """
        index_to_class = {idx: cl for cl, idx in class_mapping.items()}

        # Convert to tensor if input is a np.ndarray
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if x.ndim != 1:
            raise ValueError(
                f"Expected a 1-D waveform tensor of shape (T,), got shape {tuple(x.shape)}. "
                "If you have a batched input, loop over samples and call this method individually."
            )

        device = next(self.parameters()).device
        x = x.to(device)

        total_duration = x.numel() / sample_rate

        if segment_duration and total_duration > segment_duration:  # Process in segments
            segment_len = int(round(segment_duration * sample_rate))

            p, r = divmod(x.shape[0], segment_len)

            # Process the main part of the waveform that fits into full segments
            main_part = x[: int(p * segment_len)]
            if r > 0:  # If there is a remainder, pad it to create an additional segment
                remainder_part = F.pad(
                    x[int(p * segment_len) :],
                    (0, segment_len - r),
                )
                batch_segments = torch.cat([main_part, remainder_part], dim=0)
            else:
                batch_segments = main_part

            # Create batches of segments and run inference
            batch_segments = batch_segments.view(-1, segment_len)
            y_preds_batches, posteriors_batches = [], []

            for start in range(0, batch_segments.size(0), batch_size):
                batch = batch_segments[start : start + batch_size]
                batch_inference = self.predict(batch)
                y_preds_batches.append(batch_inference["y_preds"])
                posteriors_batches.append(batch_inference["posteriors"])

            inference = {
                "y_preds": np.concatenate(y_preds_batches),
                "posteriors": np.concatenate(posteriors_batches),
            }

            # Accumulate segment-level labels
            segment_labels = [index_to_class[pred] for pred in inference["y_preds"]]

            # Majority vote to get final prediction (tie-break by mean posterior)
            y_preds = inference["y_preds"]
            posteriors = inference["posteriors"]

            num_classes = len(class_mapping)
            counts = np.bincount(y_preds, minlength=num_classes)
            sum_posteriors = np.bincount(y_preds, weights=posteriors, minlength=num_classes)

            valid = counts > 0
            mean_posteriors = np.zeros_like(sum_posteriors, dtype=float)
            mean_posteriors[valid] = sum_posteriors[valid] / counts[valid]

            candidates = np.where(valid)[0]
            final_winner_index = max(candidates, key=lambda cls: (counts[cls], mean_posteriors[cls]))
            final_posterior = mean_posteriors[final_winner_index]

            return AudioPrediction(
                final_label=index_to_class[final_winner_index],
                final_posterior=final_posterior.item(),
                segment_labels=segment_labels,
                segment_posteriors=inference["posteriors"].tolist(),
            ).to_dict()

        else:  # Process the whole waveform at once if segment_duration is not specified or it total_duration < seg_dur
            inference = self.predict(x)

            return AudioPrediction(
                final_label=index_to_class[inference["y_preds"][0]], final_posterior=inference["posteriors"][0]
            ).to_dict()

    def inference_on_file(
        self,
        path: str | Path,
        sample_rate: int,
        class_mapping: dict[str, int],
        segment_duration: float | None = None,
        batch_size: int = 4,
    ) -> dict:
        """Get prediction for an audio sample from a file path.

        Args:
            path (str): Path to an audio file supported by librosa (e.g., WAV or MP3).
            sample_rate (int): Sampling rate of audio sample.
            class_mapping (dict[str, int]): Class-to-index mapping as it is used by the model.
            segment_duration (float | None): Optional segment duration in seconds for segment-level inference.
                If provided, the last remainder is right-padded to a full segment.
            batch_size (int): Optional batch size for segment-level inference. Default is 4.

        Returns:
            dict: A dictionary with keys:
                - ``final_label`` (str): Predicted class label.
                - ``final_posterior`` (float): Posterior probability for the predicted class.
                - ``segment_labels`` (list[str] | None): Per-segment labels when segmenting is used.
                - ``segment_posteriors`` (list[float] | None): Per-segment posteriors aligned with
                  ``segment_labels`` when segmenting is used.

        Example:
            >>> from deepaudiox import AudioClassifier
            >>> class_mapping = {"speech": 0, "music": 1}
            >>> model = AudioClassifier(num_classes=len(class_mapping), backbone="beats", sample_rate=16_000)
            >>> prediction = model.inference_on_file(
            ...     "path/to/audio.wav",
            ...     sample_rate=16_000,
            ...     class_mapping=class_mapping,
            ...     segment_duration=2.0,
            ...     batch_size=4,
            ... )
        """

        x, _ = librosa.load(path, sr=sample_rate)
        prediction = self.inference_on_waveform(
            x,
            sample_rate=sample_rate,
            class_mapping=class_mapping,
            segment_duration=segment_duration,
            batch_size=batch_size,
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
        """Initialize the BasePooling.

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
        """Compute embeddings from input features.

        Args:
            x (torch.Tensor): Input audio-specific features of shape (B, 1, F, T) or (B, 1, T, F).
            padding_mask (torch.Tensor | None): Optional padding mask.

        Returns:
            torch.Tensor: Embeddings of shape (B, N, D) or (B, D, H, W), where D is the embedding dimension.
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
            torch.Tensor: Final model output of shape (B, D, H, W) for CNNs or (B, N, D) for Transformers.
        """
        x = self.extract_features(x)
        return self.forward(x)
