"""
DeepAudioX: A deep learning framework for end-to-end audio classification.

This package provides modules for:
- Dataset management and preprocessing
- Model architectures
- Training, evaluation, and inference utilities
"""

__version__ = "0.1.5"

# Top-level API exports
from deepaudiox.datasets.audio_classification_dataset import (  # noqa: F401
    AudioClassificationDataset,
    audio_classification_dataset_from_dictionary,
    audio_classification_dataset_from_dir,
)
from deepaudiox.loops.evaluator import Evaluator  # noqa: F401
from deepaudiox.loops.trainer import Trainer  # noqa: F401
from deepaudiox.modules.backbones import BACKBONES  # noqa: F401
from deepaudiox.modules.constructors import (  # noqa: F401
    AudioClassifierConstructor,
    BackboneConstructor,
)
from deepaudiox.modules.pooling import POOLING  # noqa: F401

# User-friendly aliases
AudioClassifier = AudioClassifierConstructor
Backbone = BackboneConstructor

__all__ = [
    "AudioClassifier",
    "AudioClassifierConstructor",
    "AudioClassificationDataset",
    "Backbone",
    "BackboneConstructor",
    "Evaluator",
    "Trainer",
    "BACKBONES",
    "POOLING",
    "audio_classification_dataset_from_dictionary",
    "audio_classification_dataset_from_dir",
]
