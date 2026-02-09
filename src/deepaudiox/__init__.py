"""
DeepAudio-X API
===========

A self-supervised audio toolkit for audio classification .

This package provides modules for:

- datasets and audio preprocessing
- backbone and pooling architectures
- training, evaluation, and inference workflows
"""

__version__ = "0.3.2"

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
