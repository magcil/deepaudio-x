"""
This page provides the core API reference for DeepAudioX.
"""

__version__ = "0.4.0"

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
from deepaudiox.utils.training_utils import (  # noqa: F401
    get_class_mapping_from_dir,
    get_class_mapping_from_list,
)

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
    "get_class_mapping_from_dir",
    "get_class_mapping_from_list",
]
