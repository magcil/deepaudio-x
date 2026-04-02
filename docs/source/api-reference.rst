API Reference
=============

.. currentmodule:: deepaudiox

Dataset Construction
--------------------

Methods for building datasets and class mappings from directories or label lists.

.. autofunction:: get_class_mapping_from_dir

.. autofunction:: get_class_mapping_from_list

.. autoclass:: AudioClassificationDataset
   :members:
   :exclude-members: __init__
   :undoc-members:

.. autofunction:: audio_classification_dataset_from_dir

.. autofunction:: audio_classification_dataset_from_dictionary

Models & Backbones
------------------

Constructors for initializing classifiers and backbones.

.. autoclass:: deepaudiox.modules.constructors.AudioClassifierConstructor
   :members:
   :special-members: __init__
   :undoc-members:

   .. note:: Available as ``deepaudiox.AudioClassifier``.

.. autoclass:: deepaudiox.modules.constructors.BackboneConstructor
   :members:
   :special-members: __init__
   :undoc-members:

   .. note:: Available as ``deepaudiox.Backbone``.

Supported Backbones & Pooling
-----------------------------

Type aliases and runtime constants for valid backbone and pooling names.

.. data:: AVAILABLE_BACKBONES
   :annotation: = ("beats", "passt", "mobilenet_05_as", "mobilenet_10_as", "mobilenet_40_as")

   Supported pretrained backbone names available at runtime.

.. data:: AVAILABLE_POOLING
   :annotation: = ("gap", "simpool", "ep")

   Supported pooling layer names available at runtime.

.. data:: BackboneName

   Type alias: ``Literal["beats", "passt", "mobilenet_05_as", "mobilenet_10_as", "mobilenet_40_as"]``.
   Use for type-annotated code.

.. data:: PoolingName

   Type alias: ``Literal["gap", "simpool", "ep"]``.
   Use for type-annotated code.

Training & Evaluation
---------------------

Interfaces for training models and evaluating performance on held-out data.

.. autoclass:: Trainer
   :members:
   :special-members: __init__
   :undoc-members:

.. autoclass:: Evaluator
   :members:
   :special-members: __init__
   :undoc-members:

Base Classes & Inference
------------------------

Base interfaces and inference helpers used across models.

.. automodule:: deepaudiox.modules.baseclasses
   :members: BaseAudioClassifier, BaseBackbone, BasePooling
   :undoc-members:

Full Paths
----------

The API re-exports the following symbols. If you prefer importing from the original modules, use these paths:

- ``AudioClassifier`` -> ``deepaudiox.modules.constructors.AudioClassifierConstructor``
- ``Backbone`` -> ``deepaudiox.modules.constructors.BackboneConstructor``
- ``AudioClassificationDataset`` -> ``deepaudiox.datasets.audio_classification_dataset.AudioClassificationDataset``
- ``audio_classification_dataset_from_dir`` -> ``deepaudiox.datasets.audio_classification_dataset.audio_classification_dataset_from_dir``
- ``audio_classification_dataset_from_dictionary`` -> ``deepaudiox.datasets.audio_classification_dataset.audio_classification_dataset_from_dictionary``
- ``get_class_mapping_from_dir`` -> ``deepaudiox.utils.training_utils.get_class_mapping_from_dir``
- ``get_class_mapping_from_list`` -> ``deepaudiox.utils.training_utils.get_class_mapping_from_list``
- ``Trainer`` -> ``deepaudiox.loops.trainer.Trainer``
- ``Evaluator`` -> ``deepaudiox.loops.evaluator.Evaluator``
- ``BackboneName`` -> ``deepaudiox.schemas.types.BackboneName``
- ``PoolingName`` -> ``deepaudiox.schemas.types.PoolingName``
- ``AVAILABLE_BACKBONES`` -> ``deepaudiox.__init__.AVAILABLE_BACKBONES``
- ``AVAILABLE_POOLING`` -> ``deepaudiox.__init__.AVAILABLE_POOLING``
