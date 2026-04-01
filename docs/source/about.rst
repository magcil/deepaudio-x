About
=====

DeepAudioX is a PyTorch-based library that provides **simple, flexible pipelines for audio classification** using **pretrained audio foundation models** as feature extractors.

It is designed to let users train, evaluate, and run inference on **custom audio datasets** with only a few lines of code, while still allowing advanced customization when needed.

Key Features
------------

- 🔊 **Pretrained audio backbones** for feature extraction
- 🧠 **Modular pooling strategies** (e.g. GAP, SimPool, EfficientProbing)
- 🧩 **Custom classifier heads** for downstream audio classification
- 🚀 **High-level training, evaluation, and inference APIs**
- 🔁 Fully **PyTorch-native** and extensible
- 📦 Clean integration with existing PyTorch workflows

Quick Start
-----------

With DeepAudio-X, you can build and train an audio classifier in just a few lines
of code, leveraging pretrained backbones for state-of-the-art performance.
Suppose you have a dataset organized in a directory structure where each
subdirectory corresponds to a class label and contains audio files for that
class as follows::

   path/to/data/
       class_1/
           audio_1.wav
           audio_2.wav
           ...
       class_2/
           audio_3.wav
           audio_4.wav
           ...
       ...

Then you can use the following minimal code to create, train, and evaluate an
audio classifier:

.. code-block:: python

   import torch

   from deepaudiox import AudioClassifier, Evaluator, Trainer
   from deepaudiox import audio_classification_dataset_from_dir
   from deepaudiox import get_class_mapping_from_dir

   # 1) Build a dataset from a folder structure of class subdirectories
   class_mapping = get_class_mapping_from_dir(root_dir="path/to/data")
   dataset = audio_classification_dataset_from_dir(
       root_dir="path/to/data",
       sample_rate=16_000,
       class_mapping=class_mapping,
   )

   # 2) Create a classifier with a pretrained backbone
   classifier = AudioClassifier(
       num_classes=len(class_mapping),
       backbone="beats",
       sample_rate=16_000,
       pretrained=True,
       freeze_backbone=True,
   )

   # 3) Train
   trainer = Trainer(
       train_dset=dataset,
       model=classifier,
       validation_dset=dataset,  # replace with a real validation set
       batch_size=16,
       epochs=5,
   )

   trainer.train()

   classifier.load_state_dict(torch.load("checkpoint.pt"))  # Load model

   # 4) Evaluate on a test set
   evaluator = Evaluator(
       test_dset=dataset,  # replace with a real test set
       model=classifier,
       batch_size=16,
   )

   evaluator.evaluate()

