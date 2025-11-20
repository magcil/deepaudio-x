import itertools
import logging
import math
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Generator, default_generator, randperm
from torch.utils.data import Subset

from deepaudiox.datasets.audio_classification_dataset import AudioClassificationDataset


def get_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("ConsoleLogger")
    return logger


def get_class_mapping(root_dir: str) -> dict[str, int]:
    """Load the class mapping given a folder of class sub-folders.

    Args:
        root_dir (str): The path to root folder

    Returns:
        Dict: The class mapping dictionary

    """
    root_path = Path(root_dir)
    if not root_path.is_dir():
        raise ValueError(f"The path '{root_dir}' is not a directory")

    subdirs = [d for d in root_path.iterdir() if d.is_dir()]
    class_mapping = {d.name: idx for idx, d in enumerate(sorted(subdirs))}

    return class_mapping


def get_device() -> torch.device:
    """Returns the best available device for PyTorch computations.

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU available)")

    return device


def pad_collate_fn(batch) -> dict:
    """
    Collate function that pads variable-length audio tensors in a batch.

    Args:
        batch (list of dict): Each dict contains 'feature', 'class_id', and 'class_name'.

    Returns:
        dict: Batched and padded tensors.

    """
    # Extract data
    features = [item["feature"] for item in batch]
    labels = torch.tensor([item["class_id"] for item in batch], dtype=torch.long)
    class_names = [item["class_name"] for item in batch]

    # Ensure features are tensors
    features = [torch.as_tensor(f) for f in features]

    # Find max length in this batch
    max_len = max(f.shape[-1] for f in features)

    # Pad each tensor to max_len
    padded_features = [
        F.pad(f, (0, max_len - f.shape[-1])) if f.shape[-1] < max_len else f[..., :max_len] for f in features
    ]

    # Stack into a single batch tensor: (batch_size, num_channels, num_samples)
    batch_features = torch.stack(padded_features)

    return {"feature": batch_features, "class_id": labels, "class_name": class_names}


def random_split_audio_dataset(
    dataset: AudioClassificationDataset, lengths: Sequence[int | float], generator: Generator = default_generator
):
    """Extends the random_split by PyTorch.

    Randomly split an AudioClassificationDataset into non-overlapping
    ClassificationDataset of given lengths. Accounts for segmentized dataset, i.e, segments of a given recording fall
    into the same Subset.

    """

    wav_files = np.array([d["path"] for d in dataset.instances])

    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: list[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(wav_files) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(wav_files) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. This might result in an empty wav_files.", stacklevel=2
                )

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(wav_files):  # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[arg-type, call-overload]
    lengths = cast(Sequence[int], lengths)

    if dataset.segment_map:
        segment_mapping_array = np.array([d["file_path"] for d in dataset.segment_map])
        return [
            Subset(dataset, np.where(np.isin(segment_mapping_array, wav_files[offset - length : offset]))[0].tolist())
            for offset, length in zip(itertools.accumulate(lengths), lengths, strict=False)
        ]
    else:
        return [
            Subset(dataset, indices[offset - length : offset])
            for offset, length in zip(itertools.accumulate(lengths), lengths, strict=False)
        ]
