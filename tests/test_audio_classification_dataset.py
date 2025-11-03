import numpy as np
import pytest
import torch
import torchaudio
from torch.utils.data import random_split

from deepaudiox.datasets.audio_classification_dataset import AudioClassificationDataset
from deepaudiox.utils.training_utils import get_class_mapping

torchaudio.set_audio_backend("soundfile")


@pytest.fixture(autouse=True)
def mock_audio_dataset(tmp_path):
    """Create a temporary folder structure with fake WAV files."""
    train_dir = tmp_path / "audio_data" / "train"
    test_dir = tmp_path / "audio_data" / "test"
    train_a = train_dir / "class_a"
    train_b = train_dir / "class_b"
    test_a = test_dir / "class_a"
    test_b = test_dir / "class_b"
    for d in [train_a, train_b, test_a, test_b]:
        d.mkdir(parents=True)

    sample_rate = 16000
    for folder in [train_a, train_b, test_a, test_b]:
        for i in range(10 if "train" in str(folder) else 2):
            path = folder / f"file_{i}.wav"
            waveform = torch.randn(sample_rate * 3)  # 3 seconds of random audio
            torchaudio.save(str(path), waveform.unsqueeze(0), sample_rate, format="wav")

    return train_dir, test_dir


def test_dataset_splits_counts(mock_audio_dataset):
    train_dir, test_dir = mock_audio_dataset
    class_mapping = get_class_mapping(train_dir)

    dataset = AudioClassificationDataset(root_dir=train_dir, sample_rate=16000, class_mapping=class_mapping)

    train_dset, val_dset = random_split(dataset, [0.9, 0.1])

    assert len(train_dset) == 18
    assert len(val_dset) == 2


def test_item_data_types(mock_audio_dataset):
    train_dir, test_dir = mock_audio_dataset
    class_mapping = get_class_mapping(train_dir)

    dataset = AudioClassificationDataset(root_dir=train_dir, sample_rate=16000, class_mapping=class_mapping)

    print(dataset[0]["feature"])

    assert isinstance(dataset[0]["feature"], np.ndarray)
    assert isinstance(dataset[0]["class_id"], int)
    assert isinstance(dataset[0]["class_name"], str)


@pytest.mark.parametrize("segment_duration", [0.5, 1.0])
def test_segmentization(mock_audio_dataset, segment_duration):
    train_dir, test_dir = mock_audio_dataset
    class_mapping = get_class_mapping(train_dir)

    dataset = AudioClassificationDataset(
        root_dir=train_dir, sample_rate=16000, class_mapping=class_mapping, segment_duration=segment_duration
    )

    # Each 3-second audio should yield 3 segments
    expected_num_segments = (
        10 * 2 * (3 // segment_duration)
    )  # 10 files per class, 2 classes, 3 // segment_duration segments each
    assert len(dataset) == expected_num_segments

    segment = dataset[0]
    assert isinstance(segment["feature"], np.ndarray)
    assert len(segment["feature"]) == int(segment_duration * 16000)  # Check segment length
