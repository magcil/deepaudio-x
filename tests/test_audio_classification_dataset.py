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
    waveform = torch.zeros(sample_rate)
    for folder in [train_a, train_b, test_a, test_b]:
        for i in range(10 if "train" in str(folder) else 2):
            path = folder / f"file_{i}.wav"
            torchaudio.save(str(path), waveform.unsqueeze(0), sample_rate, format="wav")

    return train_dir, test_dir


def test_dataset_splits_counts(mock_audio_dataset):
    train_dir, test_dir = mock_audio_dataset
    class_mapping = get_class_mapping(train_dir)

    dataset = AudioClassificationDataset(
        root_dir = train_dir,
        sample_rate = 16000,
        class_mapping = class_mapping
    )

    train_dset, val_dset = random_split(dataset, [0.9, 0.1])

    assert len(train_dset) == 18
    assert len(val_dset) == 2


def test_item_data_types(mock_audio_dataset):
    train_dir, test_dir = mock_audio_dataset
    class_mapping = get_class_mapping(train_dir)

    dataset = AudioClassificationDataset(
        root_dir = train_dir,
        sample_rate = 16000,
        class_mapping = class_mapping
    )

    print(dataset[0]['feature'])

    assert isinstance(dataset[0]['feature'], np.ndarray)
    assert isinstance(dataset[0]['class_id'], int)
    assert isinstance(dataset[0]['class_name'], str)