import pytest
import torch
import torchaudio

from deepaudiox.data_classes.data_config import DataConfig
from deepaudiox.datasets.dataset_manager import DatasetManager

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
    """Test that the number of instances in each split is correct."""
    train_dir, test_dir = mock_audio_dataset
    data_config = DataConfig(train_dir=str(train_dir), test_dir=str(test_dir))
    dataset_manager = DatasetManager(data_config)

    train_dataset = dataset_manager.get_dataset("train")
    validation_dataset = dataset_manager.get_dataset("validation")
    test_dataset = dataset_manager.get_dataset("test")

    assert len(train_dataset) == 18
    assert len(validation_dataset) == 2
    assert len(test_dataset) == 4


def test_labelids_are_ints(mock_audio_dataset):
    """Test that the data type of label IDs is integer."""
    train_dir, test_dir = mock_audio_dataset
    data_config = DataConfig(train_dir=str(train_dir), test_dir=str(test_dir))
    dataset_manager = DatasetManager(data_config)

    train_dataset = dataset_manager.get_dataset("train")
    validation_dataset = dataset_manager.get_dataset("validation")
    test_dataset = dataset_manager.get_dataset("test")

    assert isinstance(train_dataset[0]["label_id"], int)
    assert isinstance(validation_dataset[0]["label_id"], int)
    assert isinstance(test_dataset[0]["label_id"], int)


def test_features_are_tensors(mock_audio_dataset):
    """Test that the data type of features is torch.Tensor."""
    train_dir, test_dir = mock_audio_dataset
    data_config = DataConfig(train_dir=str(train_dir), test_dir=str(test_dir))
    dataset_manager = DatasetManager(data_config)

    train_dataset = dataset_manager.get_dataset("train")
    validation_dataset = dataset_manager.get_dataset("validation")
    test_dataset = dataset_manager.get_dataset("test")

    assert isinstance(train_dataset[0]["feature"], torch.Tensor)
    assert isinstance(validation_dataset[0]["feature"], torch.Tensor)
    assert isinstance(test_dataset[0]["feature"], torch.Tensor)
