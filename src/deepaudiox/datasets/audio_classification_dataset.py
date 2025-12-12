from pathlib import Path
from typing import TypedDict

import librosa
import numpy as np
from torch.utils.data import Dataset


class WaveDict(TypedDict):
    feature: np.ndarray
    class_id: int
    class_name: str


class AudioClassificationDataset(Dataset):
    """PyTorch Dataset for audio classification tasks.

    This dataset loads audio files and returns a dictionary containing the raw waveform
    (under the key ``"feature"``), the corresponding class name, and the integer class ID
    defined in ``class_mapping``. The ``file_to_class_mapping`` argument must be a dictionary
    of the form::

        {"abs/path/to/audio.wav": "class_name"}

    Optionally, the dataset can segment each audio file into fixed-duration chunks using
    ``segment_duration``. When enabled, each segment becomes an individual dataset sample.

    Attributes:
        file_to_class_mapping (dict): Mapping from file paths to class names.
        sample_rate (int): Target sampling rate for audio loading.
        class_mapping (dict): Mapping from string class labels to integer IDs.
    """

    def __init__(
        self,
        file_to_class_mapping: dict[str, str],
        sample_rate: int,
        class_mapping: dict[str, int],
        segment_duration: float | None = None,
    ):
        """Initialize the dataset.

        Args:
            file_to_class_mapping (dict): Mapping from file paths to class names.
            sample_rate (int): Target sampling rate for audio loading.
            class_mapping (dict): Mapping from string labels to integer IDs.
            segment_duration (float | None): Duration of audio segments in seconds. If None, load full audio.
        """

        self.file_to_class_mapping = file_to_class_mapping
        self.instances = [
            {"path": Path(path), "class_name": class_name} for path, class_name in file_to_class_mapping.items()
        ]
        self.segment_map = []

        self.sample_rate = sample_rate
        self.class_mapping = class_mapping
        self.segment_duration = segment_duration

        if self.segment_duration is not None:
            self.segmentize_audios(self.segment_duration)

    def __len__(self) -> int:
        """Return the number of items in the dataset.

        Returns:
            int: Total number of samples.

        """
        return len(self.segment_map) if self.segment_map else len(self.instances)

    def __getitem__(self, idx: int) -> WaveDict:
        """Get a single dataset item by index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            WaveDict: A dictionary containing the class_id; class_name and the waveform.

        """

        # If segmentize is true
        if self.segment_map:
            item = self.segment_map[idx]
            segment_idx = item["segment_idx"]

            waveform, _ = librosa.load(
                path=item["file_path"],
                sr=self.sample_rate,
                mono=True,
                offset=segment_idx * self.segment_duration,
                duration=self.segment_duration,
            )

            return {
                "feature": waveform,
                "class_id": self.class_mapping[item["class_name"]],
                "class_name": item["class_name"],
            }

        else:
            item = self.instances[idx]

            waveform, _ = librosa.load(path=item["path"], sr=self.sample_rate, mono=True)

            return {
                "feature": waveform,
                "class_id": self.class_mapping[item["class_name"]],
                "class_name": item["class_name"],
            }

    def segmentize_audios(self, segment_duration: float):
        """Segmentize all audio files in the dataset into fixed-duration segments.

        Args:
            segment_duration (int): Duration of each segment in seconds.

        """
        for item in self.instances:
            waveform, _ = librosa.load(path=item["path"], sr=self.sample_rate, mono=True)
            total_samples = waveform.shape[0]
            segment_samples = int(segment_duration * self.sample_rate)
            num_segments = max(1, total_samples // segment_samples)

            for seg_idx in range(num_segments):
                self.segment_map.append(
                    {"file_path": item["path"], "class_name": item["class_name"], "segment_idx": seg_idx}
                )


def audio_classification_dataset_from_dir(
    root_dir: str,
    sample_rate: int,
    class_mapping: dict[str, int],
    segment_duration: float | None = None,
) -> AudioClassificationDataset:
    """Create an AudioClassificationDataset from a directory structure.

    Args:
        root_dir (str | Path): Root directory containing class sub-folders.
        sample_rate (int): Target sampling rate for audio loading.
        class_mapping (dict): Mapping from string labels to integer IDs.
        segment_duration (float | None): Duration of audio segments in seconds. If None, load full audio.

    Returns:
        AudioClassificationDataset: The constructed dataset.
    """
    root_path = Path(root_dir)
    file_to_class_mapping = {}

    subdirs = [d for d in root_path.iterdir() if d.is_dir()]
    for _idx, subdir in enumerate(sorted(subdirs)):
        audio_files = list(subdir.glob("**/*.wav")) + list(subdir.glob("**/*.mp3"))
        for audio_file in audio_files:
            file_to_class_mapping[audio_file] = subdir.name

    return AudioClassificationDataset(
        file_to_class_mapping=file_to_class_mapping,
        sample_rate=sample_rate,
        class_mapping=class_mapping,
        segment_duration=segment_duration,
    )


def audio_classification_dataset_from_dictionary(
    file_to_class_mapping: dict[str, str],
    sample_rate: int,
    class_mapping: dict[str, int],
    segment_duration: float | None = None,
) -> AudioClassificationDataset:
    """Create an AudioClassificationDataset from a file-to-class mapping dictionary.

    Args:
        file_to_class_mapping (dict): Mapping from file paths to class names.
        sample_rate (int): Target sampling rate for audio loading.
        class_mapping (dict): Mapping from string labels to integer IDs.
        segment_duration (float | None): Duration of audio segments in seconds. If None, load full audio.

    Returns:
        AudioClassificationDataset: The constructed dataset.
    """
    return AudioClassificationDataset(
        file_to_class_mapping=file_to_class_mapping,
        sample_rate=sample_rate,
        class_mapping=class_mapping,
        segment_duration=segment_duration,
    )
