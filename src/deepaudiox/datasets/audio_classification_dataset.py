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

    This dataset loads audio files from a specified directory. Each
    item returned by the dataset contains the label; label id, and the waveform of the audio as numpy array.

    Attributes:
        root_dir (str): Root directory containing the audio files.
        sample_rate (int): Target sampling rate for audio loading.
        class_mapping (dict): Mapping from string labels to integer IDs.

    """

    def __init__(
        self,
        root_dir: str | Path,
        sample_rate: int,
        class_mapping: dict[str, int],
        segment_duration: float | None = None,
    ):
        """Initialize the dataset.

        Args:
            root_dir (str): Root directory containing the audio files.
            sample_rate (int): Target sampling rate for audio loading.
            class_mapping (dict): Mapping from string labels to integer IDs.
            segment_duration (float | None): Duration of audio segments in seconds. If None, load full audio.
            drop_corrupted (bool): Whether to drop corrupted audio files. Defaults to False.
        """
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.class_mapping = class_mapping
        self.segment_duration = segment_duration
        self.instances = self._load_instance_paths_and_classes(root_dir)
        self.segment_map = []

        if self.segment_duration is not None:
            self.segmentize_audios(self.segment_duration)

    def _load_instance_paths_and_classes(self, root_dir: str | Path) -> list[dict[str, str]]:
        """Scan a given directory for class sub-folders and audio files and load metadata.

        Args:
            root_dir (str): Directory to scan for audio files.

        Returns:
            list: List of dictionaries with keys 'file_path' and 'label'.

        """
        root_path = Path(root_dir)
        instances = []

        if not root_path.is_dir():
            raise ValueError(f"The path '{root_dir}' is not a directory")

        for child_directory in root_path.iterdir():
            if child_directory.is_dir():
                for audio_file in child_directory.rglob("*.wav"):
                    instances.append({"path": str(audio_file), "class_name": child_directory.name})
                for audio_file in child_directory.rglob("*.mp3"):
                    instances.append({"path": str(audio_file), "class_name": child_directory.name})

        return instances

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
                "class_name": item["class_name"]
            }

        else:
            item = self.instances[idx]

            waveform, _ = librosa.load(path=item["path"], sr=self.sample_rate, mono=True)

            return {
                "feature": waveform,
                "class_id": self.class_mapping[item["class_name"]],
                "class_name": item["class_name"]
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
