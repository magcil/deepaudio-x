from pathlib import Path
from typing import TypedDict

import numpy as np
from torch.utils.data import Dataset

from deepaudiox.utils.audio_utils import load_audio


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
        class_mapping: dict[str, int]
    ):
        """Initialize the dataset.

        Args:
            root_dir (str): Root directory containing the audio files.
            sample_rate (int): Target sampling rate for audio loading.
            class_mapping (dict): Mapping from string labels to integer IDs.

        """
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.class_mapping = class_mapping
        self.instances = self._load_instance_paths_and_classes(root_dir)

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
                    instances.append(
                        {
                            "path": str(audio_file),
                            "class_name": child_directory.name
                        }
                    )
                for audio_file in child_directory.rglob("*.mp3"):
                    instances.append(
                        {
                            "path": str(audio_file),
                            "class_name": child_directory.name
                        }
                    )

        return instances
    
    def __len__(self) -> int:
        """Return the number of items in the dataset.

        Returns:
            int: Total number of samples.

        """
        return len(self.instances)

    def __getitem__(self, idx: int) -> WaveDict:
        """Get a single dataset item by index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            WaveDict: A dictionary containing the class_id; class_name and the waveform.

        """
        item = self.instances[idx]
        
        waveform = load_audio(
            file_path=item['path'],
            start_sample=0,
            end_sample=None
        )

        return {
            "feature": waveform,
            "class_id": self.class_mapping[item['class_name']],
            "class_name": item['class_name']
        }
