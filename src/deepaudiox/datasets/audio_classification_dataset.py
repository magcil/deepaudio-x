from pathlib import Path

import librosa
import soundfile as sf
from torch.utils.data import Dataset

from deepaudiox.dtos.dataset_items import AudioClassificationItem


class AudioClassificationDataset(Dataset):
    """PyTorch Dataset for audio classification tasks.

    This dataset loads audio files from a specified directory. Each
    item returned by the dataset contains the label; label id, and the waveform of the audio as numpy array.

    Attributes:
        file_to_class_mapping (dict): Mapping from file paths to class names.
        sample_rate (int): Target sampling rate for audio loading.
        class_mapping (dict): Mapping from string labels to integer IDs.
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
        self.sample_rate = sample_rate
        self.class_mapping = class_mapping
        self.file_to_class_mapping = file_to_class_mapping

        self.items = [
            AudioClassificationItem(
                path = Path(path),
                class_name = class_name,
                y_true = self.class_mapping[class_name]
            ) for path, class_name in file_to_class_mapping.items()
        ]
        
        self.segment_duration = segment_duration
        if self.segment_duration:
            self._apply_segmentation(segment_duration)


    def __len__(self) -> int:
        """Return the number of items in the dataset.

        Returns:
            int: Total number of samples.

        """
        return len(self.items)


    def __getitem__(self, idx: int) -> dict:
        """Get a single dataset item by index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: An AudioClassificationItem in the form of dictionary.

        """
        item = self.items[idx]

        item.feature = librosa.load(
            path=item.path,
            sr=self.sample_rate,
            mono=True,
            offset=item.segment_idx*self.segment_duration if self.segment_duration else 0,
            duration=self.segment_duration,
        )[0]

        return item.to_dict()


    def _apply_segmentation(self, segment_duration: float):
        """Segmentize all audio files into fixed-duration segments.
        Drops the last partial segment.
        """

        for item in list(self.items):
            with sf.SoundFile(item.path) as f:
                total_duration = len(f) / f.samplerate  # seconds

            if total_duration < segment_duration:
                continue  # or raise, depending on your policy

            num_segments = int(total_duration // segment_duration)

            # seg_idx=0 already exists
            for seg_idx in range(1, num_segments):
                self.items.append(
                    AudioClassificationItem(
                        path=item.path,
                        y_true=item.y_true,
                        segment_idx=seg_idx,
                        class_name=item.class_name,
                    )
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
