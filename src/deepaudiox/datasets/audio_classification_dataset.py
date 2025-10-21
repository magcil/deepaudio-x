from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from ..utils.audio_utils import load_audio


class AudioClassificationDataset(Dataset):
    """PyTorch Dataset for audio classification tasks.

    This dataset loads audio files and encodes their labels as integers. Each
    item returned by the dataset contains the label and the audio feature tensor.

    Attributes:
        root_dir (str): Root directory containing the audio files.
        metadata (list): List of dictionaries with keys including 'file_path' and 'label'.
        sample_rate (int): Target sampling rate for audio loading.
        label_to_id (dict): Mapping from string labels to integer IDs.

    """

    def __init__(self, root_dir: str, metadata: list, sample_rate: int):
        """Initialize the dataset.

        Args:
            root_dir (str): Root directory containing the audio files.
            metadata (list): List of dictionaries with metadata for each audio file.
            sample_rate (int): Target sampling rate for audio loading.

        """
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.metadata = metadata

        labels = [instance_dict["label"] for instance_dict in self.metadata]
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        self.label_to_id = {label: idx for idx, label in enumerate(label_encoder.classes_)}

    def __len__(self):
        """Return the number of items in the dataset.

        Returns:
            int: Total number of samples.

        """
        return len(self.metadata)

    def __getitem__(self, idx):
        """Get a single dataset item by index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing the label and the feature tensor.

        """
        instance_item = self.metadata[idx]

        waveform = load_audio(
            file_path=instance_item["file_path"],
            start_sample=instance_item.get("start_sample", 0),
            end_sample=instance_item.get("end_sample", None),
        )
        
        return {
            "feature": waveform,
            "label": instance_item["label"],
            "label_id": self.label_to_id[instance_item["label"]],
        }