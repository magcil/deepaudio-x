from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ..data_classes.data_config import DataConfig
from .audio_classification_dataset import AudioClassificationDataset


class DatasetManager:
    """Manager for train, validation, and test audio datasets.

    This class handles loading audio file metadata from directories,
    splitting datasets into train and validation sets, generating
    PyTorch Dataset instances, and returning PyTorch
    DataLoaders for each split.

    Attributes:
        train_dir (str): Directory containing training data.
        test_dir (str): Directory containing test data.
        sample_rate (int): Sampling rate for audio data.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of workers for data loading.
        train_dataset (AudioClassificationDataset): Dataset for training.
        validation_dataset (AudioClassificationDataset): Dataset for validation.
        test_dataset (AudioClassificationDataset): Dataset for testing.
        train_split (list): Metadata for training split.
        validation_split (list): Metadata for validation split.
        test_split (list): Metadata for test split.

    """

    def __init__(self, config: DataConfig):
        """Initialize the DatasetManager.

        Args:
            config (DataConfig): Configuration containing settings.

        """
        self.train_dir = config.train_dir
        self.test_dir = config.test_dir
        self.sample_rate = config.sample_rate
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

        self.train_split = self._load_instance_metadata(config.train_dir)
        self.validation_split = []
        self.test_split = self._load_instance_metadata(config.test_dir)

        self._produce_splits(config.seed)
        self._generate_datasets()

    def _load_instance_metadata(self, root_dir: str):
        """Scan a given directory for class sub-folders and audio files and load metadata.

        Args:
            root_dir (str): Directory to scan for audio files.

        Returns:
            list: List of dictionaries with keys 'file_path' and 'label'.

        Raises:
            ValueError: If the given path is not a directory.

        """
        root_path = Path(root_dir)
        metadata = []

        if not root_path.is_dir():
            raise ValueError(f"The path '{root_dir}' is not a directory")

        for child_directory in root_path.iterdir():
            if child_directory.is_dir():
                for audio_file in child_directory.rglob("*.wav"):
                    metadata.append({"file_path": str(audio_file), "label": child_directory.name})
                for audio_file in child_directory.rglob("*.mp3"):
                    metadata.append({"file_path": str(audio_file), "label": child_directory.name})

        return metadata

    def _produce_splits(self, seed: int):
        """Split the training metadata into train and validation sets.

        Uses stratified splitting based on labels to ensure class
        balance in each subset.

        Args:
            seed (int): Random seed for reproducibility.

        Raises:
            ValueError: If no labels are found in the training split.

        """
        labels = [item["label"] for item in self.train_split]
        if len(labels) == 0:
            raise ValueError("No labels found in this dataset for split.")

        train_data, validation_data, _, _ = train_test_split(
            self.train_split, labels, stratify=labels, test_size=0.1, random_state=seed
        )

        self.train_split = train_data
        self.validation_split = validation_data

        return

    def _generate_datasets(self):
        """Generate PyTorch Dataset instances for each split."""
        self.train_dataset = AudioClassificationDataset(
            root_dir=self.train_dir, metadata=self.train_split, sample_rate=self.sample_rate
        )

        self.validation_dataset = AudioClassificationDataset(
            root_dir=self.train_dir, metadata=self.validation_split, sample_rate=self.sample_rate
        )

        self.test_dataset = AudioClassificationDataset(
            root_dir=self.test_dir, metadata=self.test_split, sample_rate=self.sample_rate
        )

    def get_dataset(self, split: str) -> AudioClassificationDataset:
        """Return the PyTorch dataset for the specified dataset split.

        Args:
            split (str): Name of the split, must be one of
                            'train', 'validation', or 'test'.

        Returns:
            DataLoader: AudioClassificationDataset (PyTorch) Dataset for the specified split.

        Raises:
            ValueError: If an invalid split name is provided.

        """
        split_map = {
            "train": self.train_dataset,
            "validation": self.validation_dataset,
            "test": self.test_dataset,
        }
        dataset = split_map.get(split)
        if dataset is None:
            raise ValueError(f"Invalid split name: {split}")

        return dataset

    def get_dataloader(self, split: str) -> DataLoader:
        """Return a DataLoader for the specified dataset split.

        Args:
            split (str): Name of the split, must be one of
                            'train', 'validation', or 'test'.

        Returns:
            DataLoader: PyTorch DataLoader for the specified split.

        Raises:
            ValueError: If an invalid split name is provided.

        """
        split_map = {
            "train": self.train_dataset,
            "validation": self.validation_dataset,
            "test": self.test_dataset,
        }
        dataset = split_map.get(split)
        if dataset is None:
            raise ValueError(f"Invalid split name: {split}")

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=(split == "train"), num_workers=self.num_workers)
