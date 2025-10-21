from pathlib import Path

from torch.utils.data import Dataset

from ..utils.audio_utils import load_audio


class AudioClassificationDataset(Dataset):
    """PyTorch Dataset for audio classification tasks.

    This dataset loads audio files and encodes their labels as integers. Each
    item returned by the dataset contains the label and the audio feature tensor.

    Attributes:
        root_dir (str): Root directory containing the audio files.
        sample_rate (int): Target sampling rate for audio loading.
        class_mapping (dict): Mapping from string labels to integer IDs.
        instance_paths (list): List of paths of the included instances.
        instance_classes (list): List of classes of the included instances.

    """

    def __init__(
        self, 
        root_dir: str, 
        sample_rate: int, 
        class_mapping: dict, 
        instance_paths: list,
        instance_classes: list
    ):
        """Initialize the dataset.

        Args:
            root_dir (str): Root directory containing the audio files.
            sample_rate (int): Target sampling rate for audio loading.
            class_mapping (dict): Mapping from string labels to integer IDs.
            instance_paths (list): List of paths of the included instances.
            instance_classes (list): List of classes of the included instances.

        """
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.class_mapping = class_mapping
        self.instance_paths = instance_paths if instance_paths else []
        self.instance_classes = instance_classes if instance_classes else []

        # Load instance paths and classes, if not provided
        if not self.instance_paths or not self.instance_classes:
            self._load_instance_paths_and_classes(root_dir)
        

    def _load_instance_paths_and_classes(self, root_dir: str):
        """Scan a given directory for class sub-folders and audio files and load metadata.

        Args:
            root_dir (str): Directory to scan for audio files.

        Returns:
            list: List of dictionaries with keys 'file_path' and 'label'.

        Raises:
            ValueError: If the given path is not a directory.

        """
        root_path = Path(root_dir)
        instance_paths = []
        instance_classes = []

        if not root_path.is_dir():
            raise ValueError(f"The path '{root_dir}' is not a directory")

        for child_directory in root_path.iterdir():
            if child_directory.is_dir():
                for audio_file in child_directory.rglob("*.wav"):
                    instance_paths.append(str(audio_file))
                    instance_classes.append(child_directory.name)
                for audio_file in child_directory.rglob("*.mp3"):
                    instance_paths.append(str(audio_file))
                    instance_classes.append(child_directory.name)

        self.instance_paths = instance_paths
        self.instance_classes = instance_classes

        return
    
    def __len__(self):
        """Return the number of items in the dataset.

        Returns:
            int: Total number of samples.

        """
        return len(self.instance_paths)

    def __getitem__(self, idx):
        """Get a single dataset item by index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing the label and the feature tensor.

        """
        instance_path = self.instance_paths[idx]
        instance_class = self.instance_classes[idx]
        instance_class_id = self.class_mapping[instance_class]

        waveform = load_audio(
            file_path=instance_path,
            start_sample=0,
            end_sample=None
        )

        return waveform, instance_class_id
