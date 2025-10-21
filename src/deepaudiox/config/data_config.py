from dataclasses import dataclass


@dataclass
class DataConfig:
    """Configuration for training and testing datasets.

    Attributes:
        train_dir (str): Path to the training data directory.
        test_dir (str): Path to the testing data directory.
        sample_rate (int, optional): The audio sampling rate. Defaults to 16000.
        batch_size (int, optional): Number of samples per batch. Defaults to 16.
        num_workers (int, optional): Number of worker threads for data loading. Defaults to 4.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    """

    train_dir: str
    test_dir: str
    sample_rate: int = 16000
    batch_size: int = 16
    num_workers: int = 4
    seed: int = 42
