from dataclasses import dataclass


@dataclass
class DataConfig:
    """Configuration for training and testing datasets.

    Attributes:
        train_dir (str): Path to the training data directory.
        test_dir (str): Path to the testing data directory. Defaults to None.
        sample_rate (int): The audio sampling rate. Defaults to 16000.
        batch_size (int): Number of samples per batch. Defaults to 16.
        num_workers (int): Number of worker threads for data loading. Defaults to 4.
        seed (int): Random seed for reproducibility. Defaults to 42.
        train_ratio (float): The ratio of train split. Defaults to 0.8.
        validation_ratio (float): The ratio of validation split. Defaults to 0.1.
        test_ratio (float): The ratio of test split. Defaults to 0.2.

    """

    train_dir: str
    test_dir: str = None
    sample_rate: int = 16000
    batch_size: int = 16
    num_workers: int = 4
    seed: int = 42
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.2

@dataclass
class ModelConfig:
    """Configuration for setting up a model.

    Attributes: Currently anavailable

    """
    pass

@dataclass
class LossConfig:
    """ Configuration for setting up loss functions.
        Every loss function config class inherits LossConfig.

    Attributes:
        name (str): The name used by the registry to build the configuration 
                    of a loss function. Defaults to CrossEntropyLoss.
    
    """
    name: str = "CrossEntropyLoss"

@dataclass
class SchedulingConfig:
    """ Configuration for setting up a scheduler.
        Every scheduling config class inherits SchedulingConfig.

    Attributes:
        name (str): The name of the scheduler. Defaults to CosineAnnealingLR.

    """
    epochs: int
    name: str = "CosineAnnealingLR"

@dataclass
class OptimizationConfig:
    """ Configuration for setting up optimizers.
        Every optimizer config class inherits LossConfig.

    Attributes:
        name (str): The name used by the registry to build the configuration 
                    of an optimizer. Defaults to ADAM.
    
    """
    name: str = "ADAM"