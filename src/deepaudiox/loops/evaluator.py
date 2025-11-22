from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepaudiox.callbacks.console_logger import ConsoleLogger
from deepaudiox.callbacks.reporter import Reporter
from deepaudiox.datasets.audio_classification_dataset import AudioClassificationDataset
from deepaudiox.modules.base_audio_classifier import BaseAudioClassifier
from deepaudiox.utils.training_utils import get_device, get_logger, pad_collate_fn


@dataclass
class State:
    """Dataclass that stores variables
        accessed throughout the testing lifecycle.

    Attributes:
        y_true (np.ndarray): A NumPy array of true labels.
        y_pred (np.ndarray): A NumPy array of predicted labels.
        posteriors (np.ndarray): A NumPy array of posterior probabilities.
    """

    y_true: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    y_pred: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    posteriors: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))


class Evaluator:
    """The core SDK module for testing a model.

    The Evaluator assembles all modules required for testing
    and performs the testing process.

    Attributes:
        state (State): Stores testing variables.
        device (str): The device used for testing.
        class_mapping (dict): A mapping between class names and IDs.
        logger (logging.Logger): A module used for logging messages.
        test_dloader (torch.DataLoader): The DataLoader of the testing set.
        model (BaseAudioClassifier): An AudioClassifier module inhereting from BaseAudioClassifier.
        callbacks (list): A list of callbacks used throughout the testing lifecycle.
    """

    def __init__(
        self,
        test_dset: AudioClassificationDataset,
        model: BaseAudioClassifier,
        class_mapping: dict,
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        """Initialize the Evaluator.

        Args:
            test_dset (AudioClassificationDataset): The testing dataset.
            model (BaseAudioClassifier): An AudioClassifier module inhereting from BaseAudioClassifier.
            class_mapping (dict): A mapping between class names and IDs.
            batch_size (int, optional): The batch size for Python Data Loaders. Defaults to 16.
            num_workers (int, optional): The number of workers for Python Data Loaders. Defaults to 4.
        """
        self.state = State()
        self.device = get_device()
        self.class_mapping = class_mapping

        # Configure logger
        self.logger = get_logger()

        # Load dataset
        self.test_dloader = DataLoader(
            test_dset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=pad_collate_fn,
        )

        # Load model
        self.model = model
        self.model.to(self.device)
        self.model.eval()

        # Configure callbacks
        self.callbacks = [ConsoleLogger(logger=self.logger), Reporter(logger=self.logger)]

    def evaluate(self):
        """Perform the testing process."""
        with torch.no_grad(), tqdm(self.test_dloader, unit="batch", leave=False, desc="Evaluation phase") as vbatch:
            for _i, item in enumerate(vbatch, 1):
                # Move inputs
                features = item["feature"].to(self.device)
                y_true = item["class_id"].cpu().numpy()

                # Run model prediction
                inference = self.model.predict(features)
                y_pred = np.array(inference["y_preds"], dtype=int)
                post = np.array(inference["posteriors"], dtype=float)

                # Update testing state (NumPy arrays)
                self.state.y_true = np.concatenate([self.state.y_true, y_true])
                self.state.y_pred = np.concatenate([self.state.y_pred, y_pred])
                self.state.posteriors = np.concatenate([self.state.posteriors, post])

        # Execute callbacks at the end of testing
        for cb in self.callbacks:
            cb.on_testing_end(self)
