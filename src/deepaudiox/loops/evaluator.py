from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepaudiox.callbacks.console_logger import ConsoleLogger
from deepaudiox.callbacks.reporter import Reporter
from deepaudiox.datasets.audio_classification_dataset import AudioClassificationDataset
from deepaudiox.modules.baseclasses import BaseAudioClassifier
from deepaudiox.utils.training_utils import get_device, get_logger, pad_collate_fn


@dataclass
class State:
    """Dataclass that stores variables accessed throughout the testing lifecycle.

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
        device_index: int | None = None,
    ):
        """Initialize the Evaluator.

        Args:
            test_dset (AudioClassificationDataset): The testing dataset.
            model (BaseAudioClassifier): An AudioClassifier module inhereting from BaseAudioClassifier.
            class_mapping (dict): A mapping between class names and IDs.
            batch_size (int, optional): The batch size for Python Data Loaders. Defaults to 16.
            num_workers (int, optional): The number of workers for Python Data Loaders. Defaults to 4.
            device_index (int | None): The GPU device index to use. If None, uses the default GPU if available.

        Example:
            >>> import torch
            >>> from deepaudiox import AudioClassifier, Evaluator
            >>> from deepaudiox import audio_classification_dataset_from_dir, get_class_mapping_from_dir
            >>> class_mapping = get_class_mapping_from_dir(root_dir="path/to/data")
            >>> test_dataset = audio_classification_dataset_from_dir(
            ...     root_dir="path/to/data",
            ...     sample_rate=16_000,
            ...     class_mapping=class_mapping,
            ... )
            >>> model = AudioClassifier(num_classes=len(class_mapping), backbone="beats", sample_rate=16_000)
            >>> model.load_state_dict(torch.load("checkpoint.pt"))
            >>> evaluator = Evaluator(test_dset=test_dataset, model=model, class_mapping=class_mapping)
            >>> evaluator.evaluate()
        """
        self.state = State()
        self.device = get_device(device_index=device_index)
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

    @torch.inference_mode()
    def evaluate(self) -> None:
        """Run the full evaluation loop over the test set.

        Iterates over all batches in ``test_dloader``, accumulates true labels,
        predicted labels, and posterior probabilities into ``self.state``, then
        triggers the registered callbacks via ``on_testing_end``.

        After this method returns, ``self.state`` holds:
            - ``y_true`` (np.ndarray): Ground-truth class indices, shape (N,).
            - ``y_pred`` (np.ndarray): Predicted class indices, shape (N,).
            - ``posteriors`` (np.ndarray): Max posterior probability per sample, shape (N,).

        Note:
            The model is expected to already be in eval mode (set in ``__init__``).
            Runs under ``torch.inference_mode()`` — gradients are fully disabled.
            Callbacks (``ConsoleLogger``, ``Reporter``) are executed once after the loop.
        """
        # Lists to accumulate evaluation results, i.e., true_labels, prediction_labels, and posteriors
        y_true_batches, y_pred_batches, posterior_batches = [], [], []
        with tqdm(self.test_dloader, unit="batch", leave=False, desc="Evaluation phase") as tbar:
            for batch in tbar:
                # Move inputs
                x = batch["feature"].to(self.device)
                y_true = batch["y_true"].cpu().numpy()

                # Run model prediction
                inference = self.model.predict(x)
                y_pred = np.array(inference["y_preds"], dtype=int)
                post = np.array(inference["posteriors"], dtype=float)

                # Update lists with new batch results
                y_true_batches.append(y_true)
                y_pred_batches.append(y_pred)
                posterior_batches.append(post)

        # Concatenate all results outside the loop
        self.state.y_true = np.concatenate(y_true_batches)
        self.state.y_pred = np.concatenate(y_pred_batches)
        self.state.posteriors = np.concatenate(posterior_batches)

        # Execute callbacks at the end of testing
        for cb in self.callbacks:
            cb.on_testing_end(self)
