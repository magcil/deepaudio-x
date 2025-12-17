from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepaudiox.callbacks.checkpointer import Checkpointer
from deepaudiox.callbacks.console_logger import ConsoleLogger
from deepaudiox.callbacks.early_stopper import EarlyStopper
from deepaudiox.datasets.audio_classification_dataset import AudioClassificationDataset
from deepaudiox.modules.base_audio_classifier import BaseAudioClassifier
from deepaudiox.utils.training_utils import get_device, get_logger, pad_collate_fn, random_split_audio_dataset


@dataclass
class State:
    """Dataclass that stores variables accessed throught the training lifecycle.

    Attributes:
        current_epoch (int): The current epoch of the training process. Dafaults to 1.
        lowest_loss (float): The lowest loss achieved during training. Defaults to np.Inf.
        train_loss (list): An ordered list of train losses, by epoch.
        validation_loss (list): An orderd list of validation lossed, by epoch.
        early_stop (bool): Determines if training should be early stopped. Defaults to False.

    """

    current_epoch: int = 1
    lowest_loss: float = np.inf
    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    early_stop: bool = False


class Trainer:
    """The core SDK module for training a model.

    The Trainer assembles all modules required for training
    and performs the training process.

    Attributes:
        state (State): Stores training variables.
        epochs (int): The maximum number of training epochs.
        device (str): The device used for training.
        logger (logging.Logger): A module used for logging messages.
        train_dloader (torch.DataLoader): The DataLoader of the training set.
        validation_dloader (torch.DataLoader): The DataLoader of the validation set.
        model (BaseAudioClassifier): The BaseAudioClassifier to be trained.
        optimizer (torch.optim.Optimizer): The optimizer of the training process.
        lr_scheduler (torch.optim.Optimizer): The scheduler of the training process.
        loss_function (nn.Module): The loss function used for optimization.
        callbacks (list): A list of callbacks used throught the training lifecycle.

    """

    def __init__(
        self,
        train_dset: AudioClassificationDataset,
        model: BaseAudioClassifier,
        optimizer: torch.optim.Optimizer,
        loss_function: nn.Module | None,
        lr_scheduler: LRScheduler | None,
        device_index: int | None,
        train_ratio: float = 0.8,
        validation_dset: AudioClassificationDataset | None = None,
        epochs: int = 50,
        patience: int = 5,
        num_workers: int = 4,
        batch_size: int = 16,
        path_to_checkpoint: str = "checkpoint.pt",
    ):
        """Initialize the Trainer.

        Args:
            train_dset (AudioClassificationDataset): The training dataset.
            validation_dset (AudioClassificationDataset): The validation dataset.
            model (BaseAudioClassifier): The model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            loss_function (nn.Module | None): The loss function used for training. Uses CrossEntropy if None.
            lr_scheduler (LRScheduler | None): The scheduler used for training. No scheduler if None.
            train_ratio (float, optional): The ratio of the train split. Defaults to 0.8.
            epochs (int, optional): The maximum number of training epochs. Defaults to 50.
            patience (int, optional): The maximum number of epochs with no decrease in loss. Defaults to 5.
            num_workers (int, optional): The number of workers for Python Data Loaders. Defaults to 4.
            batch_size (int, optional): The batch size for Python Data Loaders. Defaults to 16.
            path_to_checkpoint (str, optional): The path to the saved model checpoint. Defaults to "checkpoint.pt".
            device_index (int | None): The GPU device index to use. If None, uses the default GPU if available.
        """
        # Configure training state
        self.state = State()
        self.epochs = epochs
        self.device = get_device(device_index=device_index)

        # Configure logger
        self.logger = get_logger()

        # Load datasets
        self.train_dloader, self.validation_dloader = self._setup_dataloaders(
            train_dset=train_dset,
            validation_dset=validation_dset,
            train_ratio=train_ratio,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # Load model and training modules
        self.model = model
        self.model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_function = loss_function or nn.CrossEntropyLoss()

        # Configure callbacks
        self.callbacks = [
            ConsoleLogger(logger=self.logger),
            Checkpointer(path_to_checkpoint=path_to_checkpoint, logger=self.logger),
            EarlyStopper(patience=patience, logger=self.logger),
        ]

    def train(self) -> None:
        """Perform the training process"""

        # Execute callbacks in the beginning of training
        for cb in self.callbacks:
            cb.on_train_start(self)

        # Initiate training
        for epoch in range(1, self.epochs + 1):
            if self.state.early_stop:
                break

            self.state.current_epoch = epoch
            train_loss, val_loss = 0.0, 0.0

            # Execute callbacks in the beginning of the epoch
            for cb in self.callbacks:
                cb.on_epoch_start(self)

            # Perform trainng
            self.model.train()
            with tqdm(self.train_dloader, unit="batch", leave=False, desc="Training phase") as tbar:
                # Optimize the model by batch
                for batch in tbar:
                    self.optimizer.zero_grad()
                    x = batch["feature"].to(self.device)
                    y_true = batch["y_true"].to(self.device)
                    y_pred = self.model(x)
                    batch_loss = self.loss_function(y_pred, y_true)
                    batch_loss.backward()
                    self.optimizer.step()

                    train_loss += batch_loss.item()

                train_loss /= max(1, len(self.train_dloader))

            # Perform validation
            self.model.eval()
            with torch.no_grad():
                with tqdm(self.validation_dloader, unit="batch", leave=False, desc="Validation phase") as vbar:
                    # Compute validation loss by batch
                    for batch in vbar:
                        x = batch["feature"].to(self.device)
                        y_true = batch["y_true"].to(self.device)
                        y_pred = self.model(x)
                        batch_loss = self.loss_function(y_pred, y_true)
                        val_loss += batch_loss.item()

                val_loss /= len(self.validation_dloader)

            # Update scheduling
            if self.scheduler:
                self.scheduler.step()

            # Update training state
            self.state.train_loss.append(train_loss)
            self.state.validation_loss.append(val_loss)

            # Execute callbacks at the end of the epoch
            for cb in self.callbacks:
                cb.on_epoch_end(self)

        # Execute callbacks at the end of training
        for cb in self.callbacks:
            cb.on_train_end(self)

        return

    def _setup_dataloaders(
        self,
        train_dset: AudioClassificationDataset,
        validation_dset: AudioClassificationDataset | None,
        train_ratio: float,
        batch_size: int,
        num_workers: int,
    ):
        """Generate PyTorch DataLoaders for training and validation splits.

        Args:
            train_dset (AudioClassificationDataset): Training dataset.
            validation_dset (AudioClassificationDataset): Validation dataset.
            batch_size (int, optional): The batch size for Python Data Loaders. Defaults to 16.
            num_workers (int, optional): The number of workers for Python Data Loaders. Defaults to 4.
            train_ratio (float, optional): The ratio of the train split. Defaults to 0.8.

        """
        # Split to train and validation
        if validation_dset is None:
            train_dataset, validation_dataset = random_split_audio_dataset(train_dset, train_ratio)
        else:
            train_dataset = train_dset
            validation_dataset = validation_dset

        # Produce DataLoaders
        train_dloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=pad_collate_fn,
        )

        validation_dloader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=pad_collate_fn,
        )

        return train_dloader, validation_dloader
