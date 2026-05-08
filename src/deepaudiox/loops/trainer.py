import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepaudiox.callbacks.base_callback import BaseCallback
from deepaudiox.callbacks.checkpointer import Checkpointer
from deepaudiox.callbacks.early_stopper import EarlyStopper
from deepaudiox.datasets.audio_classification_dataset import AudioClassificationDataset
from deepaudiox.modules.baseclasses import BaseAudioClassifier
from deepaudiox.schemas.types import DeviceName
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
    current_patience: int = 0


class Trainer:
    """The core SDK module for training a model.

    The Trainer assembles all modules required for training
    and performs the training process.

    Attributes:
        state (State): Stores training variables.
        epochs (int): The maximum number of training epochs.
        verbose (bool): Whether to log epoch-level artifacts.
        device (str): The device used for training.
        logger (logging.Logger): A module used for logging messages.
        train_dloader (torch.DataLoader): The DataLoader of the training set.
        validation_dloader (torch.DataLoader): The DataLoader of the validation set.
        model (BaseAudioClassifier): The BaseAudioClassifier to be trained.
        optimizer (torch.optim.Optimizer): The optimizer of the training process.
        scheduler (LRScheduler): The learning rate scheduler of the training process.
        loss_function (nn.Module): The loss function used for optimization.
        callbacks (list): A list of callbacks used throughout the training lifecycle.

    """

    def __init__(
        self,
        train_dset: AudioClassificationDataset,
        model: BaseAudioClassifier,
        validation_dset: AudioClassificationDataset | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        learning_rate: float = 1e-3,
        lr_scheduler: LRScheduler | None = None,
        loss_function: nn.Module | None = None,
        train_ratio: float = 0.8,
        epochs: int = 100,
        patience: int | None = None,
        num_workers: int = 4,
        batch_size: int = 16,
        path_to_checkpoint: str = "checkpoint.pt",
        device: DeviceName = "cpu",
        device_index: int | None = None,
        verbose: bool = True,
    ):
        """Initialize the Trainer.

        Args:
            train_dset (AudioClassificationDataset): The training dataset.
            model (BaseAudioClassifier): The model to be trained.
            validation_dset (AudioClassificationDataset | None): The validation dataset. If None, a split is created
                from train_dset using train_ratio.
            optimizer (torch.optim.Optimizer): The optimizer used for training. Adam if None.
            learning_rate (float): Learning rate used when optimizer is None. Defaults to 1e-3.
            lr_scheduler (LRScheduler | None): The scheduler used for training. ReduceLROnPlateau if None.
            loss_function (nn.Module | None): The loss function used for training. Uses CrossEntropy if None.
            train_ratio (float, optional): The ratio of the train split when validation_dset is None. Defaults to 0.8.
            epochs (int, optional): The maximum number of training epochs. Defaults to 100.
            patience (int | None): Epochs to wait without loss improvement before stopping. Disabled if None.
            num_workers (int, optional): The number of workers for Python Data Loaders. Defaults to 4.
            batch_size (int, optional): The batch size for Python Data Loaders. Defaults to 16.
            path_to_checkpoint (str, optional): The path to the saved model checpoint. Defaults to "checkpoint.pt".
            device (DeviceName): The device to use for training. One of ``"cuda"``, ``"mps"``, or ``"cpu"``.
                Defaults to ``"cpu"``.
            device_index (int | None): The GPU device index. Only applicable when ``device="cuda"``.
                If ``None``, uses the default CUDA device.
            verbose (bool): If True, logs epoch-level artifacts (loss, time). If False, only
                start/end messages and the final training summary are printed. Defaults to True.

        Example:
            >>> from deepaudiox import AudioClassifier, Trainer
            >>> from deepaudiox import audio_classification_dataset_from_dir, get_class_mapping_from_dir
            >>> class_mapping = get_class_mapping_from_dir(root_dir="path/to/data")
            >>> train_dataset = audio_classification_dataset_from_dir(
            ...     root_dir="path/to/data",
            ...     sample_rate=16_000,
            ...     class_mapping=class_mapping,
            ... )
            >>> model = AudioClassifier(num_classes=len(class_mapping), backbone="beats", sample_rate=16_000)
            >>> trainer = Trainer(train_dset=train_dataset, model=model, epochs=10)
            >>> trainer.train()
        """
        # Configure training state
        self.state = State()
        self.epochs = epochs
        self.verbose = verbose
        self.path_to_checkpoint = path_to_checkpoint
        self.device = get_device(device=device, device_index=device_index)

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
        self.optimizer = optimizer or Adam(params=self.model.parameters(), lr=learning_rate)
        self.scheduler = lr_scheduler or ReduceLROnPlateau(self.optimizer, "min")
        self.loss_function = loss_function or nn.CrossEntropyLoss()

        # Configure callbacks — Checkpointer must precede EarlyStopper (updates lowest_loss first)
        self.callbacks: list[BaseCallback] = [Checkpointer(path_to_checkpoint=path_to_checkpoint, logger=self.logger)]
        if patience:
            self.callbacks.append(EarlyStopper(patience=patience, logger=self.logger))

        self._epoch_start_time: float = 0.0

    def train_step(self) -> float:
        """Run one pass over the training set.

        Sets the model to train mode, iterates over ``train_dloader``,
        performs forward + backward + optimizer step per batch.

        Returns:
            float: Average training loss over the epoch.
        """
        train_loss = 0.0
        self.model.train()
        with tqdm(self.train_dloader, unit="batch", leave=False, desc="Training phase") as tbar:
            for batch in tbar:
                self.optimizer.zero_grad()
                x = batch["feature"].to(self.device)
                y_true = batch["y_true"].to(self.device)
                y_pred = self.model(x)
                batch_loss = self.loss_function(y_pred, y_true)
                batch_loss.backward()
                self.optimizer.step()
                train_loss += batch_loss.item()
        return train_loss / max(1, len(self.train_dloader))

    def val_step(self) -> float:
        """Run one pass over the validation set.

        Sets the model to eval mode, iterates over ``validation_dloader``
        under ``torch.no_grad()``.

        Returns:
            float: Average validation loss over the epoch.
        """
        val_loss = 0.0
        self.model.eval()
        with torch.no_grad(), tqdm(self.validation_dloader, unit="batch", leave=False, desc="Validation phase") as vbar:
            for batch in vbar:
                x = batch["feature"].to(self.device)
                y_true = batch["y_true"].to(self.device)
                y_pred = self.model(x)
                batch_loss = self.loss_function(y_pred, y_true)
                val_loss += batch_loss.item()
        return val_loss / len(self.validation_dloader)

    def epoch_step(self) -> tuple[float, float]:
        """Run one complete training epoch.

        Logs the epoch header and metrics when ``verbose=True``, calls ``train_step()``
        and ``val_step()``, updates the LR scheduler and
        ``self.state``, then executes ``on_epoch_end`` callbacks (which may trigger
        early stopping or checkpointing).

        Note:
            ``self.state.current_epoch`` must be set by the caller before
            invoking this method — ``train()`` does this automatically.
            When calling ``epoch_step()`` directly, set it yourself:
            ``trainer.state.current_epoch = epoch``.

        Returns:
            tuple[float, float]: ``(train_loss, val_loss)`` for the epoch.

        Example:
            >>> from deepaudiox import AudioClassifier, Trainer
            >>> from deepaudiox import audio_classification_dataset_from_dir, get_class_mapping_from_dir
            >>> class_mapping = get_class_mapping_from_dir(root_dir="path/to/data")
            >>> train_dataset = audio_classification_dataset_from_dir(
            ...     root_dir="path/to/data",
            ...     sample_rate=16_000,
            ...     class_mapping=class_mapping,
            ... )
            >>> model = AudioClassifier(num_classes=len(class_mapping), backbone="beats", sample_rate=16_000)
            >>> trainer = Trainer(train_dset=train_dataset, model=model, epochs=10)
            >>> for epoch in range(1, trainer.epochs + 1):
            ...     trainer.state.current_epoch = epoch
            ...     train_loss, val_loss = trainer.epoch_step()
            ...     print(f"Epoch {epoch} — train: {train_loss:.4f}, val: {val_loss:.4f}")
            ...     if trainer.state.early_stop:
            ...         break
        """
        self._epoch_start_time = time.time()
        if self.verbose:
            self.logger.info(f"[Epoch {self.state.current_epoch}/{self.epochs}]")

        for cb in self.callbacks:
            cb.on_epoch_start(self)

        train_loss = self.train_step()
        val_loss = self.val_step()

        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

        self.state.train_loss.append(train_loss)
        self.state.validation_loss.append(val_loss)

        if self.verbose:
            elapsed = time.time() - self._epoch_start_time
            self.logger.info(
                f"Epoch {self.state.current_epoch} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val. Loss: {val_loss:.4f} | "
                f"Time: {elapsed:.2f}s"
            )

        for cb in self.callbacks:
            cb.on_epoch_end(self)

        return train_loss, val_loss

    def train(self) -> None:
        """Perform the full training process.

        Epoch-level output is controlled by ``verbose``. The training summary
        (best epoch, losses, checkpoint path) is always printed on completion.
        """
        self.logger.info("Training has started.")

        for cb in self.callbacks:
            cb.on_train_start(self)

        for epoch in range(1, self.epochs + 1):
            if self.state.early_stop:
                break
            self.state.current_epoch = epoch
            self.epoch_step()

        for cb in self.callbacks:
            cb.on_train_end(self)

        best_idx = int(np.argmin(self.state.validation_loss))
        best_val_loss = self.state.validation_loss[best_idx]
        best_train_loss = self.state.train_loss[best_idx]
        best_epoch = best_idx + 1
        sep = "─" * 52
        self.logger.info(
            f"\n{sep}\n"
            f" Training Complete\n"
            f"  Best Epoch    : {best_epoch}\n"
            f"  Train Loss    : {best_train_loss:.6f}\n"
            f"  Val. Loss     : {best_val_loss:.6f}\n"
            f"  Checkpoint    : {self.path_to_checkpoint}\n"
            f"{sep}"
        )

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
        pin_memory = self.device.type == "cuda"
        persistent_workers = num_workers > 0
        train_dloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=pad_collate_fn,
        )

        validation_dloader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=pad_collate_fn,
        )

        return train_dloader, validation_dloader
