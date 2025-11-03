import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from src.deepaudiox.callbacks.checkpointer import Checkpointer
from src.deepaudiox.callbacks.console_logger import ConsoleLogger
from src.deepaudiox.callbacks.early_stopper import EarlyStopper
from src.deepaudiox.datasets.audio_classification_dataset import AudioClassificationDataset
from src.deepaudiox.utils.training_utils import get_device, pad_collate_fn


@dataclass
class State:
    """ Dataclass that stores variables 
        accessed throught the training lifecycle.

    Attributes:
        current_epoch (int): The current epoch of the training process. Dafaults to 1.
        lowest_loss (float): The lowest loss achieved during training. Defaults to np.Inf.
        train_loss (list): An ordered list of train losses, by epoch.
        validation_loss (list): An orderd list of validation lossed, by epoch.
        early_stop (bool): Determines if training should be early stopped. Defaults to False.
    
    """
    current_epoch = 1
    lowest_loss = np.inf
    train_loss = []
    validation_loss = []
    early_stop = False

class Trainer:
    """ The core SDK module for training a model.
    
    The Trainer assembles all modules required for training 
    and performs the training process.

    Attributes:
        state (State): Stores training variables.
        epochs (int): The maximum number of training epochs.
        device (str): The device used for training.
        logger (object): A module used for logging messages.
        train_dloader (torch.DataLoader): The DataLoader of the training set.
        validation_dloader (torch.DataLoader): The DataLoader of the validation set.
        model (nn.Module): The trained model.
        optimizer (torch.optim.Optimizer): The optimizer of the training process.
        lr_scheduler (torch.optim.Optimizer): The scheduler of the training process.
        loss_function (nn.Module): The loss function used for optimization.
        callbacks (list): A list of callbacks used throught the training lifecycle.

    """
    def __init__(
        self,
        train_dset: AudioClassificationDataset,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_function: nn.Module,
        lr_scheduler: LRScheduler,
        train_ratio: float = 0.8,
        epochs: int = 10,
        learning_rate: float = 1e-3,
        patience: int = 5,
        num_workers: int = 4,
        batch_size: int = 16,
        path_to_checkpoint: str = "checkpoint.pt"
    ):  
        """ Initialize the Trainer.

        Args:
            train_dset (AudioClassificationDataset): The training dataset.
            model (nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            loss_function (nn.Module): The loss function used for training.
            lr_scheduler (LRScheduler): The scheduler used for training.
            train_ratio (float, optional): The ratio of the train split. Defaults to 0.8.
            epochs (int, optional): The maximum number of training epochs. Defaults to 10.
            learning_rate (float, optional): The learning rate used for optimization. Defaults to 1e-3.
            patience (int, optional): The maximum number of epochs with no decrease in loss. Defaults to 5.
            num_workers (int, optional): The number of workers for Python Data Loaders. Defaults to 4.
            batch_size (int, optional): The batch size for Python Data Loaders. Defaults to 16.
            path_to_checkpoint (str, optional): The path to the saved model checpoint. Defaults to "checkpoint.pt".

        """
        # Configure training state
        self.state = State()
        self.epochs = epochs
        self.device = get_device()

        # Configure logger
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        self.logger = logging.getLogger("ConsoleLogger")

        # Load datasets
        self.train_dloader = None
        self.validation_dloader = None
        self._setup_dataloaders(
            train_dset = train_dset,
            train_ratio = train_ratio,
            batch_size = batch_size,
            num_workers = num_workers
        )

        # Load mock model
        self.model = model
        self.model.to(self.device)

        # Configure optimizer
        self.optimizer = optimizer

        # Configure scheduler
        self.scheduler = lr_scheduler

        # Configure loss function
        self.loss_function = loss_function

        # Configure callbacks
        self.callbacks = [
            ConsoleLogger(logger=self.logger),
            Checkpointer(path_to_checkpoint=path_to_checkpoint, logger=self.logger),
            EarlyStopper(patience=patience, logger=self.logger)
        ]

    def train(self):
        """Perform the training process"""

        # Execute callbacks in the beginning of training
        for cb in self.callbacks: 
            cb.on_train_start(self)

        # Initiate training
        for epoch in range(1, self.epochs + 1):
            self.state.current_epoch = epoch
            train_loss, val_loss = 0.0, 0.0

            if not self.state.early_stop:
                # Execute callbacks in the beginning of the epoch
                for cb in self.callbacks: 
                    cb.on_epoch_start(self)

                # Execute training loop
                self.model.train()
                with tqdm(self.train_dloader, unit="batch", leave=False, desc="Training phase") as tbatch:
                    # Optimize the model by batch
                    for _i, item in enumerate(tbatch, 1):
                        self.optimizer.zero_grad()
                        features = item['feature'].to(self.device)
                        y_true = item['class_id'].to(self.device)
                        y_pred = self.model(features) 
                        batch_loss = self.loss_function(y_pred, y_true)                    
                        batch_loss.backward()
                        train_loss += batch_loss.item()
                        self.optimizer.step()

                    train_loss /= len(self.train_dloader)
                    self.scheduler.step()

                # Execute validation loop
                self.model.eval()
                with torch.no_grad():
                    with tqdm(self.validation_dloader, unit="batch", leave=False, desc="Validation phase") as vbatch:
                        # Compute validation loss by batch
                        for _i, item in enumerate(vbatch, 1):
                            features = item['feature'].to(self.device)
                            y_true = item['class_id'].to(self.device)
                            y_pred = self.model(features) 
                            batch_loss = self.loss_function(y_pred, y_true)
                            val_loss += batch_loss.item()
                            
                    val_loss /= len(self.validation_dloader)

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
        train_ratio: float,
        batch_size: int,
        num_workers: int
    ):
        """Generate PyTorch DataLoaders for training and validation splits.
        
        Arguments:
            train_dset (AudioClassificationDataset): The training dataset.
            batch_size (int, optional): The batch size for Python Data Loaders. Defaults to 16.
            num_workers (int, optional): The number of workers for Python Data Loaders. Defaults to 4.
            train_ratio (float, optional): The ratio of the train split. Defaults to 0.8.

        """
        # Split to train and validation
        train_dset, validation_dset = random_split(train_dset, [train_ratio, 1-train_ratio])

        # Produce DataLoaders
        self.train_dloader = DataLoader(
            train_dset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,  
            pin_memory=True,
            collate_fn=pad_collate_fn
        )

        self.validation_dloader = DataLoader(
            validation_dset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=pad_collate_fn
        )

        return