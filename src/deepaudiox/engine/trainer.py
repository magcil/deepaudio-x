from src.deepaudiox.config.training_config import TrainingConfig
from src.deepaudiox.config.data_config import DataConfig
from src.deepaudiox.optimizers.optimizer_registry import build_optimizer
from src.deepaudiox.schedulers.scheduler_registry import build_scheduler
from src.deepaudiox.loss_functions.loss_registry import build_loss_function
from src.deepaudiox.callbacks.console_logger import ConsoleLogger
from src.deepaudiox.callbacks.checkpointer import Checkpointer
from src.deepaudiox.callbacks.early_stopper import EarlyStopper
from src.deepaudiox.callbacks.reporter import Reporter
from src.deepaudiox.utils.training_utils import get_device
from src.deepaudiox.datasets.audio_classification_dataset import AudioClassificationDataset
from src.deepaudiox.utils.training_utils import get_class_mapping, pad_collate_fn
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from dataclasses import dataclass
import torch
import logging
import numpy as np

from src.deepaudiox.models.wav2vec_classifier import Wav2VecClassifier

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
    lowest_loss = np.Inf
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
        optimizer (object): The optimizer of the training process.
        scheduler (object): The scheduler of the training process.
        loss_function (object): The loss function used for optimization.
        callbacks (list): A list of callbacks used throught the training lifecycle.

    """
    def __init__(self, config: TrainingConfig):  
        """Initialize the Trainer.

        Args:
            config (TrainingConfig): Configuration for the training process.

        """
        # Configure training state
        self.state = State()
        self.epochs = config.epochs
        self.device = get_device()

        # Configure logger
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        self.logger = logging.getLogger("ConsoleLogger")

        # Load datasets
        self.train_dloader = None
        self.validation_dloader = None
        self._setup_dataloaders(config.data_config)

        # Load mock model
        self.model = Wav2VecClassifier()

        # Configure optimizer
        self.optimizer = build_optimizer(
            model_params=self.model.parameters(),
            config=config.optimization_config
        )

        # Configure scheduler
        self.scheduler = build_scheduler(
            optimizer=self.optimizer,
            config=config.scheduling_config
        )

        print(self.scheduler.T_max)

        # Configure loss function
        self.loss_function = build_loss_function(config=config.loss_config)

        # Configure callbacks
        self.callbacks = [
            ConsoleLogger(logger=self.logger),
            Checkpointer(output_dir=config.output_dir, logger=self.logger),
            EarlyStopper(patience=config.patience, logger=self.logger),
            Reporter(output_dir=config.output_dir)
        ]

    def train(self):
        """Perform the training process"""

        # Execute callbacks in the beginning of training
        for cb in self.callbacks: cb.on_train_start(self)

        # Initiate training
        for epoch in range(1, self.epochs + 1):
            self.state.current_epoch = epoch
            train_loss, val_loss = 0.0, 0.0

            if not self.state.early_stop:
                # Execute callbacks in the beginning of the epoch
                for cb in self.callbacks: cb.on_epoch_start(self)

                # Execute training loop
                self.model.train()
                with tqdm(self.train_dloader, unit="batch", leave=False, desc=f"Training phase") as tbatch:
                    # Optimize the model by batch
                    for i, item in enumerate(tbatch, 1):
                        self.optimizer.zero_grad()
                        features = item['feature'].to(self.device)
                        y_pred = self.model(features) 
                        y_true = item['class_id'].to(self.device)
                        batch_loss = self.loss_function(y_pred, y_true)                    
                        batch_loss.backward()
                        train_loss += batch_loss.item()
                        self.optimizer.step()

                    train_loss /= len(self.train_dloader)
                    self.scheduler.step()

                # Execute validation loop
                self.model.eval()
                with torch.no_grad():
                    with tqdm(self.validation_dloader, unit="batch", leave=False, desc=f"Validation phase") as vbatch:
                        # Compute validation loss by batch
                        for i, item in enumerate(vbatch, 1):
                            features = item['feature'].to(self.device)
                            y_true = item['class_id'].to(self.device)
                            y_pred = self.model(features) 
                            batch_loss = self.loss_function(y_pred, y_true)
                            val_loss += batch_loss.item()
                            
                    val_loss /= len(self.validation_dloader)

                # Update training state
                self.state.train_loss.append(train_loss)
                self.state.validation_loss.append(val_loss)
                if self.state.lowest_loss > val_loss:
                    self.state.lowest_loss = val_loss

                # Execute callbacks at the end of the epoch
                for cb in self.callbacks: cb.on_epoch_end(self)
        # Execute callbacks at the end of training
        for cb in self.callbacks: cb.on_train_end(self)

        return

    def _setup_dataloaders(self, config: DataConfig):
        """Generate PyTorch DataLoaders for training and validation splits.
        
        Arguments:
            config (DataConfig): Configuration required for loading the training data.

        """
        # Load data from given directory
        class_mapping = get_class_mapping(config.train_dir)
        dataset = AudioClassificationDataset(
            root_dir=config.train_dir,
            sample_rate=config.sample_rate,
            class_mapping=class_mapping
        )

        # Split to train and validation
        train_ratio = 1 - config.validation_ratio
        train_dset, validation_dset = random_split(dataset, [train_ratio, config.validation_ratio])

        # Produce DataLoaders
        self.train_dloader = DataLoader(
            train_dset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,  
            pin_memory=True,
            collate_fn=pad_collate_fn
        )

        self.validation_dloader = DataLoader(
            validation_dset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=pad_collate_fn
        )

        return