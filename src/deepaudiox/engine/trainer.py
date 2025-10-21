from src.deepaudiox.config.training_config import TrainingConfig
from src.deepaudiox.datasets.dataset_manager import DatasetManager
from src.deepaudiox.optimizers.optimizer_registry import build_optimizer
from src.deepaudiox.schedulers.scheduler_registry import build_scheduler
from src.deepaudiox.loss_functions.loss_registry import build_loss_function
import torchvision.models as models
from src.deepaudiox.callbacks.console_logger import ConsoleLogger
from src.deepaudiox.utils.training_utils import get_device
from tqdm import tqdm
from dataclasses import dataclass

import torch

from src.deepaudiox.models.wav2vec_classifier import Wav2VecClassifier

@dataclass
class State:
    current_epoch = 1
    train_loss = []
    validation_loss = []
    step = 0

class Trainer:
    def __init__(
        self,
        config: TrainingConfig
    ):  
        # Configure training state
        self.state = State()

        self.epochs = config.epochs

        self.device = get_device()

        # Load datasets
        self.dataset_manager = DatasetManager(config.data_config)

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

        # Configure loss function
        self.loss_function = build_loss_function(config=config.loss_config)

        # Configure callbacks
        self.callbacks = [
            ConsoleLogger(log_interval=20)
        ]

    def train(self):
        for cb in self.callbacks: cb.on_train_start(self)

        for epoch in range(1, self.epochs + 1):
            self.state.current_epoch = epoch
            for cb in self.callbacks: cb.on_epoch_start(self)

            train_loss, val_loss = 0.0, 0.0

            self.model.train()

            train_dataloader = self.dataset_manager.get_dataloader("train")
            with tqdm(train_dataloader, unit="batch", leave=False, desc=f"Training phase") as tbatch:
                for i, (features, lengths, labels) in enumerate(tbatch, 1):
                    self.state.step += 1
                    # features = features.to(self.device)
                    # y_true = labels.to(self.device)
                    # y_pred = self.model(features, lengths) 

                    # self.optimizer.zero_grad()
                    # batch_loss = self.loss_function(y_pred, y_true)                    
                    # batch_loss.backward()
                    # train_loss += batch_loss.item()
                    # self.optimizer.step()

                    for cb in self.callbacks: cb.on_step_end(self)

                train_loss /= len(train_dataloader)
                # self.scheduler.step()

            self.model.eval()
            validation_dataloader = self.dataset_manager.get_dataloader("validation")
            with torch.no_grad():
                with tqdm(validation_dataloader, unit="batch", leave=False, desc=f"Validation phase") as vbatch:
                    for i, (features, lengths, labels) in enumerate(vbatch, 1):
                        self.state.step += 1
                        # features = features.to(self.device)
                        # y_true = labels.to(self.device)
                        # y_pred = self.model(features, lengths) 

                        # batch_loss = self.loss_function(y_pred, y_true)
                        # val_loss += batch_loss.item()

                        for cb in self.callbacks: cb.on_step_end(self)
                val_loss /= len(validation_dataloader)
            for cb in self.callbacks: cb.on_validation_end(self)
            for cb in self.callbacks: cb.on_epoch_end(self)
        for cb in self.callbacks: cb.on_train_end(self)