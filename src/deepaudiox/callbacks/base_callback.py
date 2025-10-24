from abc import ABC, abstractmethod


class BaseCallback(ABC):
    """Base class for all training callbacks.
    
    Includes methods that are called throught the training process
    and are responsible for saving checkpoints, logging messages,
    handling early stopping, and reporting training results.

    """
    @abstractmethod
    def on_train_start(self, trainer): pass
    @abstractmethod
    def on_train_end(self, trainer): pass
    @abstractmethod
    def on_epoch_start(self, trainer): pass
    @abstractmethod
    def on_epoch_end(self, trainer): pass
    @abstractmethod
    def on_step_end(self, trainer): pass