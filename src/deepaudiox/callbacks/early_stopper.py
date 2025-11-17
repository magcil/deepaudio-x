import logging

from deepaudiox.callbacks.base_callback import BaseCallback
from deepaudiox.utils.training_utils import get_logger


class EarlyStopper(BaseCallback):
    """Training callback for handling early stopping.

    Keep track of elapsed epochs with no decrease in loss
    and terminates the training process when patience is exceeded.

    Attributes:
        logger (logging.Logger): A module for logging messages. Defaults to None.
        patience (int): The maximum number of epochs with no decrease in loss
        elapsed_epochs (int): The number of concurrent epochs with no decrease in loss.

    """

    def __init__(self, patience: int = 5, logger: logging.Logger | None = None):
        """Initialize the callback.

        Args:
            logger (): A module for logging messages. Defaults to None.
            patience (int): The maximum number of epochs with no decrease in loss. Defaults to 5.

        """

        self.logger = logger or get_logger()
        self.patience = patience
        self.elapsed_epochs = 0

    def on_epoch_end(self, trainer):
        """When epoch ends, keep track of elapsed epochs with no reduction in loss.

        Args:
            trainer (trainer.Trainer): The training module of the SDK.

        """

        # Assess if loss was reduced
        latest_validation_loss = trainer.state.validation_loss[-1]
        if trainer.state.lowest_loss < latest_validation_loss:
            self.elapsed_epochs += 1
        else:
            self.elapsed_epochs = 0

        # Log warning message
        if self.elapsed_epochs >= int(0.8 * self.patience):
            self.logger.info(f"[EARLY STOPPING] Elapsed epochs: {self.elapsed_epochs} out of {self.patience}")

        # Terminate training
        if self.elapsed_epochs == self.patience:
            trainer.state.early_stop = True
            self.logger.info("[EARLY STOPPING] Patience exceeded, early stoping ...")

        return
