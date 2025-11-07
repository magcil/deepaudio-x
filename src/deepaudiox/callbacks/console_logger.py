import time

from src.deepaudiox.callbacks.base_callback import BaseCallback
from src.deepaudiox.utils.training_utils import get_logger


class ConsoleLogger(BaseCallback):
    """Training callback for logging messages in the console.

    Log messages in the console throughout the training process.

    Attributes:
        logger (): A module for logging messages.

    """

    def __init__(self, logger: object = None):
        """Initialize the callback.

        Args:
            logger (): A module for logging messages. Defaults to None.
            last_time (float): Keeps track of epoch duration.

        """
        self.last_time = time.time()
        self.logger = logger or get_logger()

    def on_epoch_start(self, trainer):
        """When epoch starts, log indicative message.

        Args:
            trainer (trainer.Trainer): The training module of the SDK.
        """
        self.logger.info(f"[Epoch {trainer.state.current_epoch}/{trainer.epochs}]")

    def on_epoch_end(self, trainer):
        """When epoch ends, log epoch duration and recorded scores.

        Args:
            trainer (trainer.Trainer): The training module of the SDK.

        """
        elapsed_time = time.time() - self.last_time
        train_loss = trainer.state.train_loss[-1]
        validation_loss = trainer.state.validation_loss[-1]

        self.logger.info(
            f"Epoch {trainer.state.current_epoch} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val. Loss: {validation_loss:.4f} | "
            f"Time: {elapsed_time:.2f}s"
        )

        self.last_time = time.time()

        return

    def on_train_end(self, trainer):
        """When train ends, show indicative message.

        Args:
            trainer (trainer.Trainer): The training module of the SDK.

        """
        self.logger.info("Training has finished.")

        return
