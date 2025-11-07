from pathlib import Path

import torch

from src.deepaudiox.callbacks.base_callback import BaseCallback
from src.deepaudiox.utils.training_utils import get_logger

GREEN = "\033[92m"
ENDC = "\033[0m"


class Checkpointer(BaseCallback):
    """Training callback for saving model checkpoints.

    Keep track of validation loss and
    stores a model checkpoint when it drops.

    Attributes:
        path_to_checkpoint (str): The path to the saved checpoint.
        logger (): A module for logging messages.

    """

    def __init__(self, path_to_checkpoint: str, logger: object = None):
        """Initialize the callback.

        Args:
            path_to_checkpoint (str): The path to the saved checpoint.
            logger (): A module for logging messages. Defaults to None.

        """
        self.path_to_checkpoint = Path(path_to_checkpoint)
        self.logger = logger or get_logger()

    def on_epoch_end(self, trainer):
        """When epoch ends, check validation loss and produce checkpoint.

        Args:
            trainer (trainer.Trainer): The training module of the SDK.
        """
        latest_validation_loss = trainer.state.validation_loss[-1]

        if trainer.state.lowest_loss > latest_validation_loss:
            decrease_percentage = (trainer.state.lowest_loss - latest_validation_loss) / trainer.state.lowest_loss * 100

            self.logger.info(
                f"[CHECKPOINTER] Validation loss decreased: "
                f"({trainer.state.lowest_loss:.6f} --> {latest_validation_loss:.6f}), "
                f"{GREEN}(-{decrease_percentage:.2f}%){ENDC}."
            )

            trainer.state.lowest_loss = latest_validation_loss

            try:
                self.path_to_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(trainer.model.state_dict(), self.path_to_checkpoint)
                self.logger.info(f"[CHECKPOINTER] Checkpoint saved successfully at: {self.path_to_checkpoint}")
            except PermissionError:
                self.logger.info(f"[CHECKPOINTER] Permission denied: cannot write to {self.path_to_checkpoint}")
            except FileNotFoundError:
                self.logger.info(f"[CHECKPOINTER] Directory not found: {self.path_to_checkpoint}")
            except OSError as e:
                self.logger.info(f"[CHECKPOINTER] OS error while saving checkpoint: {e}")
            except Exception as e:
                self.logger.info(f"[CHECKPOINTER] Unexpected error while saving checkpoint: {type(e).__name__}: {e}")

            return
