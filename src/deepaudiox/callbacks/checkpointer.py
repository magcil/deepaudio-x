from pathlib import Path

import torch

from src.deepaudiox.callbacks.base_callback import BaseCallback
from src.deepaudiox.utils.training_utils import get_logger

GREEN = '\033[92m'
ENDC = '\033[0m'

class Checkpointer(BaseCallback):
    """Training callback for saving model checkpoints.
    
    Keep track of validation loss and 
    stores a model checkpoint when it drops.

    Attributes:
        output_dir (str): The directory to store training output.
        logger (): A module for logging messages.

    """

    def __init__(self, output_dir: str, logger: object = None):
        """Initialize the callback.

        Args:
            output_dir (str): The directory to store training output.
            logger (): A module for logging messages. Defaults to None.

        """
        self.output_dir = Path(output_dir)
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

            checkpoint_path = self.output_dir / "checkpoint.pt"
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(trainer.model.state_dict(), checkpoint_path)
                self.logger.info(f"[CHECKPOINTER] Checkpoint saved successfully at: {checkpoint_path}")
            except PermissionError:
                self.logger.info(f"[CHECKPOINTER] Permission denied: cannot write to {checkpoint_path}")
            except FileNotFoundError:
                self.logger.info(f"[CHECKPOINTER] Directory not found: {self.output_dir}")
            except OSError as e:
                self.logger.info(f"[CHECKPOINTER] OS error while saving checkpoint: {e}")
            except Exception as e:
                self.logger.info(f"[CHECKPOINTER] Unexpected error while saving checkpoint: {type(e).__name__}: {e}")

            return