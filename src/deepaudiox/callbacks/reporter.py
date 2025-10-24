from src.deepaudiox.callbacks.base_callback import BaseCallback
from src.deepaudiox.utils.training_utils import get_logger
import pandas as pd
from pathlib import Path

class Reporter(BaseCallback):
    """Training callback for reporting training results.
    
    Summarize training results and produce reports 
    (e.g. classification reports, charts, CSV files)

    Attributes:
        logger (): A module for logging messages. Defaults to None.
        output_dir (str): The directory to store training output.

    """

    def __init__(self, output_dir: str, logger: object=None):
        """Initialize the callback.

        Args:
            logger (): A module for logging messages. Defaults to None.
            output_dir (str): The directory to store training output.

        """
        self.logger = logger or get_logger()
        self.output_dir = Path(output_dir)
    
    def on_train_end(self, trainer):
        """When train ends, save CSV file with train and validation losses.
        (usefull for monitoring overfitiing and producing charts)
        
        Args:
            trainer (trainer.Trainer): The training module of the SDK.

        """
        losses_df = pd.DataFrame({
            "epoch": list(range(1, len(trainer.state.train_loss)+1)),
            "train_loss": trainer.state.train_loss,
            "val_loss": trainer.state.validation_loss
        })

        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_dir / "loss_history.csv"
        try:
            losses_df.to_csv(csv_path, index=False)
            self.logger.info(f"[REPORTER] Loss history saved successfully at: {csv_path}")
        except PermissionError:
            self.logger.info(f"[REPORTER] Permission denied: cannot write to {csv_path}")
        except FileNotFoundError:
            self.logger.info(f"[REPORTER] Directory not found: {self.output_dir}")
        except OSError as e:
            self.logger.info(f"[REPORTER] OS error while saving loss history: {e}")
        except Exception as e:
            self.logger.info(f"[REPORTER] Unexpected error while saving loss history: {type(e).__name__}: {e}")

        return

