from dataclasses import dataclass

from deepaudiox.config.data_config import DataConfig
from deepaudiox.config.loss_config import LossConfig
from deepaudiox.config.model_config import ModelConfig
from deepaudiox.config.optimization_config import OptimizationConfig
from deepaudiox.config.scheduling_config import SchedulingConfig


@dataclass
class TrainingConfig:
    """Configuration for setting up the training process.

    Attributes:
        data_config (DataConfig): The configuration parameters for loading data.
        loss_config (LossConfig): The configuration parameters of the loss function.
        optimization_config (OptimizationConfig): The configuration parameters of the optimizer.
        scheduling_config (SchedulingConfig): The configuration parameters of the scheduler.
        model_config (ModelConfig): The configuration parameters of the model.
        output_dir (str): The path to the training output folder.
        epochs (int): The maximum number of training epochs. Defaults to 10.
        patience (int): The maximum number of training epochs with no reduction in loss. Defaults to 5.

    """
    data_config: DataConfig
    loss_config: LossConfig
    optimization_config: OptimizationConfig
    scheduling_config: SchedulingConfig
    model_config: ModelConfig
    output_dir: str
    epochs: int = 10
    patience: int = 5
