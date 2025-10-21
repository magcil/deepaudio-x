from dataclasses import dataclass
from .data_config import DataConfig
from .optimization_config import OptimizationConfig
from .scheduling_config import SchedulingConfig
from .model_config import ModelConfig
from .loss_config import LossConfig


@dataclass
class TrainingConfig:
    """Configuration for setting up the training process.

    Attributes:
        optimization_config (OptimizationConfig): The configuration parameters of the optimizer
        scheduling_config (SchedulingConfig): The configuration parameters of the scheduler
        data_config (DataConfig): The configuration parameters for loading data
        model_config (ModelConfig): The configuration parameters of the model
        loss_config (LossConfig): The configuration parameters of the loss function

    """
    loss_config: LossConfig
    optimization_config: OptimizationConfig
    scheduling_config: SchedulingConfig
    data_config: DataConfig
    model_config: ModelConfig
    epochs: int = 5
