import torch
from dataclasses import dataclass
from src.deepaudiox.config.loss_config_registry import register_loss_config

@dataclass
class LossConfig:
    """ Configuration for setting up loss functions.
        Every loss function config class inherits LossConfig.

    Attributes:
        name (str): The name used by the registry to build the configuration 
                    of a loss function. Defaults to CrossEntropyLoss.
    
    """
    name: str = "CrossEntropyLoss"

@dataclass 
@register_loss_config("CrossEntropyLoss")
class CrossEntropyLossConfig(LossConfig):
    """ Configuration for setting up a Cross Entropy Loss module.
        https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    
    Attributes:
        reduction (str): Specifies the reduction to apply to the output. Defaults to mean.
        label_smoothing (float): Specifies the amount of smoothing when computing the loss. Defaults to 0.0.

    """

    reduction: str = 'mean'
    label_smoothing: float = 0.0

