from dataclasses import dataclass

from deepaudiox.config.base_config import LossConfig
from deepaudiox.config.loss_config_registry import register_loss_config


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

