import torch.nn as nn

class AudioClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_layers=None, activation="relu", batch_norm=False):
        """Audio classification head for downstream tasks.

        Attributes:
            in_dim (int): Input feature dimension.
            num_classes (int): Number of output classes.
            hidden_layers (list[int] or None): List of hidden layer sizes. If None, just a single linear layer.
            activation (str): Activation function name ("relu", "gelu", "tanh" or "leakyrelu").
            batch_norm (bool): Whether to use BatchNorm1d after each Linear layer.

        """
        super().__init__()
        
        if hidden_layers is None or len(hidden_layers) == 0:
            hidden_layers = []
        
        layers = []
        input_dim = in_dim
        
        activation_fn = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "leakyrelu": nn.LeakyReLU()
        }.get(activation.lower(), nn.ReLU())
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation_fn)
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, num_classes))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
