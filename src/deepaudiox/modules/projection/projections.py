import torch
import torch.nn as nn
import torch.nn.functional as F

from deepaudiox.modules.projection.base_projection import BaseProjection


class DivEncLayer(BaseProjection):
    # Divided Encoder Layer for dimensionality reduction
    def __init__(self, in_dim: int, out_dim: int, unit_dim: list[int] | None = None) -> None:
        """Divided Encoder Layer for dimensionality reduction.
        Args:
            in_dim (int): Input dimension (i.e. number of splits).
            out_dim (int): Output dimension after projection.
            unit_dim (list): List containing the dimensions of the two linear layers.
        """
        if unit_dim is None:
            unit_dim = [32, 1]
        super().__init__()
        assert in_dim % out_dim == 0, "out_dim must be divisible by in_dim"
        self.split_fc_layers: nn.ModuleList = nn.ModuleList()
        self.out_dim = out_dim
        self.in_dim: int = in_dim
        self.unit_dim: list[int] = unit_dim
        self.v: int = int(in_dim / out_dim)
        self._construct_layers()

    def _construct_layers(self) -> None:
        for _i in range(self.out_dim):
            seq = nn.Sequential()
            seq.append(nn.Linear(self.v, self.unit_dim[0]))
            seq.append(nn.ELU())
            seq.append(nn.LayerNorm(self.unit_dim[0]))
            seq.append(nn.Linear(self.unit_dim[0], self.unit_dim[1]))
            self.split_fc_layers.append(seq)

    def _split_encoding(self, x_slices: torch.Tensor) -> torch.Tensor:
        out: list[torch.Tensor] = []
        for i in range(self.out_dim):
            out.append(self.split_fc_layers[i](x_slices[:, i, :]))
        return torch.concat(out, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.reshape(x, (x.shape[0], self.out_dim, -1))
        return F.normalize(self._split_encoding(x), p=2.0)
