import torch
import torch.nn as nn
import torch.nn.functional as F

from deepaudiox.modules.projection.base_projection import BaseProjection


class DivEncLayer(BaseProjection):
    """Divide Encoder Layer for dimensionality reduction.

    This layer implements divide encoder layer as presented in `Simultaneous Feature Learning and Hash Coding with
    Deep Neural Networks` (arxiv.org/pdf/1504.03410). In brief, the input vector is subdivided into out_dim subvectors
    of dimension in_dim // out_dim. Each subvector is passed through a series of linear-elu layers specified by
    linear_layers mapped to a single value forming the final out_dim L2 norm-projected vector.

    """

    def __init__(self, in_dim: int, out_dim: int, linear_layers: list[int] | None = None):
        """Initialized a divide encoder layer.

        Args:
            in_dim (int): Input embedding dimension.
            out_dim (int): Output dimension after projection.
            linear_layers (list): List containing the dimensions of the linear-elu layers.

        Example:
            >>> from deepaudiox.modules.projection.projections import DivEncLayer
            >>> projection = DivEncLayer(in_dim=768, out_dim=128) # 768 is Beats embedding
        """
        if linear_layers is None:
            linear_layers = [32, 1]
        super().__init__(in_dim=in_dim, out_dim=out_dim)
        assert in_dim % out_dim == 0, "in_dim must be divisible by out_dim"
        self.split_fc_layers: nn.ModuleList = nn.ModuleList()
        self.linear_layers: list[int] = linear_layers
        self.v: int = int(in_dim / out_dim)
        self._construct_layers()

    def _construct_layers(self) -> None:
        for _i in range(self.out_dim):
            seq = nn.Sequential()
            seq.append(nn.Linear(self.v, self.linear_layers[0]))
            seq.append(nn.ELU())
            seq.append(nn.LayerNorm(self.linear_layers[0]))
            seq.append(nn.Linear(self.linear_layers[0], self.linear_layers[1]))
            self.split_fc_layers.append(seq)

    def _split_encoding(self, x_slices: torch.Tensor) -> torch.Tensor:
        out: list[torch.Tensor] = []
        for i in range(self.out_dim):
            out.append(self.split_fc_layers[i](x_slices[:, i, :]))
        return torch.concat(out, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.reshape(x, (x.shape[0], self.out_dim, -1))
        return F.normalize(self._split_encoding(x), p=2.0)
