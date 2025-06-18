from torch_geometric.nn import TransformerConv
from torch import nn
import torch


class GNN_TransformerConv(nn.Module):
    """
    Graph Neural Network using [TransformerConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.TransformerConv.html) layers from PyTorch Geometric.

    Args:
        input_dim (int): Dimensionality of input node features.
        hidden_dim (int): Hidden dimension size for TransformerConv layers.
        output_dim (int): Output dimension size.
        edge_dim (int): Dimensionality of edge features.
        num_layers (int): Number of TransformerConv layers.
        heads (int, optional): Number of attention heads.
        mask_dim (int, optional): Dimension of mask vector.
        mask_value (float, optional): Initial mask value.
        learn_mask (bool, optional): Whether mask values are learnable.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        edge_dim: int,
        num_layers: int,
        heads: int = 1,
        mask_dim: int = 6,
        mask_value: float = -1.0,
        learn_mask: bool = False,
    ):
        super(GNN_TransformerConv, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.heads = heads
        self.mask_dim = mask_dim
        self.mask_value = mask_value
        self.learn_mask = learn_mask

        self.layers = nn.ModuleList()
        current_dim = input_dim  # First layer takes `input_dim` as input

        for _ in range(self.num_layers):
            self.layers.append(
                TransformerConv(
                    current_dim,
                    self.hidden_dim,
                    heads=self.heads,
                    edge_dim=self.edge_dim,
                    beta=False,
                ),
            )
            # Update the dimension for the next layer
            current_dim = self.hidden_dim * self.heads

        # Fully connected (MLP) layers after the GAT layers
        self.mlps = nn.Sequential(
            nn.Linear(self.hidden_dim * self.heads, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, output_dim),
        )

        if learn_mask:
            self.mask_value = nn.Parameter(
                torch.randn(mask_dim) + mask_value,
                requires_grad=True,
            )
        else:
            self.mask_value = nn.Parameter(
                torch.zeros(mask_dim) + mask_value,
                requires_grad=False,
            )

    def forward(self, x, pe, edge_index, edge_attr, batch):
        """
        Forward pass for the GPSTransformer.

        Args:
            x (Tensor): Input node features of shape [num_nodes, input_dim].
            pe (Tensor): Positional encoding of shape [num_nodes, pe_dim] (not used).
            edge_index (Tensor): Edge indices for graph convolution.
            edge_attr (Tensor): Edge feature tensor.
            batch (Tensor): Batch vector assigning nodes to graphs (not used).

        Returns:
            output (Tensor): Output node features of shape [num_nodes, output_dim].
        """
        for conv in self.layers:
            x = conv(x, edge_index, edge_attr)
            x = nn.LeakyReLU()(x)

        x = self.mlps(x)
        return x
