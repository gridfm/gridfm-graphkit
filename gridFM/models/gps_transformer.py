from torch_geometric.nn import GPSConv, GINEConv
from torch import nn
import torch


class GPSTransformer(nn.Module):
    """
    A GPS (Graph Transformer) model based on [GPSConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GPSConv.html) and [GINEConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html) layers from Pytorch Geometric.

    This model encodes node features and positional encodings separately,
    then applies multiple graph convolution layers with batch normalization,
    and finally decodes to the output dimension.

    Args:
        input_dim (int): Dimension of input node features.
        hidden_dim (int): Hidden dimension size for all layers.
        output_dim (int): Dimension of the output node features.
        edge_dim (int): Dimension of edge features.
        pe_dim (int): Dimension of the positional encoding.
            Must be less than hidden_dim.
        num_layers (int): Number of GPSConv layers.
        heads (int, optional): Number of attention heads in GPSConv.
        dropout (float, optional): Dropout rate in GPSConv.
        mask_dim (int, optional): Dimension of the mask vector.
        mask_value (float, optional): Initial value for learnable mask parameters.
        learn_mask (bool, optional): Whether to learn mask values as parameters.

    Raises:
        ValueError: If `pe_dim` is not less than `hidden_dim`.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        edge_dim: int,
        pe_dim: int,
        num_layers: int,
        heads: int = 1,
        dropout: float = 0.0,
        mask_dim: int = 6,
        mask_value: float = -1.0,
        learn_mask: bool = True,
    ):
        super(GPSTransformer, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.pe_dim = pe_dim
        self.heads = heads
        self.dropout = dropout
        self.mask_dim = mask_dim
        self.mask_value = mask_value
        self.learn_mask = learn_mask

        if not pe_dim < hidden_dim:
            raise ValueError(
                "positional encoding dimension must be smaller than model hidden dimension",
            )

        self.layers = nn.ModuleList()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim - self.pe_dim),
            nn.LeakyReLU(),
        )
        self.input_norm = nn.BatchNorm1d(self.hidden_dim - self.pe_dim)
        self.pe_norm = nn.BatchNorm1d(self.pe_dim)

        for _ in range(self.num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
                nn.LeakyReLU(),
            )
            self.layers.append(
                nn.ModuleDict(
                    {
                        "conv": GPSConv(
                            channels=self.hidden_dim,
                            conv=GINEConv(nn=mlp, edge_dim=self.edge_dim),
                            heads=self.heads,
                            dropout=self.dropout,
                        ),
                        "norm": nn.BatchNorm1d(
                            self.hidden_dim,
                        ),  # BatchNorm after each graph layer
                    },
                ),
            )

        self.pre_decoder_norm = nn.BatchNorm1d(self.hidden_dim)
        # Fully connected (MLP) layers after the GAT layers
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
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
            pe (Tensor): Positional encoding of shape [num_nodes, pe_dim].
            edge_index (Tensor): Edge indices for graph convolution.
            edge_attr (Tensor): Edge feature tensor.
            batch (Tensor): Batch vector assigning nodes to graphs.

        Returns:
            output (Tensor): Output node features of shape [num_nodes, output_dim].
        """
        x_pe = self.pe_norm(pe)

        x = self.encoder(x)
        x = self.input_norm(x)

        x = torch.cat((x, x_pe), 1)
        for layer in self.layers:
            x = layer["conv"](
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch,
            )
            x = layer["norm"](x)

        x = self.pre_decoder_norm(x)
        x = self.decoder(x)

        return x
