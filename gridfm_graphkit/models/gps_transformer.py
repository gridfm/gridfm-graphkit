from gridfm_graphkit.io.registries import MODELS_REGISTRY
from torch_geometric.nn import GPSConv, GINEConv
from torch import nn
import torch


@MODELS_REGISTRY.register("GPSTransformer")
class GPSTransformer(nn.Module):
    """
    A GPS (Graph Transformer) model based on [GPSConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GPSConv.html) and [GINEConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html) layers from Pytorch Geometric.

    This model encodes node features and positional encodings separately,
    then applies multiple graph convolution layers with batch normalization,
    and finally decodes to the output dimension.

    Args:
        args (NestedNamespace): Parameters

    Attributes:
        input_dim (int): Dimension of input node features. From ``args.model.input_dim``.
        hidden_dim (int): Hidden dimension size for all layers. From ``args.model.hidden_size``.
        output_dim (int): Dimension of the output node features. From ``args.model.output_dim``.
        edge_dim (int): Dimension of edge features. From ``args.model.edge_dim``.
        pe_dim (int): Dimension of the positional encoding. Must be less than ``hidden_dim``. From ``args.model.pe_dim``.
        num_layers (int): Number of GPSConv layers. From ``args.model.num_layers``.
        heads (int, optional): Number of attention heads in GPSConv. From ``args.model.attention_head``. Defaults to 1.
        dropout (float, optional): Dropout rate in GPSConv. From ``args.model.dropout``. Defaults to 0.0.
        mask_dim (int, optional): Dimension of the mask vector. From ``args.data.mask_dim``. Defaults to 6.
        mask_value (float, optional): Initial value for learnable mask parameters. From ``args.data.mask_value``. Defaults to -1.0.
        learn_mask (bool, optional): Whether to learn mask values as parameters. From ``args.data.learn_mask``. Defaults to True.

    Raises:
        ValueError: If `pe_dim` is not less than `hidden_dim`.
    """

    def __init__(self, args):
        super().__init__()

        # === Required (no defaults in original) ===
        self.input_dim = args.model.input_dim
        self.hidden_dim = args.model.hidden_size
        self.output_dim = args.model.output_dim
        self.edge_dim = args.model.edge_dim
        self.pe_dim = args.model.pe_dim
        self.num_layers = args.model.num_layers

        # === Optional (defaults in original) ===
        self.heads = getattr(args.model, "attention_head", 1)
        self.dropout = getattr(args.model, "dropout", 0.0)
        self.mask_dim = getattr(args.data, "mask_dim", 6)
        self.mask_value = getattr(args.data, "mask_value", -1.0)
        self.learn_mask = getattr(args.data, "learn_mask", True)

        if not self.pe_dim < self.hidden_dim:
            raise ValueError(
                "positional encoding dimension must be smaller than model hidden dimension",
            )

        self.layers = nn.ModuleList()

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim - self.pe_dim),
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
            nn.Linear(self.hidden_dim, self.output_dim),
        )

        if self.learn_mask:
            self.mask_value = nn.Parameter(
                torch.randn(self.mask_dim) + self.mask_value,
                requires_grad=True,
            )
        else:
            self.mask_value = nn.Parameter(
                torch.zeros(self.mask_dim) + self.mask_value,
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
