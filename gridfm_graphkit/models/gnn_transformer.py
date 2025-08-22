from gridfm_graphkit.io.registries import MODELS_REGISTRY
from torch_geometric.nn import TransformerConv
from torch import nn
import torch


@MODELS_REGISTRY.register("GNN_TransformerConv")
class GNN_TransformerConv(nn.Module):
    """
    Graph Neural Network using [TransformerConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.TransformerConv.html) layers from PyTorch Geometric.

    Args:
        args (NestedNamespace): Parameters

    Attributes:
        input_dim (int): Dimensionality of input node features. From ``args.model.input_dim``.
        hidden_dim (int): Hidden dimension size for TransformerConv layers. From ``args.model.hidden_dim``.
        output_dim (int): Output dimension size. From ``args.model.output_dim``.
        edge_dim (int): Dimensionality of edge features. From ``args.model.edge_dim``.
        num_layers (int): Number of TransformerConv layers. From ``args.model.num_layers``.
        heads (int, optional): Number of attention heads. From ``args.model.heads``. Defaults to 1.
        mask_dim (int, optional): Dimension of mask vector. From ``args.data.mask_dim``. Defaults to 6.
        mask_value (float, optional): Initial mask value. From ``args.data.mask_value``. Defaults to -1.0.
        learn_mask (bool, optional): Whether mask values are learnable. From ``args.data.learn_mask``. Defaults to True.
    """

    def __init__(self, args):
        super().__init__()

        # === Required (no defaults originally) ===
        self.input_dim = args.model.input_dim
        self.hidden_dim = args.model.hidden_size
        self.output_dim = args.model.output_dim
        self.edge_dim = args.model.edge_dim
        self.num_layers = args.model.num_layers

        # === Optional (had defaults originally) ===
        self.heads = getattr(args.model, "attention_head", 1)
        self.mask_dim = getattr(args.data, "mask_dim", 6)
        self.mask_value = getattr(args.data, "mask_value", -1.0)
        self.learn_mask = getattr(args.data, "learn_mask", False)

        self.layers = nn.ModuleList()
        current_dim = self.input_dim  # First layer takes `input_dim` as input

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
