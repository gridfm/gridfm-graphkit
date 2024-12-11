from torch_geometric.nn import GATConv
from torch import nn
import torch


class GAT(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        edge_dim,
        num_layers,
        heads=1,
        mask_dim=6,
        mask_value=-1,
        learn_mask=True,
    ):
        super(GAT, self).__init__()
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
                GATConv(
                    current_dim,
                    self.hidden_dim,
                    heads=self.heads,
                    edge_dim=self.edge_dim,
                )
            )
            # Update the dimension for the next layer
            current_dim = self.hidden_dim * self.heads

        # Fully connected (MLP) layers after the GAT layer
        self.mlps = nn.Sequential(
            nn.Linear(self.hidden_dim * self.heads, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, output_dim),
        )

        if learn_mask:
            self.mask_value = nn.Parameter(
                torch.randn(mask_dim) + mask_value, requires_grad=True
            )
        else:
            self.mask_value = nn.Parameter(
                torch.zeros(mask_dim) + mask_value, requires_grad=False
            )


    def forward(self, x, edge_index, edge_attr):
        for conv in self.layers:
            x = conv(x, edge_index, edge_attr)
            x = nn.LeakyReLU()(x)

        x = self.mlps(x)
        return x
