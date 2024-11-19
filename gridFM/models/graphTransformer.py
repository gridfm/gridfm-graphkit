from torch_geometric.nn import TransformerConv
from torch import nn


class GraphTransformer(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, edge_dim, num_layers, heads=1
    ):
        super(GraphTransformer, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.heads = heads

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
                )
            )
            # Update the dimension for the next layer
            current_dim = self.hidden_dim * self.heads

        # Fully connected (MLP) layers after the GAT layers
        self.mlps = nn.Sequential(
            nn.Linear(self.hidden_dim * self.heads, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        for conv in self.layers:
            x = conv(x, edge_index, edge_attr)
            x = nn.LeakyReLU()(x)

        x = self.mlps(x)
        return x
