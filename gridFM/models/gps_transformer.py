from torch_geometric.nn import GPSConv, GINEConv
from torch import nn
import torch


class GPSTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        edge_dim,
        pe_dim,
        num_layers,
        heads=1,
        dropout=0.0,
        mask_dim=6,
        mask_value=-1,
        learn_mask=True,
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

        assert (
            pe_dim < hidden_dim
        ), "positional encoding dimension must be smaller than model hidden dimension"

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
                            self.hidden_dim
                        ),  # BatchNorm after each graph layer
                    }
                )
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
                torch.randn(mask_dim) + mask_value, requires_grad=True
            )
        else:
            self.mask_value = nn.Parameter(
                torch.zeros(mask_dim) + mask_value, requires_grad=False
            )

    def forward(self, x, pe, edge_index, edge_attr, batch):
        x_pe = self.pe_norm(pe)

        x = self.encoder(x)
        x = self.input_norm(x)

        x = torch.cat((x, x_pe), 1)
        for layer in self.layers:
            x = layer["conv"](
                x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch
            )
            x = layer["norm"](x)

        x = self.pre_decoder_norm(x)
        x = self.decoder(x)

        return x
