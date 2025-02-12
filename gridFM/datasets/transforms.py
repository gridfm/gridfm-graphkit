import torch
from torch_geometric.transforms import BaseTransform
from gridFM.datasets.globals import *

class AddEdgeWeights(BaseTransform):
    r"""A transform that computes the magnitude of the complex admittance
    (stored in edge features) and adds it as an `edge_weight` attribute.

    Assumes that the edge features tensor contains two columns:
    the real and imaginary parts of the admittance, respectively.
    """
    def forward(self, data):
        # Ensure edge features exist and have the correct size
        assert hasattr(data, 'edge_attr'), "Data must have 'edge_attr'."

        # Extract real and imaginary parts of admittance
        real = data.edge_attr[:, G]
        imag = data.edge_attr[:, B]

        # Compute the magnitude of the complex admittance
        edge_weight = torch.sqrt(real**2 + imag**2)

        # Add the computed edge weights to the data object
        data.edge_weight = edge_weight

        return data
