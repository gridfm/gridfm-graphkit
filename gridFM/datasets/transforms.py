from gridFM.datasets.globals import PQ, PV, REF, PG, QG, VM, VA, G, B

import torch
from torch_geometric.transforms import BaseTransform


class AddEdgeWeights(BaseTransform):
    r"""Computes the magnitude of the complex admittance
    (stored in edge features) and adds it as an `edge_weight` attribute.
    """

    def forward(self, data):
        assert hasattr(data, "edge_attr"), "Data must have 'edge_attr'."

        # Extract real and imaginary parts of admittance
        real = data.edge_attr[:, G]
        imag = data.edge_attr[:, B]

        # Compute the magnitude of the complex admittance
        edge_weight = torch.sqrt(real**2 + imag**2)

        # Add the computed edge weights to the data object
        data.edge_weight = edge_weight

        return data


class AddIdentityMask(BaseTransform):
    r"""Creates an identity mask, and adds it as a `mask` attribute.

    The mask is generated such that every entry is False, so no masking is actually applied
    """

    def __init__(self):
        super().__init__()

    def forward(self, data):
        assert hasattr(data, "y"), "Data must have ground truth 'y'."

        # Generate an identity mask
        mask = torch.zeros_like(data.y, dtype=torch.bool)

        # Add the mask to the data object
        data.mask = mask

        return data


class AddRandomMask(BaseTransform):
    r"""Creates a random mask, and adds it as a `mask` attribute.

    The mask is generated such that each entry is `True` with probability
    `mask_ratio` and `False` otherwise.
    """

    def __init__(self, mask_dim, mask_ratio):
        super().__init__()
        self.mask_dim = mask_dim
        self.mask_ratio = mask_ratio

    def forward(self, data):
        assert hasattr(data, "x"), "Data must have node features 'x'."

        # Generate a random mask
        mask = torch.rand(data.x.size(0), self.mask_dim) < self.mask_ratio

        # Add the mask to the data object
        data.mask = mask

        return data


class AddPFMask(BaseTransform):
    r"""A transform that creates a mask according to the power flow problem and assigns it as a `mask` attribute."""

    def __init__(self):
        super().__init__()

    def forward(self, data):
        # Ensure the data object has the required attributes
        assert hasattr(data, "x"), "Data must have node features 'x'."
        assert hasattr(data, "y"), "Data must have ground truth 'y'."

        # Generate masks for each type of node
        mask_PQ = data.x[:, PQ] == 1  # PQ buses
        mask_PV = data.x[:, PV] == 1  # PV buses
        mask_REF = data.x[:, REF] == 1  # Reference buses

        # Initialize the mask tensor with False values
        mask = torch.zeros_like(data.y, dtype=torch.bool)

        mask[mask_PQ, VM] = True  # Mask Vm for PQ buses
        mask[mask_PQ, VA] = True  # Mask Va for PQ buses

        mask[mask_PV, QG] = True  # Mask Qg for PV buses
        mask[mask_PV, VA] = True  # Mask Va for PV buses

        mask[mask_REF, PG] = True  # Mask Pg for REF buses
        mask[mask_REF, QG] = True  # Mask Qg for REF buses

        # Attach the mask to the data object
        data.mask = mask

        return data


class AddOPFMask(BaseTransform):
    r"""A transform that creates a mask according to the optimal power flow problem and assigns it as a `mask` attribute."""

    def __init__(self):
        super().__init__()

    def forward(self, data):
        # Ensure the data object has the required attributes
        assert hasattr(data, "x"), "Data must have node features 'x'."
        assert hasattr(data, "y"), "Data must have ground truth 'y'."

        # Generate masks for each type of node
        mask_PQ = data.x[:, PQ] == 1  # PQ buses
        mask_PV = data.x[:, PV] == 1  # PV buses
        mask_REF = data.x[:, REF] == 1  # Reference buses

        # Initialize the mask tensor with False values
        mask = torch.zeros_like(data.y, dtype=torch.bool)

        mask[mask_PQ, VM] = True  # Mask Vm for PQ
        mask[mask_PQ, VA] = True  # Mask Va for PQ

        mask[mask_PV, PG] = True  # Mask Pg for PV
        mask[mask_PV, QG] = True  # Mask Qg for PV
        mask[mask_PV, VM] = True  # Mask Vm for PV
        mask[mask_PV, VA] = True  # Mask Va for PV

        mask[mask_REF, PG] = True  # Mask Pg for REF
        mask[mask_REF, QG] = True  # Mask Qg for REF
        mask[mask_REF, VM] = True  # Mask Vm for REF
        mask[mask_REF, VA] = True  # Mask Va for REF

        # Attach the mask to the data object
        data.mask = mask

        return data
