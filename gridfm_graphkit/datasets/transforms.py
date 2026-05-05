from copy import deepcopy
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import HeteroData
from gridfm_graphkit.datasets.globals import (
    # Generator feature indices
    G_ON,
    # Edge feature indices
    B_ON,
    YFF_TT_I,
    YFF_TT_R,
    YFT_TF_I,
    YFT_TF_R,
)
from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer


class RemoveInactiveGenerators(BaseTransform):
    """
    Removes generators where G_ON == 0.
    Uses the global index G_ON to access generator on/off flag.
    """

    def forward(self, data):
        # Mask of generators that are ON
        active_mask = data["gen"].x[:, G_ON] == 1

        num_gen = data["gen"].num_nodes

        # Mapping old generator IDs → new compact IDs
        old_to_new = torch.full((num_gen,), -1, dtype=torch.long)
        old_to_new[active_mask] = torch.arange(active_mask.sum())

        # Filter generator node features
        data["gen"].x = data["gen"].x[active_mask]
        data["gen"].x = data["gen"].x[:, :G_ON]
        data["gen"].y = data["gen"].y[active_mask]

        # ---- Update hetero edges ----

        # gen → bus edges
        e = data["gen", "connected_to", "bus"].edge_index
        keep = active_mask[e[0]]  # generator is source
        new_e = e[:, keep].clone()
        new_e[0] = old_to_new[new_e[0]]
        data["gen", "connected_to", "bus"].edge_index = new_e

        # bus → gen edges
        e = data["bus", "connected_to", "gen"].edge_index
        keep = active_mask[e[1]]  # generator is target
        new_e = e[:, keep].clone()
        new_e[1] = old_to_new[new_e[1]]
        data["bus", "connected_to", "gen"].edge_index = new_e

        return data


class RemoveInactiveBranches(BaseTransform):
    """
    Removes branches where B_ON == 0.
    Uses global index B_ON in edge_attr.
    """

    def forward(self, data):
        et = ("bus", "connects", "bus")

        # Mask for active (in-service) branches
        active_mask = data[et].edge_attr[:, B_ON] == 1

        # Apply the mask
        data[et].edge_index = data[et].edge_index[:, active_mask]
        data[et].edge_attr = data[et].edge_attr[active_mask]
        data[et].edge_attr = data[et].edge_attr[:, :B_ON]
        data[et].y = data[et].y[active_mask]

        return data


class ApplyMasking(BaseTransform):
    """
    Apply masking to data
    """

    def __init__(self, args):
        super().__init__()
        self.mask_value = args.data.mask_value

    def forward(self, data):
        data.x_dict["bus"][data.mask_dict["bus"]] = self.mask_value
        data.x_dict["gen"][data.mask_dict["gen"]] = self.mask_value
        data.edge_attr_dict[("bus", "connects", "bus")][data.mask_dict["branch"]] = (
            self.mask_value
        )

        return data


class LoadGridParamsFromPath(BaseTransform):
    """Inject static grid parameters from a saved grid template into each sample."""

    def __init__(self, args):
        super().__init__()
        self.grid_path = args.task.grid_path
        self.grid_data = HeteroData.from_dict(
            torch.load(self.grid_path, weights_only=True),
        )

        # Normalizer is needed in order to normalize the grid_data in case the input data is normalized
        self.normalizer = HeteroDataMVANormalizer(args)

        # Set to a dummy value since it is needed for the normalizer transform, but the column vn_kv will be ignored.
        self.normalizer.vn_kv_max = 1

    def forward(self, data):
        if hasattr(data, "is_normalized"):
            self.normalizer.baseMVA = data.baseMVA
            grid_data = deepcopy(self.grid_data)
            self.normalizer.transform(grid_data)
        else:
            grid_data = deepcopy(self.grid_data)

        cols = [YFF_TT_R, YFF_TT_I, YFT_TF_R, YFT_TF_I, B_ON]
        data[("bus", "connects", "bus")].edge_attr[:, cols] = grid_data[
            ("bus", "connects", "bus")
        ].edge_attr[:, cols]
        data["gen"].x[:, G_ON] = grid_data["gen"].x[:, G_ON]
        return data
