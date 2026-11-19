import os
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
    TAP,
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
    """Replace the admittances and element status with those of a reference grid.

    The measurements and the target state still come from the scenario. Only
    the grid model given to the estimator changes. This simulates admittance
    and topology errors the estimator does not know about. Use it for
    evaluation only.

    `task.grid_path` points to one processed scenario (`processed/data_index_<i>.pt`)
    of a dataset generated without admittance or topology perturbations.
    The reference grid must have the same branches, in the same order, as the scenario.
    The file is read on first use, so that dataset only has to be processed by then.
    """

    def __init__(self, args):
        super().__init__()
        self.grid_path = args.task.grid_path
        self.grid_data = None
        # Only the admittances are copied, so vn_kv_max does not matter.
        self.normalizer = HeteroDataMVANormalizer(args)
        self.normalizer.vn_kv_max = 1

    def forward(self, data):
        if self.grid_data is None:
            if not os.path.exists(self.grid_path):
                raise FileNotFoundError(
                    f"Reference grid {self.grid_path} not found. Process the unperturbed dataset first, "
                    "for example by evaluating once with --data_path on it.",
                )
            self.grid_data = HeteroData.from_dict(
                torch.load(self.grid_path, weights_only=True),
            )
        # The scenario is already normalized. Normalize the reference grid the same way.
        self.normalizer.baseMVA = data.baseMVA
        grid_data = deepcopy(self.grid_data)
        self.normalizer.transform(grid_data)

        cols = [YFF_TT_R, YFF_TT_I, YFT_TF_R, YFT_TF_I, B_ON]
        data[("bus", "connects", "bus")].edge_attr[:, cols] = grid_data[
            ("bus", "connects", "bus")
        ].edge_attr[:, cols]
        data["gen"].x[:, G_ON] = grid_data["gen"].x[:, G_ON]
        return data


class NoiseGridParameters(BaseTransform):
    """Give the estimator noisy branch resistance and reactance.

    Each branch r and x is multiplied by a factor drawn from U(1 - sigma, 1 + sigma),
    with sigma = `task.grid_noise_r` and `task.grid_noise_x`. The admittances in
    the input are rebuilt from the noisy values. Line charging is not changed.
    The true and noisy (r, x, bc) are stored in `data["impedance"].y` and `.x`.
    With both sigmas at 0 the data is unchanged.
    """

    def __init__(self, args):
        super().__init__()
        self.grid_noise_r = getattr(args.task, "grid_noise_r", 0)
        self.grid_noise_x = getattr(args.task, "grid_noise_x", 0)
        self.grid_noise_bc = getattr(args.task, "grid_noise_bc", 0)

    @staticmethod
    def add_noise_admittance(adm: torch.Tensor, sigma: float):
        """Multiply by U(1 - sigma, 1 + sigma)."""
        if sigma == 0:
            return adm
        assert sigma > 0
        dist = torch.distributions.uniform.Uniform(1 - sigma, 1 + sigma)
        return adm * dist.sample(adm.shape)

    def forward(self, data):
        if (
            (self.grid_noise_bc == 0)
            and (self.grid_noise_r == 0)
            and (self.grid_noise_x == 0)
        ):
            return data

        # Forward edges come first, then the reverse edges in the same order.
        edge_attr = deepcopy(data[("bus", "connects", "bus")].edge_attr)
        assert edge_attr.shape[0] % 2 == 0
        n = edge_attr.shape[0] // 2
        tap = edge_attr[:n, TAP]
        g = -tap * edge_attr[:n, YFT_TF_R]
        b = -tap * edge_attr[:n, YFT_TF_I]
        bc = 2 * (tap**2 * edge_attr[:n, YFF_TT_I] - b)
        r, x = g / (g**2 + b**2), -b / (g**2 + b**2)

        data["impedance"].y = torch.stack([r, x, bc], dim=1)
        r_n = self.add_noise_admittance(r, self.grid_noise_r)
        x_n = self.add_noise_admittance(x, self.grid_noise_x)
        bc_n = self.add_noise_admittance(bc, self.grid_noise_bc)
        data["impedance"].x = torch.stack([r_n, x_n, bc_n], dim=1)

        g_n, b_n = r_n / (r_n**2 + x_n**2), -x_n / (r_n**2 + x_n**2)
        edge_attr[:n, YFF_TT_R] = 1 / tap**2 * g_n
        edge_attr[:n, YFF_TT_I] = 1 / tap**2 * (b_n + bc_n / 2)
        edge_attr[:n, YFT_TF_R] = -1 / tap * g_n
        edge_attr[:n, YFT_TF_I] = -1 / tap * b_n

        edge_attr[n:, YFF_TT_R] = g_n
        edge_attr[n:, YFF_TT_I] = b_n + bc_n / 2
        edge_attr[n:, YFT_TF_R] = -1 / tap * g_n
        edge_attr[n:, YFT_TF_I] = -1 / tap * b_n

        data[("bus", "connects", "bus")].edge_attr = edge_attr
        return data
