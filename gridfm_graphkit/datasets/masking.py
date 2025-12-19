from gridfm_graphkit.io.registries import MASKING_REGISTRY

import torch
from torch_geometric.transforms import BaseTransform
from gridfm_graphkit.datasets.globals import (
    # Node indices
    PG,
    QG,
    VM,
    VA,
    PQ,
    PV,
    REF,
    # Bus feature indices
    PD_H,
    QD_H,
    QG_H,
    VM_H,
    VA_H,
    PQ_H,
    PV_H,
    REF_H,
    MIN_VM_H,
    MAX_VM_H,
    MIN_QG_H,
    MAX_QG_H,
    BS,
    GS,
    VN_KV,
    # Generator feature indices
    PG_H,
    MIN_PG,
    MAX_PG,
    C0_H,
    C1_H,
    C2_H,
    # Edge feature indices
    P_E,
    Q_E,
)
from torch_geometric.utils import degree
from torch_geometric.nn import MessagePassing


@MASKING_REGISTRY.register("none")
class AddIdentityMask(BaseTransform):
    """Creates an identity mask, and adds it as a `mask` attribute.

    The mask is generated such that every entry is False, so no masking is actually applied
    """

    def __init__(self, args):
        super().__init__()

    def forward(self, data):
        if not hasattr(data, "y"):
            raise AttributeError("Data must have ground truth 'y'.")

        # Generate an identity mask
        mask = torch.zeros_like(data.y, dtype=torch.bool)

        # Add the mask to the data object
        data.mask = mask

        return data


@MASKING_REGISTRY.register("rnd")
class AddRandomMask(BaseTransform):
    """Creates a random mask, and adds it as a `mask` attribute.

    The mask is generated such that each entry is `True` with probability
    `mask_ratio` and `False` otherwise.
    """

    def __init__(self, args):
        super().__init__()
        self.mask_dim = args.data.mask_dim
        self.mask_ratio = args.data.mask_ratio

    def forward(self, data):
        if not hasattr(data, "x"):
            raise AttributeError("Data must have node features 'x'.")

        # Generate a random mask
        mask = torch.rand(data.x.size(0), self.mask_dim) < self.mask_ratio

        # Add the mask to the data object
        data.mask = mask

        return data


class AddPFHeteroMask(BaseTransform):
    """Creates masks for a heterogeneous power flow graph."""

    def __init__(self):
        super().__init__()

    def forward(self, data):
        bus_x = data.x_dict["bus"]
        gen_x = data.x_dict["gen"]

        # Identify bus types
        mask_PQ = bus_x[:, PQ_H] == 1
        mask_PV = bus_x[:, PV_H] == 1
        mask_REF = bus_x[:, REF_H] == 1

        # Initialize mask tensors
        mask_bus = torch.zeros_like(bus_x, dtype=torch.bool)
        mask_gen = torch.zeros_like(gen_x, dtype=torch.bool)

        mask_bus[:, MIN_VM_H] = True
        mask_bus[:, MAX_VM_H] = True
        mask_bus[:, MIN_QG_H] = True
        mask_bus[:, MAX_QG_H] = True
        mask_bus[:, VN_KV] = True

        mask_gen[:, MIN_PG] = True
        mask_gen[:, MAX_PG] = True
        mask_gen[:, C0_H] = True
        mask_gen[:, C1_H] = True
        mask_gen[:, C2_H] = True

        # --- PQ buses ---
        mask_bus[mask_PQ, VM_H] = True
        mask_bus[mask_PQ, VA_H] = True

        # --- PV buses ---
        mask_bus[mask_PV, VA_H] = True
        mask_bus[mask_PV, QG_H] = True

        # --- REF buses ---
        mask_bus[mask_REF, VM_H] = True
        mask_bus[mask_REF, QG_H] = True
        # --- Generators connected to REF buses ---
        gen_bus_edges = data.edge_index_dict[("gen", "connected_to", "bus")]
        gen_indices, bus_indices = gen_bus_edges
        ref_gens = gen_indices[mask_REF[bus_indices]]
        mask_gen[ref_gens, PG_H] = True

        mask_branch = torch.zeros_like(
            data.edge_attr_dict[("bus", "connects", "bus")],
            dtype=torch.bool,
        )
        mask_branch[:, P_E] = True
        mask_branch[:, Q_E] = True

        data.mask_dict = {
            "bus": mask_bus,
            "gen": mask_gen,
            "branch": mask_branch,
            "PQ": mask_PQ,
            "PV": mask_PV,
            "REF": mask_REF,
        }

        return data


class AddOPFHeteroMask(BaseTransform):
    """Creates masks for a heterogeneous power flow graph."""

    def __init__(self):
        super().__init__()

    def forward(self, data):
        bus_x = data.x_dict["bus"]
        gen_x = data.x_dict["gen"]

        # Identify bus types
        mask_PQ = bus_x[:, PQ_H] == 1
        mask_PV = bus_x[:, PV_H] == 1
        mask_REF = bus_x[:, REF_H] == 1

        # Initialize mask tensors
        mask_bus = torch.zeros_like(bus_x, dtype=torch.bool)
        mask_gen = torch.zeros_like(gen_x, dtype=torch.bool)

        # --- PQ buses ---
        mask_bus[mask_PQ, VM_H] = True
        mask_bus[mask_PQ, VA_H] = True

        # --- PV buses ---
        mask_bus[mask_PV, VA_H] = True
        mask_bus[mask_PV, VM_H] = True
        mask_bus[mask_PV, QG_H] = True

        # --- REF buses ---
        mask_bus[mask_REF, QG_H] = True
        mask_bus[mask_REF, VM_H] = True

        mask_gen[:, PG_H] = True

        mask_branch = torch.zeros_like(
            data.edge_attr_dict[("bus", "connects", "bus")],
            dtype=torch.bool,
        )
        mask_branch[:, P_E] = True
        mask_branch[:, Q_E] = True

        data.mask_dict = {
            "bus": mask_bus,
            "gen": mask_gen,
            "branch": mask_branch,
            "PQ": mask_PQ,
            "PV": mask_PV,
            "REF": mask_REF,
        }

        return data


class AddPretrainMask(BaseTransform):
    """
    Probabilistic masking for pre-training a power flow heterogeneous graph.

    Strategy:
    - Vm, Va at buses → mask with p_high
    - Pg at generators → mask with p_high
    - Pd, Qd, node type flags (PQ, PV, REF) → never masked
    - Qg → always masked
    - Other bus features → mask with p_low
    - Other generator features → mask with p_low
    - P, Q on edges → mask with p_low
    """

    def __init__(self, args):
        super().__init__()
        self.p_high = args.data.mask_p_high
        self.p_low = args.data.mask_p_low

    def forward(self, data):
        bus_x = data.x_dict["bus"]
        gen_x = data.x_dict["gen"]
        # Identify bus types
        mask_PQ = bus_x[:, PQ_H] == 1
        mask_PV = bus_x[:, PV_H] == 1
        mask_REF = bus_x[:, REF_H] == 1

        # ===================
        # === BUS MASKING ===
        # ===================
        bus_x = data.x_dict["bus"]
        num_bus, bus_dim = bus_x.shape

        # Mask first with low probability
        mask_bus = torch.rand(num_bus, bus_dim) < self.p_low

        # Overwrite the masking with HIGH probability for Vm, Va
        mask_bus[:, VM_H] = torch.rand(num_bus) < self.p_high
        mask_bus[:, VA_H] = torch.rand(num_bus) < self.p_high

        # NEVER mask Pd, Qd, PQ/PV/REF flags
        never_bus = torch.tensor([PD_H, QD_H, PQ_H, PV_H, REF_H, BS, GS])
        mask_bus[:, never_bus] = False

        # ALWAYS mask Qg
        mask_bus[:, QG_H] = True

        # ======================
        # === GENERATOR MASK ===
        # ======================
        gen_x = data.x_dict["gen"]
        num_gen, gen_dim = gen_x.shape
        mask_gen = torch.rand(num_gen, gen_dim) < self.p_low

        # Enforce that Pg follows p_high, overwrite
        mask_gen[:, PG_H] = torch.rand(num_gen) < self.p_high

        # ======================
        # === BRANCH MASKING ===
        # ======================
        branch_x = data.edge_attr_dict[("bus", "connects", "bus")]
        num_br, br_dim = branch_x.shape

        mask_branch = torch.zeros((num_br, br_dim), dtype=torch.bool)

        # P, Q masked with low probability
        mask_branch[:, P_E] = torch.rand(num_br) < self.p_low
        mask_branch[:, Q_E] = torch.rand(num_br) < self.p_low

        data.mask_dict = {
            "bus": mask_bus,
            "gen": mask_gen,
            "branch": mask_branch,
            "PQ": mask_PQ,
            "PV": mask_PV,
            "REF": mask_REF,
        }

        return data


class BusToGenBroadcaster(MessagePassing):
    def __init__(self, aggr="add"):
        super().__init__(aggr=aggr)

    def forward(self, x_bus, edge_index_bus2gen, num_gen):
        # TODO propagate the standard deviation by dividing by sqrt of number of gens per bus
        deg = degree(edge_index_bus2gen[0], num_nodes=x_bus.shape[0]).unsqueeze(-1)
        return self.propagate(
            edge_index_bus2gen,
            x=x_bus / torch.sqrt(deg),
            size=(x_bus.size(0), num_gen),
        )

    def message(self, x_j):
        return x_j


class SimulateMeasurements(BaseTransform):
    def __init__(self, args):
        super().__init__()
        self.measurements = args.task.measurements
        self.relative_measurement = getattr(args.task, "relative_measurement", True)
        self.measurement_distribution = getattr(args.task, "noise_type", "Gaussian")
        self.bus2gen_broadcaster = BusToGenBroadcaster()

    def place_measurement_std_and_outliers(self, std, outliers, features, measurement):
        measurement_mask = torch.rand(std.shape[0]) < measurement.mask_ratio
        outliers_mask = torch.rand(std.shape[0]) < measurement.outlier_ratio
        outliers_mask = torch.logical_and(outliers_mask, ~measurement_mask)
        for feature in features:
            std[~measurement_mask, feature] = measurement.std
            outliers[outliers_mask, feature] = True
        return std, outliers

    def add_noise(self, data, mask, std):
        if self.measurement_distribution == "Gaussian":
            return torch.where(mask, data, data + std * torch.randn(std.shape))

        elif self.measurement_distribution == "Laplace":
            b = std / torch.sqrt(torch.tensor(2))
            dist = torch.distributions.laplace.Laplace(0, 1)
            return torch.where(mask, data, data + b * dist.sample(b.shape))

        elif self.measurement_distribution == "Uniform":
            dist = torch.distributions.uniform.Uniform(-1, 1)
            return torch.where(
                mask,
                data,
                data + torch.sqrt(torch.tensor(3)) * std * dist.sample(std.shape),
            )

    def forward(self, data):
        std_bus = torch.full_like(data["bus"].y, float("inf"), dtype=torch.float)
        outliers_bus = torch.full_like(data["bus"].y, False, dtype=torch.bool)

        std_bus, outliers_bus = self.place_measurement_std_and_outliers(
            std_bus,
            outliers_bus,
            [VM_H],
            self.measurements.vm,
        )
        std_bus, outliers_bus = self.place_measurement_std_and_outliers(
            std_bus,
            outliers_bus,
            [PD_H, QD_H, QG_H],
            self.measurements.power_inj,
        )
        std_gen = self.bus2gen_broadcaster(
            std_bus[:, [PD_H]],
            data[("bus", "connected_to", "gen")]["edge_index"],
            data["gen"].x.shape[0],
        )

        std_branch = torch.full_like(
            data[("bus", "connects", "bus")].edge_attr[:, :2],
            float("inf"),
            dtype=torch.float,
        )
        outliers_branch = torch.full_like(
            data[("bus", "connects", "bus")].edge_attr[:, :2],
            False,
            dtype=torch.bool,
        )

        std_branch, outliers_branch = self.place_measurement_std_and_outliers(
            std_branch,
            outliers_branch,
            [P_E, Q_E],
            self.measurements.power_flow,
        )
        mask_bus, mask_branch, mask_gen = (
            torch.isinf(std_bus),
            torch.isinf(std_branch),
            torch.isinf(std_gen),
        )

        if self.relative_measurement:
            std_bus = torch.where(mask_bus, std_bus, std_bus * torch.abs(data["bus"].y))
            std_branch = torch.where(
                mask_branch,
                std_branch,
                std_branch
                * torch.abs(data[("bus", "connects", "bus")].edge_attr[:, :2]),
            )
        else:
            std_bus = torch.where(mask_bus, std_bus, std_bus * data.baseMVA)
            std_branch = torch.where(mask_branch, std_branch, std_branch * data.baseMVA)

        data["bus"].x[:, : data["bus"].y.size(1)] = self.add_noise(
            data["bus"].x[:, : data["bus"].y.size(1)],
            mask_bus,
            std_bus,
        )
        data["gen"].x[:, : data["gen"].y.size(1)] = self.add_noise(
            data["gen"].x[:, : data["gen"].y.size(1)],
            mask_gen,
            std_gen,
        )
        data[("bus", "connects", "bus")].edge_attr[:, :2] = self.add_noise(
            data[("bus", "connects", "bus")].edge_attr[:, :2],
            mask_branch,
            std_branch,
        )

        # Save all masks and stds
        extra_dims_bus = data["bus"].x.size(1) - data["bus"].y.size(1)
        extra_dims_gen = data["gen"].x.size(1) - data["gen"].y.size(1)
        extra_dims_branch = (
            data[("bus", "connects", "bus")]["edge_attr"].shape[1]
            - mask_branch.shape[1]
        )

        data.mask_dict = {
            "bus": torch.nn.functional.pad(mask_bus, (0, extra_dims_bus)),
            "std_bus": std_bus,
            "outliers_bus": outliers_bus,
            "gen": torch.nn.functional.pad(mask_gen, (0, extra_dims_gen)),
            "std_gen": std_gen,
            "branch": torch.nn.functional.pad(mask_branch, (0, extra_dims_branch)),
            "std_branch": std_branch,
            "outliers_branch": outliers_branch,
        }

        return data


@MASKING_REGISTRY.register("pf")
class AddPFMask(BaseTransform):
    """Creates a mask according to the power flow problem and assigns it as a `mask` attribute."""

    def __init__(self, args):
        super().__init__()

    def forward(self, data):
        # Ensure the data object has the required attributes
        if not hasattr(data, "y"):
            raise AttributeError("Data must have ground truth 'y'.")

        if not hasattr(data, "x"):
            raise AttributeError("Data must have node features 'x'.")

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


@MASKING_REGISTRY.register("opf")
class AddOPFMask(BaseTransform):
    """Creates a mask according to the optimal power flow problem and assigns it as a `mask` attribute."""

    def __init__(self, args):
        super().__init__()

    def forward(self, data):
        # Ensure the data object has the required attributes
        if not hasattr(data, "y"):
            raise AttributeError("Data must have ground truth 'y'.")

        if not hasattr(data, "x"):
            raise AttributeError("Data must have node features 'x'.")

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
