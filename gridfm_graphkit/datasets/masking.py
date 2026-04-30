import torch
from torch_geometric.transforms import BaseTransform
from gridfm_graphkit.datasets.globals import (
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
    ANG_MIN,
    ANG_MAX,
    RATE_A,
)
from torch_geometric.utils import degree
from torch_geometric.nn import MessagePassing


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
        mask_branch[:, ANG_MIN] = True
        mask_branch[:, ANG_MAX] = True
        mask_branch[:, RATE_A] = True

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


class BusToGenBroadcaster(MessagePassing):
    """Broadcast per-bus values to connected generators via graph propagation."""
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
    """Add configurable noise/outliers and masks to simulate measured quantities."""
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

    def add_noise(self, in_tensor, mask, std):
        if self.measurement_distribution == "Gaussian":
            return torch.where(
                mask,
                in_tensor,
                in_tensor + std * torch.randn(std.shape),
            )

        elif self.measurement_distribution == "Laplace":
            b = std / torch.sqrt(torch.tensor(2))
            dist = torch.distributions.laplace.Laplace(0, 1)
            return torch.where(mask, in_tensor, in_tensor + b * dist.sample(b.shape))

        elif self.measurement_distribution == "Uniform":
            dist = torch.distributions.uniform.Uniform(-1, 1)
            return torch.where(
                mask,
                in_tensor,
                in_tensor + torch.sqrt(torch.tensor(3)) * std * dist.sample(std.shape),
            )

    def add_outliers(self, in_tensor, mask_outliers, std):
        random_signs = 2 * torch.randint(0, 2, in_tensor.shape) - 1
        outlier_samples = 3 * std * random_signs
        return torch.where(mask_outliers, in_tensor + outlier_samples, in_tensor)

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
            data[("bus", "connected_to", "gen")].edge_index,
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

        data["bus"].x[:, : data["bus"].y.size(1)] = self.add_outliers(
            data["bus"].x[:, : data["bus"].y.size(1)],
            outliers_bus,
            std_bus,
        )

        data[("bus", "connects", "bus")].edge_attr[:, :2] = self.add_outliers(
            data[("bus", "connects", "bus")].edge_attr[:, :2],
            outliers_branch,
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
