from typing import List
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
    """Copy a bus feature to every generator on that bus."""

    def __init__(self, aggr="add"):
        # Each generator is on one bus and gets one message, so the aggregation does not matter.
        super().__init__(aggr=aggr)

    def forward(self, x_bus, edge_index_bus2gen, num_gen):
        return self.propagate(
            edge_index_bus2gen,
            x=x_bus,
            size=(x_bus.size(0), num_gen),
        )

    def message(self, x_j):
        return x_j


class SimulateMeasurements(BaseTransform):
    """Turn the true state of a scenario into noisy, partial SCADA measurements.

    Measured values are voltage magnitude, bus injections (loads, bus Qg and
    generator Pg) and branch flows. For each measurement type:

    - a fraction `mask_ratio` is not measured and is masked,
    - each measured value gets Gaussian noise with relative standard deviation `std`,
    - a fraction `outlier_ratio` of the measured values is shifted by 3 standard deviations.

    The random draws are seeded from the scenario's voltage magnitudes, so one
    scenario always gets the same measurements. The neural estimator and the
    WLS baseline therefore see the same measurements.
    """

    def __init__(self, args):
        super().__init__()
        self.measurements = args.task.measurements
        self.relative_measurement = getattr(args.task, "relative_measurement", True)
        if not self.relative_measurement:
            raise ValueError(
                "Only relative measurement noise is supported. Set task.relative_measurement to true.",
            )
        noise_type = getattr(args.task, "noise_type", "Gaussian")
        if noise_type != "Gaussian":
            raise ValueError(
                f"Only Gaussian measurement noise is supported, got noise_type {noise_type}.",
            )
        self.bus2gen_broadcaster = BusToGenBroadcaster()

    def place_measurement_std_and_outliers(
        self,
        std: torch.Tensor,
        outliers: torch.Tensor,
        features: List[int],
        measurement,
    ):
        """Pick which rows are measured and which measured rows are outliers.

        All `features` of a row share one draw. For example, P and Q of a branch
        flow are measured together and are outliers together.
        """
        # True means not measured.
        measurement_mask = torch.rand(std.shape[0]) < measurement.mask_ratio
        outliers_mask = torch.rand(std.shape[0]) < measurement.outlier_ratio
        outliers_mask = torch.logical_and(outliers_mask, ~measurement_mask)
        for feature in features:
            std[~measurement_mask, feature] = measurement.std
            outliers[outliers_mask, feature] = True
        return std, outliers

    def add_noise(self, in_tensor: torch.Tensor, mask: torch.Tensor, std: torch.Tensor):
        """Add zero-mean Gaussian noise with standard deviation `std` to the entries where `mask` is False."""
        return torch.where(mask, in_tensor, in_tensor + std * torch.randn(std.shape))

    def add_outliers(
        self,
        in_tensor: torch.Tensor,
        mask_outliers: torch.Tensor,
        std: torch.Tensor,
    ):
        """Shift the entries where `mask_outliers` is True by 3 standard deviations, with a random sign."""
        random_signs = 2 * torch.randint(0, 2, in_tensor.shape) - 1
        outlier_samples = 3 * std * random_signs
        return torch.where(mask_outliers, in_tensor + outlier_samples, in_tensor)

    def forward(self, data):
        # Seed from the scenario itself. The global random state is restored at the end.
        cpu_state = torch.random.get_rng_state()
        random_seed = int((data["bus"].x[:, VM_H] * 1000).sum())
        torch.manual_seed(random_seed)

        # Relative standard deviation for PD, QD, QG, VM, VA. Infinity means not measured.
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

        # A bus injection measurement also covers the Pg of the generators on that bus.
        std_gen = self.bus2gen_broadcaster(
            std_bus[:, [PD_H]],
            data[("bus", "connected_to", "gen")].edge_index,
            data["gen"].x.shape[0],
        )

        # Generator Pg measurements never get outliers.

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

        # Relative to absolute standard deviation. torch.where keeps unmeasured entries
        # at infinity instead of inf * 0 = nan.
        std_bus = torch.where(mask_bus, std_bus, std_bus * torch.abs(data["bus"].y))
        std_gen = torch.where(mask_gen, std_gen, std_gen * torch.abs(data["gen"].y))
        std_branch = torch.where(
            mask_branch,
            std_branch,
            std_branch * torch.abs(data[("bus", "connects", "bus")].edge_attr[:, :2]),
        )

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

        # std and outlier tensors have y.size(1) columns. Masks are padded to x.size(1)
        # columns for ApplyMasking. The padded columns are not masked here.
        extra_dims_bus = data["bus"].x.size(1) - data["bus"].y.size(1)
        extra_dims_gen = data["gen"].x.size(1) - data["gen"].y.size(1)
        extra_dims_branch = (
            data[("bus", "connects", "bus")]["edge_attr"].shape[1]
            - mask_branch.shape[1]
        )
        mask_bus = torch.nn.functional.pad(mask_bus, (0, extra_dims_bus))
        mask_gen = torch.nn.functional.pad(mask_gen, (0, extra_dims_gen))
        mask_branch = torch.nn.functional.pad(mask_branch, (0, extra_dims_branch))

        # OPF limits and costs are zeroed in StateEstimationTask.shared_step, because the WLS baseline needs them.
        data.mask_dict = {
            "bus": mask_bus,
            "std_bus": std_bus,
            "outliers_bus": outliers_bus,
            "gen": mask_gen,
            "std_gen": std_gen,
            "branch": mask_branch,
            "std_branch": std_branch,
            "outliers_branch": outliers_branch,
        }
        torch.random.set_rng_state(cpu_state)
        return data
