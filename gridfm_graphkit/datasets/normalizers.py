from gridfm_graphkit.io.registries import NORMALIZERS_REGISTRY
import os
import torch
from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from gridfm_graphkit.datasets.globals import (
    # Bus feature indices
    PD_H,
    QD_H,
    QG_H,
    VA_H,
    MIN_QG_H,
    MAX_QG_H,
    GS,
    BS,
    VN_KV,
    # Output feature indices
    PG_OUT,
    QG_OUT,
    PG_OUT_GEN,
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
    YFF_TT_R,
    YFT_TF_I,
    ANG_MIN,
    ANG_MAX,
    RATE_A,
)


class Normalizer(ABC):
    """
    Abstract base class for all normalization strategies.
    """

    # Subclasses should set this to "fit_on_train" or "fit_on_dataset"
    fit_strategy: str = "fit_on_train"

    @abstractmethod
    def fit(self, data_path: str, scenario_ids: List[int]) -> dict:
        """
        Fit normalization parameters from raw data on disk.

        Args:
            data_path: Path to the raw data directory containing parquet files.
            scenario_ids: List of scenario IDs to use for fitting.

        Returns:
            Dictionary of computed parameters.
        """

    @abstractmethod
    def fit_from_dict(self, params: dict):
        """
        Set parameters from a precomputed dictionary.

        Args:
            params: Dictionary of parameters.
        """

    @abstractmethod
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input data.

        Args:
            data: Input tensor.

        Returns:
            Normalized tensor.
        """

    @abstractmethod
    def inverse_transform(self, normalized_data: torch.Tensor) -> torch.Tensor:
        """
        Undo normalization.

        Args:
            normalized_data: Normalized tensor.

        Returns:
            Original tensor.
        """

    @abstractmethod
    def get_stats(self) -> dict:
        """
        Return the stored normalization statistics for logging/inspection.
        """


@NORMALIZERS_REGISTRY.register("HeteroDataMVANormalizer")
class HeteroDataMVANormalizer(Normalizer):
    """
    In power systems, a suitable normalization strategy must preserve the physical properties of
    the system. A known method is the conversion to the per-unit (p.u.) system, which expresses
    electrical quantities such as voltage, current, power, and impedance as fractions of predefined
    base values. These base values are usually chosen based on system parameters, such as rated
    voltage. The per-unit conversion ensures that power system equations remain scale-invariant,
    preserving fundamental physical relationships.
    """

    fit_strategy = "fit_on_train"

    def __init__(self, args):
        """
        Args:
            args (NestedNamespace): Parameters

        Attributes:
            baseMVA (float): baseMVA found in casefile. From ``args.data.baseMVA``.
        """
        self.baseMVA_orig = getattr(args.data, "baseMVA", 100)
        self.baseMVA = None

    def to(self, device):
        pass

    def fit(self, data_path: str, scenario_ids: List[int]) -> dict:
        """
        Fit normalization parameters by loading raw parquet data from disk.

        Args:
            data_path: Path to the raw data directory containing bus_data.parquet and gen_data.parquet.
            scenario_ids: List of scenario IDs to use for fitting.

        Returns:
            Dictionary of computed parameters.
        """
        bus_data = pd.read_parquet(os.path.join(data_path, "bus_data.parquet"))
        gen_data = pd.read_parquet(os.path.join(data_path, "gen_data.parquet"))

        assert (
            bus_data.scenario.min() == 0
            and bus_data.scenario.max() == len(bus_data.scenario.unique()) - 1
        )

        bus_data = bus_data[bus_data["scenario"].isin(scenario_ids)]
        gen_data = gen_data[gen_data["scenario"].isin(scenario_ids)]

        if self.baseMVA is None:
            pd_values = bus_data["Pd"]
            qd_values = bus_data["Qd"]
            pg_values = gen_data["p_mw"]
            qg_values = bus_data["Qg"]

            non_zero_values = pd.concat(
                [
                    pd_values[pd_values != 0],
                    qd_values[qd_values != 0],
                    pg_values[pg_values != 0],
                    qg_values[qg_values != 0],
                ],
            )

            self.baseMVA = np.percentile(non_zero_values, 95)
        self.vn_kv_max = float(bus_data["vn_kv"].max())

        return {
            "baseMVA_orig": torch.tensor(self.baseMVA_orig, dtype=torch.float),
            "baseMVA": torch.tensor(self.baseMVA, dtype=torch.float),
            "vn_kv_max": torch.tensor(self.vn_kv_max, dtype=torch.float),
        }

    def fit_from_dict(self, params: dict):
        # Base MVA
        self.baseMVA = params.get("baseMVA").item()
        self.baseMVA_orig = params.get("baseMVA_orig").item()

        # vn_kv
        self.vn_kv_max = params.get("vn_kv_max").item()

    def transform(self, data: HeteroData):
        if self.baseMVA is None or self.baseMVA == 0:
            raise ValueError("BaseMVA not properly set")

        # --- Bus input normalization --- PD, QD, QG, MIN_QG, MAX_QG, VA, GS, BS, VN_KV (9)
        data.x_dict["bus"][:, PD_H] /= self.baseMVA
        data.x_dict["bus"][:, QD_H] /= self.baseMVA
        data.x_dict["bus"][:, QG_H] /= self.baseMVA
        data.x_dict["bus"][:, MIN_QG_H] /= self.baseMVA
        data.x_dict["bus"][:, MAX_QG_H] /= self.baseMVA
        data.x_dict["bus"][:, VA_H] *= torch.pi / 180.0
        data.x_dict["bus"][:, GS] *= self.baseMVA_orig / self.baseMVA
        data.x_dict["bus"][:, BS] *= self.baseMVA_orig / self.baseMVA
        data.x_dict["bus"][:, VN_KV] /= self.vn_kv_max

        # --- Bus label normalization --- PD, QD, QG, VA (4)
        data.y_dict["bus"][:, PD_H] /= self.baseMVA
        data.y_dict["bus"][:, QD_H] /= self.baseMVA
        data.y_dict["bus"][:, QG_H] /= self.baseMVA
        data.y_dict["bus"][:, VA_H] *= torch.pi / 180.0

        # --- Generator input normalization --- PG, MIN_PG, MAX_PG, C0, C1, C2 (6)
        data.x_dict["gen"][:, PG_H] /= self.baseMVA
        data.x_dict["gen"][:, MIN_PG] /= self.baseMVA
        data.x_dict["gen"][:, MAX_PG] /= self.baseMVA
        data.x_dict["gen"][:, C0_H] = torch.sign(
            data.x_dict["gen"][:, C0_H],
        ) * torch.log1p(torch.abs(data.x_dict["gen"][:, C0_H]))
        data.x_dict["gen"][:, C1_H] = torch.sign(
            data.x_dict["gen"][:, C1_H],
        ) * torch.log1p(torch.abs(data.x_dict["gen"][:, C1_H]))
        data.x_dict["gen"][:, C2_H] = torch.sign(
            data.x_dict["gen"][:, C2_H],
        ) * torch.log1p(torch.abs(data.x_dict["gen"][:, C2_H]))

        # --- Generator label normalization --- PG (1)
        data.y_dict["gen"][:, PG_H] /= self.baseMVA

        # --- Edge input normalization --- P_E, Q_E , Ys, ANG_MIN, ANG_MAX, RATE_A
        data.edge_attr_dict[("bus", "connects", "bus")][:, P_E] /= self.baseMVA
        data.edge_attr_dict[("bus", "connects", "bus")][:, Q_E] /= self.baseMVA
        data.edge_attr_dict[("bus", "connects", "bus")][:, YFF_TT_R : YFT_TF_I + 1] *= (
            self.baseMVA_orig / self.baseMVA
        )
        data.edge_attr_dict[("bus", "connects", "bus")][:, ANG_MIN] *= torch.pi / 180.0
        data.edge_attr_dict[("bus", "connects", "bus")][:, ANG_MAX] *= torch.pi / 180.0
        data.edge_attr_dict[("bus", "connects", "bus")][:, RATE_A] /= self.baseMVA
        data.baseMVA = self.baseMVA
        data.is_normalized = True

    def inverse_transform(self, data: HeteroData):
        if self.baseMVA is None or self.baseMVA == 0:
            raise ValueError("BaseMVA not properly set")

        if not data.is_normalized.all():
            raise ValueError("Attempting to denormalize data which is not normalized")

        if (data.baseMVA != self.baseMVA).any():
            raise ValueError(
                f"Normalizer baseMVA was {self.baseMVA} but Data object baseMVA is {data.baseMVA}",
            )

        # -------- BUS INPUT INVERSE NORMALIZATION --------
        # NOTE: VA (bus input & label) are intentionally kept in
        # radians after inverse_transform -- the physics layers (ComputeBranchFlow,
        # ComputeNodeResiduals, etc.) expect radians.
        #
        # WARNING: GS, BS, and edge admittances (Y) are NOT restored to their
        # original casefile per-unit values. The transform scales them by
        # (baseMVA_orig / baseMVA), but the inverse multiplies by baseMVA
        # (not baseMVA / baseMVA_orig), yielding physical SI units
        # (original * baseMVA_orig). This is intentional for the physics layers.
        data.x_dict["bus"][:, PD_H] *= self.baseMVA
        data.x_dict["bus"][:, QD_H] *= self.baseMVA
        data.x_dict["bus"][:, QG_H] *= self.baseMVA
        data.x_dict["bus"][:, MIN_QG_H] *= self.baseMVA
        data.x_dict["bus"][:, MAX_QG_H] *= self.baseMVA
        data.x_dict["bus"][:, GS] *= (
            self.baseMVA
        )  # -> physical units (not original p.u.)
        data.x_dict["bus"][:, BS] *= (
            self.baseMVA
        )  # -> physical units (not original p.u.)
        data.x_dict["bus"][:, VN_KV] *= self.vn_kv_max

        # -------- BUS LABEL INVERSE NORMALIZATION --------
        data.y_dict["bus"][:, PD_H] *= self.baseMVA
        data.y_dict["bus"][:, QD_H] *= self.baseMVA
        data.y_dict["bus"][:, QG_H] *= self.baseMVA

        # -------- GENERATOR INPUT INVERSE NORMALIZATION --------
        data.x_dict["gen"][:, PG_H] *= self.baseMVA
        data.x_dict["gen"][:, MIN_PG] *= self.baseMVA
        data.x_dict["gen"][:, MAX_PG] *= self.baseMVA
        data.x_dict["gen"][:, C0_H] = torch.sign(data.x_dict["gen"][:, C0_H]) * (
            torch.exp(torch.abs(data.x_dict["gen"][:, C0_H])) - 1
        )
        data.x_dict["gen"][:, C1_H] = torch.sign(data.x_dict["gen"][:, C1_H]) * (
            torch.exp(torch.abs(data.x_dict["gen"][:, C1_H])) - 1
        )
        data.x_dict["gen"][:, C2_H] = torch.sign(data.x_dict["gen"][:, C2_H]) * (
            torch.exp(torch.abs(data.x_dict["gen"][:, C2_H])) - 1
        )

        # -------- GENERATOR LABEL INVERSE NORMALIZATION --------
        data.y_dict["gen"][:, PG_H] *= self.baseMVA

        # -------- EDGE INPUT INVERSE NORMALIZATION --------
        data.edge_attr_dict[("bus", "connects", "bus")][:, P_E] *= self.baseMVA
        data.edge_attr_dict[("bus", "connects", "bus")][:, Q_E] *= self.baseMVA
        data.edge_attr_dict[("bus", "connects", "bus")][:, YFF_TT_R : YFT_TF_I + 1] *= (
            self.baseMVA  # -> physical units (not original p.u.), see WARNING above
        )

        data.edge_attr_dict[("bus", "connects", "bus")][:, ANG_MIN] *= 180.0 / torch.pi
        data.edge_attr_dict[("bus", "connects", "bus")][:, ANG_MAX] *= 180.0 / torch.pi

        data.edge_attr_dict[("bus", "connects", "bus")][:, RATE_A] *= self.baseMVA
        data.is_normalized = False

    def inverse_output(self, output, batch):
        bus_output = output["bus"]
        gen_output = output["gen"]
        bus_output[:, PG_OUT] *= self.baseMVA
        bus_output[:, QG_OUT] *= self.baseMVA
        gen_output[:, PG_OUT_GEN] *= self.baseMVA

    def get_stats(self) -> dict:
        return {
            "baseMVA_orig": torch.tensor(self.baseMVA_orig, dtype=torch.float),
            "baseMVA": torch.tensor(self.baseMVA, dtype=torch.float),
            "vn_kv_max": torch.tensor(self.vn_kv_max, dtype=torch.float),
        }


@NORMALIZERS_REGISTRY.register("HeteroDataPerSampleMVANormalizer")
class HeteroDataPerSampleMVANormalizer(Normalizer):
    """
    Per-sample MVA normalizer: each scenario (sample) gets its own baseMVA and vn_kv_max,
    computed as the 95th percentile of Pd, Qd, Pg, Qg for that scenario. Same per-unit
    formulas as HeteroDataMVANormalizer, but applied with per-scenario scales so that
    batched data with different scenarios is normalized correctly.
    """

    fit_strategy = "fit_on_dataset"

    def __init__(self, args):
        self.baseMVA_orig = getattr(
            args.data,
            "baseMVA",
            100,
        )  # casefile base MVA (for GS/BS scaling)
        self._baseMVA_lookup = None  # tensor indexed by scenario_id
        self._vn_kv_max_lookup = None
        self._scenario_ids = None  # scenario ids that were fitted (for save/load)

    def to(self, device):
        pass

    def fit(self, data_path: str, scenario_ids: List[int]) -> dict:
        """
        Compute per-scenario baseMVA and vn_kv_max by loading raw parquet data from disk.
        For each scenario: concat Pd, Qd, Pg, Qg; take 95th percentile of non-zero as baseMVA;
        max vn_kv as vn_kv_max. Build lookup tensors indexed by scenario_id (no dicts).

        Args:
            data_path: Path to the raw data directory containing bus_data.parquet and gen_data.parquet.
            scenario_ids: List of scenario IDs to use for fitting.

        Returns:
            Dictionary of computed parameters.
        """
        bus_data = pd.read_parquet(os.path.join(data_path, "bus_data.parquet"))
        gen_data = pd.read_parquet(os.path.join(data_path, "gen_data.parquet"))

        bus_data = bus_data[bus_data["scenario"].isin(scenario_ids)]
        gen_data = gen_data[gen_data["scenario"].isin(scenario_ids)]

        baseMVA = []
        vn_kv_max = []
        scenarios = []

        bus_groups = bus_data.groupby("scenario")
        gen_groups = gen_data.groupby("scenario")

        for scenario in sorted(bus_groups.groups.keys()):
            bus_group = bus_groups.get_group(scenario)
            gen_group = gen_groups.get_group(scenario)
            pd_values = bus_group["Pd"]
            qd_values = bus_group["Qd"]
            qg_values = bus_group["Qg"]
            pg_values = gen_group["p_mw"]

            all_values = pd.concat([pd_values, qd_values, pg_values, qg_values])
            non_zero_values = all_values[all_values != 0]
            baseMVA.append(np.percentile(non_zero_values, 95))
            vn_kv_max.append(float(bus_group["vn_kv"].max()))
            scenarios.append(scenario)

        scenarios_t = torch.tensor(scenarios, dtype=torch.long)
        baseMVA_t = torch.tensor(baseMVA, dtype=torch.float)
        vn_kv_max_t = torch.tensor(vn_kv_max, dtype=torch.float)
        max_sid = int(scenarios_t.max().item())
        self._baseMVA_lookup = torch.zeros(max_sid + 1, dtype=torch.float)
        self._vn_kv_max_lookup = torch.zeros(max_sid + 1, dtype=torch.float)
        self._baseMVA_lookup[scenarios_t] = baseMVA_t
        self._vn_kv_max_lookup[scenarios_t] = vn_kv_max_t
        self._scenario_ids = scenarios_t

        return {
            "baseMVA_orig": torch.tensor(self.baseMVA_orig, dtype=torch.float),
            "scenarios": scenarios_t,
            "baseMVA": baseMVA_t,
            "vn_kv_max": vn_kv_max_t,
        }

    def fit_from_dict(self, params: dict):
        """Restore lookups and baseMVA_orig from saved params (scenarios, baseMVA, vn_kv_max tensors)."""
        scenarios = params.get("scenarios")
        baseMVA = params.get("baseMVA")
        vn_kv_max = params.get("vn_kv_max")
        max_sid = int(scenarios.max().item())
        self._baseMVA_lookup = torch.zeros(max_sid + 1, dtype=torch.float)
        self._vn_kv_max_lookup = torch.zeros(max_sid + 1, dtype=torch.float)
        self._baseMVA_lookup[scenarios] = baseMVA
        self._vn_kv_max_lookup[scenarios] = vn_kv_max
        self._scenario_ids = scenarios
        bmo = params.get("baseMVA_orig")
        self.baseMVA_orig = bmo.item() if hasattr(bmo, "item") else bmo

    def _per_node_mva(self, data: HeteroData):
        """
        Get per-node and per-edge baseMVA/vn_kv_max from data.scenario_id (single sample or batch).
        Returns (b, b_orig, vn, g, e_b, e_b_orig) with shapes (n, 1) for bus, gen, edge so they broadcast.
        Fully GPU/CPU safe.
        """
        if self._baseMVA_lookup is None:
            raise ValueError("Normalizer not fitted or lookups not built")

        device = data.x_dict["bus"].device
        dtype = data.x_dict["bus"].dtype

        bus_batch = getattr(data["bus"], "batch", None)
        gen_batch = getattr(data["gen"], "batch", None)
        n_bus = data.x_dict["bus"].size(0)
        n_gen = data.x_dict["gen"].size(0)
        edge_index = data["bus", "connects", "bus"].edge_index
        n_edge = edge_index.size(1)

        scenario_id = data["scenario_id"]

        # Scenario id per node/edge
        if bus_batch is not None:
            sid_bus = scenario_id[bus_batch]
            sid_gen = scenario_id[gen_batch]
            sid_edge = scenario_id[bus_batch[edge_index[0]]]
        else:
            sid = scenario_id.item()
            sid_bus = torch.full((n_bus,), sid, device=device, dtype=torch.long)
            sid_gen = torch.full((n_gen,), sid, device=device, dtype=torch.long)
            sid_edge = torch.full((n_edge,), sid, device=device, dtype=torch.long)

        # Move lookups to correct device/dtype before indexing
        baseMVA_lookup = self._baseMVA_lookup.to(device=device, dtype=dtype)
        vn_kv_max_lookup = self._vn_kv_max_lookup.to(device=device, dtype=dtype)

        b = baseMVA_lookup[sid_bus]
        vn = vn_kv_max_lookup[sid_bus]
        g = baseMVA_lookup[sid_gen]
        e_b = baseMVA_lookup[sid_edge]

        b_orig_val = (
            self.baseMVA_orig
            if isinstance(self.baseMVA_orig, (int, float))
            else self.baseMVA_orig.item()
        )
        b_orig = torch.full_like(b, b_orig_val)
        e_b_orig = torch.full_like(e_b, b_orig_val)

        return b, b_orig, vn, g, e_b, e_b_orig

    def transform(self, data: HeteroData):
        """Apply per-unit normalization using per-scenario baseMVA/vn_kv_max (same formulas as base MVA normalizer)."""
        if self._baseMVA_lookup is None:
            raise ValueError("Normalizer not fitted or lookups not loaded")
        b, b_orig, vn, g, e_b, e_b_orig = self._per_node_mva(data)
        # --- Bus input normalization ---
        data.x_dict["bus"][:, PD_H] /= b
        data.x_dict["bus"][:, QD_H] /= b
        data.x_dict["bus"][:, QG_H] /= b
        data.x_dict["bus"][:, MIN_QG_H] /= b
        data.x_dict["bus"][:, MAX_QG_H] /= b
        data.x_dict["bus"][:, VA_H] *= torch.pi / 180.0
        data.x_dict["bus"][:, GS] *= b_orig / b
        data.x_dict["bus"][:, BS] *= b_orig / b
        data.x_dict["bus"][:, VN_KV] /= vn

        # --- Bus label normalization ---
        data.y_dict["bus"][:, PD_H] /= b
        data.y_dict["bus"][:, QD_H] /= b
        data.y_dict["bus"][:, QG_H] /= b
        data.y_dict["bus"][:, VA_H] *= torch.pi / 180.0

        # --- Generator input normalization ---
        data.x_dict["gen"][:, PG_H] /= g
        data.x_dict["gen"][:, MIN_PG] /= g
        data.x_dict["gen"][:, MAX_PG] /= g
        data.x_dict["gen"][:, C0_H] = torch.sign(
            data.x_dict["gen"][:, C0_H],
        ) * torch.log1p(torch.abs(data.x_dict["gen"][:, C0_H]))
        data.x_dict["gen"][:, C1_H] = torch.sign(
            data.x_dict["gen"][:, C1_H],
        ) * torch.log1p(torch.abs(data.x_dict["gen"][:, C1_H]))
        data.x_dict["gen"][:, C2_H] = torch.sign(
            data.x_dict["gen"][:, C2_H],
        ) * torch.log1p(torch.abs(data.x_dict["gen"][:, C2_H]))

        # --- Generator label normalization ---
        data.y_dict["gen"][:, PG_H] /= g

        # --- Edge input normalization ---
        data.edge_attr_dict[("bus", "connects", "bus")][:, P_E] /= e_b
        data.edge_attr_dict[("bus", "connects", "bus")][:, Q_E] /= e_b
        data.edge_attr_dict[("bus", "connects", "bus")][:, YFF_TT_R : YFT_TF_I + 1] *= (
            e_b_orig.unsqueeze(1) / e_b.unsqueeze(1)
        )
        data.edge_attr_dict[("bus", "connects", "bus")][:, ANG_MIN] *= torch.pi / 180.0
        data.edge_attr_dict[("bus", "connects", "bus")][:, ANG_MAX] *= torch.pi / 180.0
        data.edge_attr_dict[("bus", "connects", "bus")][:, RATE_A] /= e_b
        data.is_normalized = True

    def inverse_transform(self, data: HeteroData):
        """Undo per-unit normalization (multiply by baseMVA, rad->deg, inverse log1p for cost coeffs)."""
        if self._baseMVA_lookup is None:
            raise ValueError("Normalizer not fitted or lookups not loaded")
        if not data.is_normalized.all():
            raise ValueError("Attempting to denormalize data which is not normalized")
        b, _, vn, g, e_b, _ = self._per_node_mva(
            data,
        )  # b_orig and e_b_orig are not used

        # -------- BUS INPUT INVERSE NORMALIZATION --------
        # NOTE: VA (bus input & label) are intentionally kept in
        # radians after inverse_transform -- the physics layers (ComputeBranchFlow,
        # ComputeNodeResiduals, etc.) expect radians.
        #
        # WARNING: GS, BS, and edge admittances (Y) are NOT restored to their
        # original casefile per-unit values. The transform scales them by
        # (b_orig / b), but the inverse multiplies by b (not b / b_orig),
        # yielding physical SI units (original * b_orig). This is intentional
        # for the physics layers.
        data.x_dict["bus"][:, PD_H] *= b
        data.x_dict["bus"][:, QD_H] *= b
        data.x_dict["bus"][:, QG_H] *= b
        data.x_dict["bus"][:, MIN_QG_H] *= b
        data.x_dict["bus"][:, MAX_QG_H] *= b
        data.x_dict["bus"][:, GS] *= b  # -> physical units (not original p.u.)
        data.x_dict["bus"][:, BS] *= b  # -> physical units (not original p.u.)
        data.x_dict["bus"][:, VN_KV] *= vn

        # -------- BUS LABEL INVERSE NORMALIZATION --------
        data.y_dict["bus"][:, PD_H] *= b
        data.y_dict["bus"][:, QD_H] *= b
        data.y_dict["bus"][:, QG_H] *= b

        # -------- GENERATOR INPUT INVERSE NORMALIZATION --------
        data.x_dict["gen"][:, PG_H] *= g
        data.x_dict["gen"][:, MIN_PG] *= g
        data.x_dict["gen"][:, MAX_PG] *= g
        data.x_dict["gen"][:, C0_H] = torch.sign(data.x_dict["gen"][:, C0_H]) * (
            torch.exp(torch.abs(data.x_dict["gen"][:, C0_H])) - 1
        )
        data.x_dict["gen"][:, C1_H] = torch.sign(data.x_dict["gen"][:, C1_H]) * (
            torch.exp(torch.abs(data.x_dict["gen"][:, C1_H])) - 1
        )
        data.x_dict["gen"][:, C2_H] = torch.sign(data.x_dict["gen"][:, C2_H]) * (
            torch.exp(torch.abs(data.x_dict["gen"][:, C2_H])) - 1
        )

        # -------- GENERATOR LABEL INVERSE NORMALIZATION --------
        data.y_dict["gen"][:, PG_H] *= g

        # -------- EDGE INPUT INVERSE NORMALIZATION --------
        data.edge_attr_dict[("bus", "connects", "bus")][:, P_E] *= e_b
        data.edge_attr_dict[("bus", "connects", "bus")][:, Q_E] *= e_b
        data.edge_attr_dict[("bus", "connects", "bus")][:, YFF_TT_R : YFT_TF_I + 1] *= (
            e_b.unsqueeze(1)
        )  # -> physical units (not original p.u.), see WARNING above
        data.edge_attr_dict[("bus", "connects", "bus")][:, ANG_MIN] *= 180.0 / torch.pi
        data.edge_attr_dict[("bus", "connects", "bus")][:, ANG_MAX] *= 180.0 / torch.pi

        data.edge_attr_dict[("bus", "connects", "bus")][:, RATE_A] *= e_b
        data.is_normalized = False

    def inverse_output(self, output, batch):
        """
        Denormalize model output (bus PG/QG, gen PG) using per-sample baseMVA from lookups.
        Fully GPU/CPU safe.
        """
        bus_output = output["bus"]
        gen_output = output["gen"]

        bus_batch = getattr(batch["bus"], "batch", None)

        # Move lookup tensor to correct device
        baseMVA_lookup = self._baseMVA_lookup.to(
            device=bus_output.device,
            dtype=bus_output.dtype,
        )

        if bus_batch is not None:
            # Batched: scenario_id per node via batch index; lookup base MVA per node
            sid_bus = batch["scenario_id"][bus_batch]
            sid_gen = batch["scenario_id"][batch["gen"].batch]
            b_bus = baseMVA_lookup[sid_bus]
            b_gen = baseMVA_lookup[sid_gen]
        else:
            # Single graph: one scenario_id; use its base MVA
            sid = batch["scenario_id"].item()
            b_bus = baseMVA_lookup[sid]
            b_gen = baseMVA_lookup[sid]

        # Scale per-unit power back to MW/Mvar
        bus_output[:, PG_OUT] *= b_bus
        bus_output[:, QG_OUT] *= b_bus
        gen_output[:, PG_OUT_GEN] *= b_gen

    def get_stats(self) -> dict:
        """Return dict of stats for saving (baseMVA_orig, scenarios, baseMVA, vn_kv_max tensors)."""
        return {
            "baseMVA_orig": torch.tensor(self.baseMVA_orig, dtype=torch.float),
            "scenarios": self._scenario_ids,
            "baseMVA": self._baseMVA_lookup[self._scenario_ids],
            "vn_kv_max": self._vn_kv_max_lookup[self._scenario_ids],
        }
