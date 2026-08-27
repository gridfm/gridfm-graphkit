"""Shared power-grid graph construction.

This module holds the single source of truth for turning per-scenario
``bus`` / ``gen`` / ``branch`` tables into a PyG :class:`~torch_geometric.data.HeteroData`
graph. Both the on-disk dataset (:class:`gridfm_graphkit.datasets.powergrid_hetero_dataset.HeteroGridDatasetDisk`)
and the vLLM serving IO processor build graphs through here so the training and
inference input layouts can never drift apart.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from gridfm_graphkit.datasets.globals import PG_H, VA_H

# Column layouts consumed by GNS_heterogeneous. Order matters: these define the
# feature-index constants in gridfm_graphkit.datasets.globals.
BUS_FEATURES = [
    "Pd",
    "Qd",
    "Qg",
    "Vm",
    "Va",
    "PQ",
    "PV",
    "REF",
    "min_vm_pu",
    "max_vm_pu",
    "min_q_mvar",
    "max_q_mvar",
    "GS",
    "BS",
    "vn_kv",
]
GEN_FEATURES = [
    "p_mw",
    "min_p_mw",
    "max_p_mw",
    "cp0_eur",
    "cp1_eur_per_mw",
    "cp2_eur_per_mw2",
    "in_service",
]
COMMON_BRANCH_FEATURES = ["tap", "ang_min", "ang_max", "rate_a", "br_status"]
FORWARD_BRANCH_FEATURES = [
    "pf",
    "qf",
    "Yff_r",
    "Yff_i",
    "Yft_r",
    "Yft_i",
] + COMMON_BRANCH_FEATURES
REVERSE_BRANCH_FEATURES = [
    "pt",
    "qt",
    "Ytt_r",
    "Ytt_i",
    "Ytf_r",
    "Ytf_i",
] + COMMON_BRANCH_FEATURES


def build_hetero_data(
    bus_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    scenario: int = 0,
) -> HeteroData:
    """Build one scenario's heterogeneous power-grid graph.

    Args:
        bus_df: Bus table for a single scenario. Buses must be in increasing
            ``bus`` order (0..N-1), matching the datakit output contract.
        gen_df: Generator table for the same scenario.
        branch_df: Branch table for the same scenario.
        scenario: Scenario id stored on the graph (default 0 for single-graph
            inference).

    Returns:
        A :class:`~torch_geometric.data.HeteroData` with ``bus`` / ``gen`` node
        stores and ``("bus","connects","bus")``, ``("gen","connected_to","bus")``
        and ``("bus","connected_to","gen")`` edge stores.
    """
    assert (bus_df["bus"].values == torch.arange(len(bus_df)).numpy()).all(), (
        "Buses are not in increasing order"
    )

    data = HeteroData()

    # Bus nodes
    data["bus"].x = torch.tensor(bus_df[BUS_FEATURES].values, dtype=torch.float)

    # Generator nodes
    gen_df = gen_df.reset_index(drop=True)
    data["gen"].x = torch.tensor(gen_df[GEN_FEATURES].values, dtype=torch.float)
    gen_df["gen_index"] = gen_df.index

    data["bus"].y = data["bus"].x[:, : (VA_H + 1)].clone()
    data["gen"].y = data["gen"].x[:, : (PG_H + 1)].clone()

    # Bus-Bus edges (branches added in both directions)
    forward_edges = torch.tensor(
        branch_df[["from_bus", "to_bus"]].values.T,
        dtype=torch.long,
    )
    forward_edge_attr = torch.tensor(
        branch_df[FORWARD_BRANCH_FEATURES].values,
        dtype=torch.float,
    )
    reverse_edges = torch.tensor(
        branch_df[["to_bus", "from_bus"]].values.T,
        dtype=torch.long,
    )
    reverse_edge_attr = torch.tensor(
        branch_df[REVERSE_BRANCH_FEATURES].values,
        dtype=torch.float,
    )

    edge_index = torch.cat([forward_edges, reverse_edges], dim=1)
    edge_attr = torch.cat([forward_edge_attr, reverse_edge_attr], dim=0)

    forward_targets = torch.tensor(branch_df[["pf", "qf"]].values, dtype=torch.float)
    reverse_targets = torch.tensor(branch_df[["pt", "qt"]].values, dtype=torch.float)
    edge_y = torch.cat([forward_targets, reverse_targets], dim=0)

    data["bus", "connects", "bus"].edge_index = edge_index
    data["bus", "connects", "bus"].edge_attr = edge_attr
    data["bus", "connects", "bus"].y = edge_y

    # Gen-Bus and Bus-Gen edges
    data["gen", "connected_to", "bus"].edge_index = torch.tensor(
        gen_df[["gen_index", "bus"]].values.T,
        dtype=torch.long,
    )
    data["bus", "connected_to", "gen"].edge_index = torch.tensor(
        gen_df[["bus", "gen_index"]].values.T,
        dtype=torch.long,
    )

    data["scenario_id"] = torch.tensor([scenario], dtype=torch.long)

    return data
