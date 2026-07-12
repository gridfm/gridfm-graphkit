import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

from gridfm_graphkit.datasets.globals import PG_H, VA_H

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


def get_partition_ids(raw_dir: str) -> Optional[list[int]]:
    bus_path = osp.join(raw_dir, "bus_data.parquet")
    if not osp.isdir(bus_path):
        return None
    partition_names = [
        name
        for name in os.listdir(bus_path)
        if name.startswith("scenario_partition=")
    ]
    if not partition_names:
        return None
    return sorted(int(name.split("=")[1]) for name in partition_names)


def read_raw_tables(raw_dir: str, partition_id: Optional[int] = None):
    if partition_id is None:
        bus_data = pd.read_parquet(osp.join(raw_dir, "bus_data.parquet"))
        gen_data = pd.read_parquet(osp.join(raw_dir, "gen_data.parquet"))
        branch_data = pd.read_parquet(osp.join(raw_dir, "branch_data.parquet"))
        return bus_data, gen_data, branch_data

    partition_suffix = f"/scenario_partition={partition_id}"
    bus_data = pd.read_parquet(
        osp.join(raw_dir, "bus_data.parquet") + partition_suffix,
    )
    gen_data = pd.read_parquet(
        osp.join(raw_dir, "gen_data.parquet") + partition_suffix,
    )
    branch_data = pd.read_parquet(
        osp.join(raw_dir, "branch_data.parquet") + partition_suffix,
    )
    return bus_data, gen_data, branch_data


def assert_scenario_index(bus_data: pd.DataFrame) -> None:
    """Scenarios are exactly 0 .. N-1 (full dataset)."""
    assert (
        bus_data["scenario"].min() == 0
        and bus_data["scenario"].max() == len(bus_data["scenario"].unique()) - 1
    )


def assert_partition_scenarios_contiguous(bus_data: pd.DataFrame) -> None:
    """Scenarios within one parquet partition form a contiguous block."""
    scenario_min = int(bus_data["scenario"].min())
    scenario_max = int(bus_data["scenario"].max())
    scenario_count = int(bus_data["scenario"].nunique())
    assert scenario_max == scenario_min + scenario_count - 1


def merge_agg_gen_into_bus(
    bus_data: pd.DataFrame,
    gen_data: pd.DataFrame,
) -> pd.DataFrame:
    agg_gen = (
        gen_data.groupby(["scenario", "bus"])[["min_q_mvar", "max_q_mvar"]]
        .sum()
        .reset_index()
    )
    return bus_data.merge(agg_gen, on=["scenario", "bus"], how="left").fillna(0)


def build_load_scenarios(bus_data: pd.DataFrame) -> Optional[torch.Tensor]:
    if "load_scenario_idx" not in bus_data.columns:
        return None
    return torch.tensor(
        bus_data.groupby("scenario", sort=True)["load_scenario_idx"].first().values,
    )


def build_load_scenarios_from_partitions(
    raw_dir: str,
    partition_ids: Iterable[int],
    *,
    show_progress: bool = False,
) -> Optional[torch.Tensor]:
    partition_ids = sorted(partition_ids)
    chunks = []
    has_load_scenario_idx = None
    iterator = (
        tqdm(partition_ids, desc="Building load_scenarios", unit="partition")
        if show_progress
        else partition_ids
    )
    for partition_id in iterator:
        bus_data = pd.read_parquet(
            osp.join(raw_dir, "bus_data.parquet") + f"/scenario_partition={partition_id}",
            columns=["scenario", "load_scenario_idx"],
        )
        if has_load_scenario_idx is None:
            has_load_scenario_idx = "load_scenario_idx" in bus_data.columns
        if not has_load_scenario_idx:
            return None
        chunks.append(
            bus_data.groupby("scenario", sort=True)["load_scenario_idx"].first().values,
        )
    return torch.tensor(np.concatenate(chunks))


def validate_partition_scenarios(
    raw_dir: str,
    partition_ids: list[int],
    *,
    show_progress: bool = False,
) -> int:
    previous_max = None
    total_scenarios = 0
    iterator = (
        tqdm(partition_ids, desc="Validating partitions", unit="partition")
        if show_progress
        else partition_ids
    )
    for partition_id in iterator:
        bus_data = pd.read_parquet(
            osp.join(raw_dir, "bus_data.parquet") + f"/scenario_partition={partition_id}",
            columns=["scenario"],
        )
        scenario_min = int(bus_data["scenario"].min())
        scenario_max = int(bus_data["scenario"].max())
        scenario_count = int(bus_data["scenario"].nunique())
        if partition_id == partition_ids[0]:
            assert scenario_min == 0
        if previous_max is not None:
            assert scenario_min == previous_max + 1
        assert scenario_max == scenario_min + scenario_count - 1
        previous_max = scenario_max
        total_scenarios += scenario_count
    assert previous_max == total_scenarios - 1
    return total_scenarios


def build_hetero_data_for_scenario(
    scenario: int,
    bus_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    branch_df: pd.DataFrame,
) -> HeteroData:
    assert (bus_df["bus"].values == torch.arange(len(bus_df))).all(), (
        "Buses are not in increasing order"
    )

    data = HeteroData()
    data["bus"].x = torch.tensor(bus_df[BUS_FEATURES].values, dtype=torch.float)

    gen_df = gen_df.reset_index()
    data["gen"].x = torch.tensor(gen_df[GEN_FEATURES].values, dtype=torch.float)
    gen_df["gen_index"] = gen_df.index

    data["bus"].y = data["bus"].x[:, : (VA_H + 1)].clone()
    data["gen"].y = data["gen"].x[:, : (PG_H + 1)].clone()

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

    forward_targets = torch.tensor(
        branch_df[["pf", "qf"]].values,
        dtype=torch.float,
    )
    reverse_targets = torch.tensor(
        branch_df[["pt", "qt"]].values,
        dtype=torch.float,
    )
    edge_y = torch.cat([forward_targets, reverse_targets], dim=0)

    data["bus", "connects", "bus"].edge_index = edge_index
    data["bus", "connects", "bus"].edge_attr = edge_attr
    data["bus", "connects", "bus"].y = edge_y

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


def _save_scenario(
    scenario: int,
    bus_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    processed_dir: str,
) -> None:
    data = build_hetero_data_for_scenario(scenario, bus_df, gen_df, branch_df)
    torch.save(
        data.to_dict(),
        osp.join(processed_dir, f"data_index_{scenario}.pt"),
    )


def process_scenarios(
    bus_data: pd.DataFrame,
    gen_data: pd.DataFrame,
    branch_data: pd.DataFrame,
    processed_dir: str,
    *,
    skip_existing: bool = True,
    show_progress: bool = True,
    progress_desc: str = "Processing scenarios",
    workers: int = 1,
) -> None:
    bus_groups = bus_data.groupby("scenario")
    gen_groups = gen_data.groupby("scenario")
    branch_groups = branch_data.groupby("scenario")

    tasks = []
    for scenario in bus_data["scenario"].unique():
        if skip_existing and osp.exists(
            osp.join(processed_dir, f"data_index_{scenario}.pt"),
        ):
            continue
        if (
            scenario not in gen_groups.groups
            or scenario not in branch_groups.groups
        ):
            raise ValueError(f"Missing gen/branch data for scenario {scenario}")

        tasks.append(
            (
                scenario,
                bus_groups.get_group(scenario),
                gen_groups.get_group(scenario),
                branch_groups.get_group(scenario),
                processed_dir,
            ),
        )

    if not tasks:
        return

    if workers <= 1:
        iterator = tqdm(tasks, desc=progress_desc) if show_progress else tasks
        for task in iterator:
            _save_scenario(*task)
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(_save_scenario, *zip(*tasks))
        if show_progress:
            results = tqdm(results, total=len(tasks), desc=progress_desc)
        for _ in results:
            pass


def process_partition(
    raw_dir: str,
    processed_dir: str,
    partition_id: int,
    *,
    skip_existing: bool = True,
    show_progress: bool = True,
    workers: int = 1,
    previous_scenario_max: Optional[int] = None,
) -> tuple[int, Optional[np.ndarray]]:
    bus_data, gen_data, branch_data = read_raw_tables(raw_dir, partition_id)
    assert_partition_scenarios_contiguous(bus_data)

    scenario_min = int(bus_data["scenario"].min())
    scenario_max = int(bus_data["scenario"].max())
    if previous_scenario_max is None:
        assert scenario_min == 0
    else:
        assert scenario_min == previous_scenario_max + 1

    load_chunk = None
    if "load_scenario_idx" in bus_data.columns:
        load_chunk = bus_data.groupby("scenario", sort=True)["load_scenario_idx"].first().values

    bus_data = merge_agg_gen_into_bus(bus_data, gen_data)
    process_scenarios(
        bus_data,
        gen_data,
        branch_data,
        processed_dir,
        skip_existing=skip_existing,
        show_progress=show_progress,
        progress_desc=f"partition {partition_id}",
        workers=workers,
    )
    return scenario_max, load_chunk


def validate_raw_partition_layout(raw_dir: str) -> list[int]:
    """Return sorted partition ids; bus/gen/branch must share the same layout."""
    bus_partition_ids = get_partition_ids(raw_dir)
    if bus_partition_ids is None:
        raise ValueError(f"No parquet partitions found under {raw_dir}")

    for table_name in ("gen_data.parquet", "branch_data.parquet"):
        table_path = osp.join(raw_dir, table_name)
        table_partition_ids = sorted(
            int(name.split("=")[1])
            for name in os.listdir(table_path)
            if name.startswith("scenario_partition=")
        )
        if table_partition_ids != bus_partition_ids:
            raise ValueError(
                f"Partition mismatch for {table_name}: "
                f"bus has {len(bus_partition_ids)} partitions, "
                f"{table_name} has {len(table_partition_ids)}",
            )

    return bus_partition_ids
