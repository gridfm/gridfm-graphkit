"""Hetero preprocess regression tests.

Goals
-----
1. Extract did not change graph tensors vs pre-refactor ``process()``.
2. Processing two Hive partitions (modules + CLI script) matches a full-table process.
3. ``workers>1`` writes the same graphs as ``workers=1``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from gridfm_graphkit.datasets.globals import PG_H, VA_H
from gridfm_graphkit.datasets.hetero_preprocess import (
    build_hetero_data_for_scenario,
    build_load_scenarios,
    merge_agg_gen_into_bus,
    process_partition,
    process_scenarios,
    read_raw_tables,
)

CASE14_RAW = Path("tests/data/case14_ieee/raw")
N_PER_PART = 2
N_SCENARIOS = 2 * N_PER_PART  # scenarios 0..3 → partitions 0 and 1

# ---------------------------------------------------------------------------
# Frozen pre-5b6ffe5 feature lists + builder (oracle for goal 1).
# Intentionally NOT imported from hetero_preprocess.
# ---------------------------------------------------------------------------
_BUS = [
    "Pd", "Qd", "Qg", "Vm", "Va", "PQ", "PV", "REF",
    "min_vm_pu", "max_vm_pu", "min_q_mvar", "max_q_mvar", "GS", "BS", "vn_kv",
]
_GEN = [
    "p_mw", "min_p_mw", "max_p_mw", "cp0_eur", "cp1_eur_per_mw",
    "cp2_eur_per_mw2", "in_service",
]
_BR_COMMON = ["tap", "ang_min", "ang_max", "rate_a", "br_status"]
_BR_FWD = ["pf", "qf", "Yff_r", "Yff_i", "Yft_r", "Yft_i"] + _BR_COMMON
_BR_REV = ["pt", "qt", "Ytt_r", "Ytt_i", "Ytf_r", "Ytf_i"] + _BR_COMMON


def _legacy_build(scenario, bus_df, gen_df, branch_df) -> HeteroData:
    """Pre-extract HeteroData construction from powergrid_hetero_dataset.process()."""
    data = HeteroData()
    data["bus"].x = torch.tensor(bus_df[_BUS].values, dtype=torch.float)
    gen_df = gen_df.reset_index()
    data["gen"].x = torch.tensor(gen_df[_GEN].values, dtype=torch.float)
    gen_df["gen_index"] = gen_df.index
    data["bus"].y = data["bus"].x[:, : VA_H + 1].clone()
    data["gen"].y = data["gen"].x[:, : PG_H + 1].clone()

    fwd_e = torch.tensor(branch_df[["from_bus", "to_bus"]].values.T, dtype=torch.long)
    rev_e = torch.tensor(branch_df[["to_bus", "from_bus"]].values.T, dtype=torch.long)
    data["bus", "connects", "bus"].edge_index = torch.cat([fwd_e, rev_e], dim=1)
    data["bus", "connects", "bus"].edge_attr = torch.cat(
        [
            torch.tensor(branch_df[_BR_FWD].values, dtype=torch.float),
            torch.tensor(branch_df[_BR_REV].values, dtype=torch.float),
        ],
        dim=0,
    )
    data["bus", "connects", "bus"].y = torch.cat(
        [
            torch.tensor(branch_df[["pf", "qf"]].values, dtype=torch.float),
            torch.tensor(branch_df[["pt", "qt"]].values, dtype=torch.float),
        ],
        dim=0,
    )
    data["gen", "connected_to", "bus"].edge_index = torch.tensor(
        gen_df[["gen_index", "bus"]].values.T, dtype=torch.long,
    )
    data["bus", "connected_to", "gen"].edge_index = torch.tensor(
        gen_df[["bus", "gen_index"]].values.T, dtype=torch.long,
    )
    data["scenario_id"] = torch.tensor([scenario], dtype=torch.long)
    return data


def _assert_hetero_equal(a: HeteroData, b: HeteroData) -> None:
    da, db = a.to_dict(), b.to_dict()
    assert da.keys() == db.keys()
    for key in da:
        ta, tb = da[key], db[key]
        if torch.is_tensor(ta):
            assert torch.equal(ta, tb), key
        else:
            assert ta.keys() == tb.keys(), key
            for sub in ta:
                assert torch.equal(ta[sub], tb[sub]), f"{key}.{sub}"


def _slice_tables(n: int = N_SCENARIOS):
    bus, gen, branch = read_raw_tables(str(CASE14_RAW))
    keep = set(range(n))
    return (
        bus[bus["scenario"].isin(keep)].copy(),
        gen[gen["scenario"].isin(keep)].copy(),
        branch[branch["scenario"].isin(keep)].copy(),
    )


def _write_partition(raw_dir: Path, table: str, part: int, df: pd.DataFrame) -> None:
    # Hive dir encodes the partition; omit the column so Arrow does not merge
    # int64 (file) with dictionary (path) when reading all partitions at once.
    out = raw_dir / table / f"scenario_partition={part}"
    out.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["scenario_partition"], errors="ignore").to_parquet(
        out / "data.parquet", index=False,
    )


def _carve_two_partitions(raw_dir: Path) -> None:
    """Scenarios 0..3 → Hive partitions 0 (0..1) and 1 (2..3)."""
    bus, gen, branch = _slice_tables()
    for part in (0, 1):
        lo, hi = part * N_PER_PART, (part + 1) * N_PER_PART
        _write_partition(raw_dir, "bus_data.parquet", part, bus[bus["scenario"].between(lo, hi - 1)])
        _write_partition(raw_dir, "gen_data.parquet", part, gen[gen["scenario"].between(lo, hi - 1)])
        _write_partition(
            raw_dir, "branch_data.parquet", part, branch[branch["scenario"].between(lo, hi - 1)],
        )


def _assert_pt_dirs_equal(ref: Path, other: Path) -> None:
    refs = sorted(ref.glob("data_index_*.pt"))
    assert refs
    assert {p.name for p in refs} == {p.name for p in other.glob("data_index_*.pt")}
    for path in refs:
        a = HeteroData.from_dict(torch.load(path, weights_only=True))
        b = HeteroData.from_dict(torch.load(other / path.name, weights_only=True))
        _assert_hetero_equal(a, b)


def test_current_builder_matches_pre_extract_legacy():
    """Goal 1: same tensors as frozen pre-5b6ffe5 process() body."""
    bus, gen, branch = _slice_tables(n=2)
    # Same merge as old process(): sum gen Q limits onto bus.
    bus = bus.merge(
        gen.groupby(["scenario", "bus"])[["min_q_mvar", "max_q_mvar"]].sum().reset_index(),
        on=["scenario", "bus"],
        how="left",
    ).fillna(0)

    for s in (0, 1):
        legacy = _legacy_build(
            s, bus[bus["scenario"] == s], gen[gen["scenario"] == s], branch[branch["scenario"] == s],
        )
        current = build_hetero_data_for_scenario(
            s, bus[bus["scenario"] == s], gen[gen["scenario"] == s], branch[branch["scenario"] == s],
        )
        _assert_hetero_equal(legacy, current)


def test_two_partitions_match_full_table_and_script(tmp_path: Path):
    """Goal 2: process_partition + CLI == full-table process_scenarios."""
    raw = tmp_path / "raw"
    _carve_two_partitions(raw)

    # Reference: load all partitions at once (old IO shape).
    ref = tmp_path / "ref"
    ref.mkdir()
    bus, gen, branch = read_raw_tables(str(raw))
    load_ref = build_load_scenarios(bus)
    process_scenarios(
        merge_agg_gen_into_bus(bus, gen), gen, branch, str(ref),
        skip_existing=False, show_progress=False, workers=1,
    )

    # Modules: partition-by-partition.
    parts = tmp_path / "parts"
    parts.mkdir()
    prev, load_chunks = None, []
    for pid in (0, 1):
        prev, chunk = process_partition(
            str(raw), str(parts), pid,
            skip_existing=False, show_progress=False, workers=1,
            previous_scenario_max=prev,
        )
        load_chunks.append(chunk)
    _assert_pt_dirs_equal(ref, parts)
    assert torch.equal(load_ref, torch.tensor(np.concatenate(load_chunks)))

    # CLI script on a fresh root with the same carved raw layout.
    root = tmp_path / "script_root"
    _carve_two_partitions(root / "raw")
    subprocess.check_call(
        [sys.executable, "scripts/process_hetero_dataset_parallel.py", str(root), "--workers", "1"],
        cwd=Path(__file__).resolve().parents[1],
    )
    script_proc = root / "processed"
    _assert_pt_dirs_equal(ref, script_proc)
    assert (script_proc / "processed_raw_files.done").read_text() == "done"
    assert torch.equal(
        load_ref, torch.load(script_proc / "load_scenarios.pt", weights_only=True),
    )


def test_multi_worker_matches_single_worker(tmp_path: Path):
    """Goal 3: ThreadPool workers>1 must write the same graphs as workers=1."""
    bus, gen, branch = _slice_tables()
    bus = merge_agg_gen_into_bus(bus, gen)
    w1, w4 = tmp_path / "w1", tmp_path / "w4"
    w1.mkdir()
    w4.mkdir()
    process_scenarios(bus, gen, branch, str(w1), skip_existing=False, show_progress=False, workers=1)
    process_scenarios(bus, gen, branch, str(w4), skip_existing=False, show_progress=False, workers=4)
    _assert_pt_dirs_equal(w1, w4)
