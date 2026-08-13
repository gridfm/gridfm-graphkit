# Copyright 2026 GridFM Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for Hive-partition streaming in HeteroGridDatasetDisk.

The central guarantee is that streaming changes *memory usage*, not *results*:
processing the same source data laid out flat vs. Hive-partitioned must produce
identical on-disk output. The remaining tests pin the mode contracts
(auto/on/off) and the partition-detection boundaries.
"""

import os
import os.path as osp
import shutil

import pandas as pd
import pytest
import torch

from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
from gridfm_graphkit.datasets.globals import VA_H, PG_H

_SRC_RAW = "tests/data/case14_ieee/raw"
_TABLES = ["bus_data.parquet", "gen_data.parquet", "branch_data.parquet"]
_N_SCENARIOS = 6  # subset of the 72-scenario fixture to keep the tests fast
_SCEN_PER_PARTITION = 3  # -> partitions {0, 1} for the subset above


def _subset(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["scenario"] < _N_SCENARIOS].copy()


def _write_flat(root: str) -> None:
    """Write the subset as single flat parquet files under ``root/raw``."""
    raw = osp.join(root, "raw")
    os.makedirs(raw, exist_ok=True)
    for table in _TABLES:
        _subset(pd.read_parquet(osp.join(_SRC_RAW, table))).to_parquet(
            osp.join(raw, table),
            index=False,
        )


def _write_partitioned(root: str) -> None:
    """Write the subset as Hive-partitioned parquet under ``root/raw``."""
    raw = osp.join(root, "raw")
    for table in _TABLES:
        df = _subset(pd.read_parquet(osp.join(_SRC_RAW, table)))
        # Drop any pre-existing partition column so pyarrow does not see it
        # both in the file and re-derived from the directory name.
        df = df.drop(columns=["scenario_partition"], errors="ignore")
        for scenario, group in df.groupby("scenario"):
            part = int(scenario) // _SCEN_PER_PARTITION
            part_dir = osp.join(raw, table, f"scenario_partition={part}")
            os.makedirs(part_dir, exist_ok=True)
            out = osp.join(part_dir, f"scenario_{int(scenario)}.parquet")
            group.to_parquet(out, index=False)


def _build(root: str, stream_partitions: str) -> HeteroGridDatasetDisk:
    # data_normalizer is only used at access time (get()), not during process().
    return HeteroGridDatasetDisk(
        root=root,
        data_normalizer=None,
        stream_partitions=stream_partitions,
    )


def _load_graphs(processed_dir: str) -> dict[str, dict]:
    graphs = {}
    for name in os.listdir(processed_dir):
        if name.startswith("data_index_") and name.endswith(".pt"):
            graphs[name] = torch.load(osp.join(processed_dir, name), weights_only=True)
    return graphs


def _assert_graphs_equal(a: dict[str, dict], b: dict[str, dict]) -> None:
    assert set(a) == set(b), "different set of scenario graphs produced"
    for name in a:
        _assert_value_equal(a[name], b[name], name)


def _assert_value_equal(va, vb, path: str) -> None:
    if isinstance(va, torch.Tensor):
        assert isinstance(vb, torch.Tensor), f"{path}: type mismatch"
        assert torch.equal(va, vb), f"{path}: tensor mismatch"
    elif isinstance(va, dict):
        assert isinstance(vb, dict), f"{path}: type mismatch"
        assert set(va) == set(vb), f"{path}: different keys"
        for key in va:
            _assert_value_equal(va[key], vb[key], f"{path}.{key}")
    else:
        assert va == vb, f"{path}: value mismatch"


def test_streaming_matches_flat(tmp_path):
    """Streaming and flat processing must yield identical scenario graphs."""
    flat_root = str(tmp_path / "flat")
    part_root = str(tmp_path / "partitioned")
    _write_flat(flat_root)
    _write_partitioned(part_root)

    _build(flat_root, "off")
    _build(part_root, "on")

    flat_graphs = _load_graphs(osp.join(flat_root, "processed"))
    part_graphs = _load_graphs(osp.join(part_root, "processed"))

    assert len(flat_graphs) == _N_SCENARIOS
    _assert_graphs_equal(flat_graphs, part_graphs)


def test_scenario_graph_structure(tmp_path):
    """Pin the absolute structure of a built graph against the case14 fixture.

    The equivalence test only proves flat == streaming; because both paths share
    ``_build_and_save_scenario``, a fault in that shared builder would corrupt
    both sides identically and still pass. These assertions check ground truth
    (shapes, slice boundaries, edge mirroring) so builder faults are caught.
    """
    root = str(tmp_path / "flat")
    _write_flat(root)
    _build(root, "off")
    g = torch.load(osp.join(root, "processed", "data_index_0.pt"), weights_only=True)

    bus, gen = g["bus"], g["gen"]
    bb = g[("bus", "connects", "bus")]
    n_bus, n_gen = 14, 5
    n_branch = 20

    # Node feature blocks and target slice boundaries (kills VA_H/PG_H mutants).
    assert bus["x"].shape == (n_bus, 15)
    assert gen["x"].shape == (n_gen, 7)
    assert bus["y"].shape == (n_bus, VA_H + 1)
    assert gen["y"].shape == (n_gen, PG_H + 1)
    assert torch.equal(bus["y"], bus["x"][:, : VA_H + 1])
    assert torch.equal(gen["y"], gen["x"][:, : PG_H + 1])

    # Directed edges = forward + reverse of every branch (kills cat/order mutants).
    ei = bb["edge_index"]
    assert ei.shape == (2, 2 * n_branch)
    assert bb["edge_attr"].shape == (2 * n_branch, 11)
    assert bb["y"].shape == (2 * n_branch, 2)
    fwd, rev = ei[:, :n_branch], ei[:, n_branch:]
    # Reverse half must mirror the forward half (kills from_bus/to_bus swaps).
    assert torch.equal(rev[0], fwd[1])
    assert torch.equal(rev[1], fwd[0])

    # Generator connectivity is symmetric between the two directed edge sets.
    g2b = g[("gen", "connected_to", "bus")]["edge_index"]
    b2g = g[("bus", "connected_to", "gen")]["edge_index"]
    assert g2b.shape == (2, n_gen)
    assert torch.equal(g2b[0], b2g[1])
    assert torch.equal(g2b[1], b2g[0])

    assert int(g["_global_store"]["scenario_id"].item()) == 0


def test_streaming_matches_flat_load_scenarios(tmp_path):
    """load_scenarios.pt must be identical between the two paths."""
    flat_root = str(tmp_path / "flat")
    part_root = str(tmp_path / "partitioned")
    _write_flat(flat_root)
    _write_partitioned(part_root)

    _build(flat_root, "off")
    _build(part_root, "on")

    flat_ls = torch.load(
        osp.join(flat_root, "processed", "load_scenarios.pt"),
        weights_only=True,
    )
    part_ls = torch.load(
        osp.join(part_root, "processed", "load_scenarios.pt"),
        weights_only=True,
    )
    assert torch.equal(flat_ls, part_ls)


def test_auto_uses_streaming_when_partitioned(tmp_path):
    """auto mode on partitioned data produces the full set of graphs."""
    root = str(tmp_path / "auto_part")
    _write_partitioned(root)
    ds = _build(root, "auto")
    assert ds._detect_partitions() == [0, 1]
    assert len(_load_graphs(osp.join(root, "processed"))) == _N_SCENARIOS


def test_auto_uses_legacy_when_flat(tmp_path):
    """auto mode on flat data detects no partitions and still processes."""
    root = str(tmp_path / "auto_flat")
    _write_flat(root)
    ds = _build(root, "auto")
    assert ds._detect_partitions() is None
    assert len(_load_graphs(osp.join(root, "processed"))) == _N_SCENARIOS


def test_on_without_partitions_raises(tmp_path):
    """stream_partitions='on' must fail loudly when data is flat."""
    root = str(tmp_path / "on_flat")
    _write_flat(root)
    with pytest.raises(RuntimeError, match="requires Hive-partitioned"):
        _build(root, "on")


def test_off_ignores_partitions(tmp_path):
    """stream_partitions='off' uses the legacy path even when partitioned."""
    root = str(tmp_path / "off_part")
    _write_partitioned(root)
    ds = _build(root, "off")
    # Partitions physically exist, but 'off' still builds every scenario graph.
    assert ds._detect_partitions() == [0, 1]
    assert len(_load_graphs(osp.join(root, "processed"))) == _N_SCENARIOS


def test_invalid_mode_raises(tmp_path):
    """An unknown stream_partitions value is rejected before any processing."""
    root = str(tmp_path / "bad")
    _write_flat(root)
    with pytest.raises(ValueError, match="stream_partitions"):
        _build(root, "sometimes")


@pytest.mark.parametrize("missing_table", _TABLES)
def test_detect_partitions_requires_all_tables(tmp_path, missing_table):
    """If any table is not partitioned, detection returns None (legacy path)."""
    root = str(tmp_path / "mixed")
    _write_partitioned(root)
    # Flatten a single table so the layout is inconsistent.
    raw = osp.join(root, "raw")
    df = _subset(pd.read_parquet(osp.join(_SRC_RAW, missing_table)))
    shutil.rmtree(osp.join(raw, missing_table))
    df.to_parquet(osp.join(raw, missing_table), index=False)

    ds = _build(root, "auto")
    assert ds._detect_partitions() is None


def test_scenario_spanning_two_partitions_raises(tmp_path):
    """A scenario whose rows are split across two partitions must raise ValueError.

    This guards the per-partition agg_gen Q-limit sum: if scenario X lives in
    both partition 0 and partition 1, the sum is computed twice on partial data
    and diverges from the legacy whole-dataset merge.  The guard must fire
    before any .pt file is written for the offending scenario.
    """
    root = str(tmp_path / "split_scenario")
    raw = osp.join(root, "raw")

    # Write a valid partitioned layout first, then inject a copy of scenario 0
    # into partition 1 so it spans two partitions.
    _write_partitioned(root)

    for table in _TABLES:
        part0_dir = osp.join(raw, table, "scenario_partition=0")
        part1_dir = osp.join(raw, table, "scenario_partition=1")
        src_file = next(
            f for f in os.listdir(part0_dir) if f.startswith("scenario_0.")
        )
        df = pd.read_parquet(osp.join(part0_dir, src_file))
        df.to_parquet(osp.join(part1_dir, "scenario_0_duplicate.parquet"), index=False)

    with pytest.raises(ValueError, match=r"Scenario 0 appears in both partition 0 and partition 1"):
        _build(root, "on")


def test_streaming_raises_on_noncontiguous_scenario_ids(tmp_path):
    """Streaming must raise when scenario ids have a gap (e.g. 0,1,2,4,5 — missing 3).

    The gap fixture is built by writing a valid partitioned layout and then
    deleting scenario 3's files from all tables.  This leaves ids {0,1,2,4,5}:
    min=0, max=5, count=5 → the contiguity check (max == count-1, i.e. 5==4) fails.
    """
    root = str(tmp_path / "gap")
    _write_partitioned(root)
    raw = osp.join(root, "raw")
    for table in _TABLES:
        gap_file = osp.join(raw, table, "scenario_partition=1", "scenario_3.parquet")
        if osp.exists(gap_file):
            os.remove(gap_file)

    with pytest.raises(ValueError, match=r"max=5, count=5"):
        _build(root, "on")


def test_streaming_accepts_contiguous_scenario_ids(tmp_path):
    """Streaming must not raise when scenario ids are a complete 0..N-1 range."""
    root = str(tmp_path / "contiguous")
    _write_partitioned(root)
    _build(root, "on")
    assert len(_load_graphs(osp.join(root, "processed"))) == _N_SCENARIOS
