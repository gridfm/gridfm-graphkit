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
            osp.join(raw, table), index=False
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
            graphs[name] = torch.load(
                osp.join(processed_dir, name), weights_only=True
            )
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


def test_streaming_matches_flat_load_scenarios(tmp_path):
    """load_scenarios.pt must be identical between the two paths."""
    flat_root = str(tmp_path / "flat")
    part_root = str(tmp_path / "partitioned")
    _write_flat(flat_root)
    _write_partitioned(part_root)

    _build(flat_root, "off")
    _build(part_root, "on")

    flat_ls = torch.load(
        osp.join(flat_root, "processed", "load_scenarios.pt"), weights_only=True
    )
    part_ls = torch.load(
        osp.join(part_root, "processed", "load_scenarios.pt"), weights_only=True
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
