"""Tests for the effective-resistance positional encoding used by HH-MPNN."""

import copy

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch_geometric.transforms import Compose

from gridfm_graphkit.datasets import effective_resistance
from gridfm_graphkit.datasets.effective_resistance import (
    BRANCH,
    ER_PE_ATTR,
    EffectiveResistancePE,
    connected_components,
    effective_resistance_pe,
)
from gridfm_graphkit.datasets.globals import B_ON, YFT_TF_I
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.datasets.transforms import RemoveInactiveBranches
from gridfm_graphkit.io.param_handler import NestedNamespace, get_task_transforms

CONFIG_PATH = "tests/config/hh_mpnn_test_config.yaml"
RAW_BRANCHES = "tests/data/case14_ieee/raw/branch_data.parquet"


def _undirected(edges, weights):
    """Edge index and weights with every edge listed in both directions."""
    edge_index = torch.tensor(edges, dtype=torch.long).T
    weight = torch.tensor(weights, dtype=torch.float)
    return torch.cat([edge_index, edge_index.flip(0)], dim=1), torch.cat([weight] * 2)


def _reference_pe(edges, weights, num_nodes):
    """Encoding as computed by the original HH-MPNN training script (numpy)."""
    laplacian = np.zeros((num_nodes, num_nodes))
    for (i, j), w in zip(edges, weights):
        laplacian[i, j] -= w
        laplacian[j, i] -= w
        laplacian[i, i] += w
        laplacian[j, j] += w

    keep = np.arange(num_nodes - 1)
    pinv = np.zeros((num_nodes, num_nodes))
    pinv[np.ix_(keep, keep)] = np.linalg.inv(laplacian[np.ix_(keep, keep)])
    centering = np.eye(num_nodes) - np.ones((num_nodes, num_nodes)) / num_nodes
    pinv = centering @ pinv @ centering
    diag = np.diag(pinv)
    resistance = diag[:, None] + diag[None, :] - 2 * pinv

    stats = np.zeros((num_nodes, 5))
    for i in range(num_nodes):
        row = np.delete(resistance[i], i)
        stats[i] = [row.mean(), np.median(row), row.std(), row.max(), row.min()]
    return torch.tensor(stats, dtype=torch.float)


def _random_connected_graph(num_nodes, seed):
    """A random spanning tree plus extra edges, with random positive weights."""
    rng = np.random.default_rng(seed)
    edges = [(int(rng.integers(0, i)), i) for i in range(1, num_nodes)]
    for _ in range(num_nodes):
        i, j = rng.choice(num_nodes, size=2, replace=False)
        if (min(i, j), max(i, j)) not in edges:
            edges.append((int(min(i, j)), int(max(i, j))))
    weights = rng.uniform(0.5, 50.0, size=len(edges)).tolist()
    return edges, weights


# --------------------------------------------------------------------------- #
# effective_resistance_pe
# --------------------------------------------------------------------------- #
def test_path_graph_closed_form():
    """On a unit path 0-1-2, resistances add up along the path."""
    edge_index, weight = _undirected([(0, 1), (1, 2)], [1.0, 1.0])
    pe = effective_resistance_pe(edge_index, weight, 3)
    expected = torch.tensor(
        [
            [1.5, 1.5, 0.5, 2.0, 1.0],  # distances 1 and 2
            [1.0, 1.0, 0.0, 1.0, 1.0],  # distances 1 and 1
            [1.5, 1.5, 0.5, 2.0, 1.0],
        ],
    )
    torch.testing.assert_close(pe, expected, atol=1e-6, rtol=0)


def test_parallel_branches_add():
    """Two parallel unit branches behave like one branch of weight two."""
    edge_index, weight = _undirected([(0, 1), (0, 1)], [1.0, 1.0])
    pe = effective_resistance_pe(edge_index, weight, 2)
    torch.testing.assert_close(pe[:, 0], torch.tensor([0.5, 0.5]))


@pytest.mark.parametrize("num_nodes", [5, 6, 30, 57])
def test_matches_original_implementation(num_nodes):
    """Agrees with the original script, for odd and even neighbour counts."""
    edges, weights = _random_connected_graph(num_nodes, seed=num_nodes)
    edge_index, weight = _undirected(edges, weights)
    pe = effective_resistance_pe(edge_index, weight, num_nodes)
    torch.testing.assert_close(
        pe,
        _reference_pe(edges, weights, num_nodes),
        atol=1e-5,
        rtol=1e-4,
    )


def test_zero_weight_edges_are_ignored():
    edge_index, weight = _undirected([(0, 1), (1, 2), (0, 2)], [1.0, 1.0, 0.0])
    path_index, path_weight = _undirected([(0, 1), (1, 2)], [1.0, 1.0])
    torch.testing.assert_close(
        effective_resistance_pe(edge_index, weight, 3),
        effective_resistance_pe(path_index, path_weight, 3),
    )


def test_disconnected_graph_uses_each_component():
    """Islands are encoded separately and an isolated bus gets zeros."""
    # Component {0, 1, 2} is a unit path, {3, 4} a single edge, 5 is isolated.
    edge_index, weight = _undirected([(0, 1), (1, 2), (3, 4)], [1.0, 1.0, 4.0])
    pe = effective_resistance_pe(edge_index, weight, 6)

    path_index, path_weight = _undirected([(0, 1), (1, 2)], [1.0, 1.0])
    torch.testing.assert_close(
        pe[:3],
        effective_resistance_pe(path_index, path_weight, 3),
    )
    torch.testing.assert_close(
        pe[3:5],
        torch.tensor([[0.25, 0.25, 0.0, 0.25, 0.25]] * 2),
    )
    torch.testing.assert_close(pe[5], torch.zeros(5))
    assert torch.isfinite(pe).all()


def test_connected_components():
    edge_index = torch.tensor([[0, 1, 3], [1, 2, 4]])
    labels = connected_components(edge_index, 6)
    assert labels.tolist() == [0, 0, 0, 3, 3, 5]


# --------------------------------------------------------------------------- #
# EffectiveResistancePE on the case14 test data
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def datamodule():
    with open(CONFIG_PATH) as f:
        args = NestedNamespace(**yaml.safe_load(f))
    dm = LitGridHeteroDataModule(args, data_dir="tests/data")

    class DummyTrainer:
        is_global_zero = True
        logger = None

    dm.trainer = DummyTrainer()
    dm.setup("fit")
    return dm


@pytest.fixture(scope="module")
def raw_branches():
    return pd.read_parquet(RAW_BRANCHES)


def _in_service(raw_branches, scenario):
    branches = raw_branches[raw_branches["scenario"] == scenario]
    return branches[branches["br_status"] == 1]


def _topology(raw_branches, scenario):
    branches = _in_service(raw_branches, scenario)
    return tuple(zip(branches["from_bus"], branches["to_bus"]))


def _without_task_transforms(dataset, idx):
    """A normalised sample as stored on disk, before any transform."""
    dataset = copy.copy(dataset)
    dataset.transform = None
    return dataset[idx]


def test_encoding_matches_raw_branch_parameters(datamodule, raw_branches):
    """The encoding equals the one built from x / (r^2 + x^2) of live branches."""
    dataset = datamodule.datasets[0]
    normalizer = datamodule.data_normalizers[0]
    # The normalizer moves admittances from the case base to its own base.
    scale = normalizer.baseMVA_orig / normalizer.baseMVA

    for idx in range(len(dataset)):
        data = dataset[idx]
        branches = _in_service(raw_branches, int(data["scenario_id"]))
        edges = list(zip(branches["from_bus"], branches["to_bus"]))
        susceptance = branches["x"] / (branches["r"] ** 2 + branches["x"] ** 2)
        edge_index, weight = _undirected(edges, (susceptance * scale).tolist())
        expected = effective_resistance_pe(edge_index, weight, data["bus"].num_nodes)
        torch.testing.assert_close(
            data["bus"][ER_PE_ATTR],
            expected,
            atol=1e-5,
            rtol=1e-4,
        )


def test_one_computation_per_distinct_topology(datamodule, raw_branches, monkeypatch):
    """Scenarios sharing a topology reuse one cached encoding."""
    dataset = datamodule.datasets[0]
    num_topologies = len(
        {_topology(raw_branches, s) for s in raw_branches["scenario"].unique()},
    )
    assert 1 < num_topologies < len(dataset)  # the fixture has outages

    calls = []
    original = effective_resistance.effective_resistance_pe

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(effective_resistance, "effective_resistance_pe", counting)
    transform = EffectiveResistancePE()
    for idx in range(len(dataset)):
        transform(_without_task_transforms(dataset, idx))

    assert len(calls) == num_topologies


def test_encoding_follows_topology(datamodule, raw_branches):
    """Equal topologies give equal encodings; an outage changes the encoding."""
    dataset = datamodule.datasets[0]
    by_topology = {}
    for idx in range(len(dataset)):
        data = dataset[idx]
        topology = _topology(raw_branches, int(data["scenario_id"]))
        by_topology.setdefault(topology, []).append(data["bus"][ER_PE_ATTR])

    encodings = []
    for same_topology in by_topology.values():
        for pe in same_topology[1:]:
            torch.testing.assert_close(pe, same_topology[0], atol=0, rtol=0)
        encodings.append(same_topology[0])
    for i in range(len(encodings)):
        for j in range(i + 1, len(encodings)):
            assert not torch.allclose(encodings[i], encodings[j])


def test_out_of_service_branches_are_excluded_before_their_removal(datamodule):
    """The status column is honoured when RemoveInactiveBranches has not run."""
    dataset = datamodule.datasets[0]
    data = _without_task_transforms(dataset, 0)
    original = EffectiveResistancePE()(data)["bus"][ER_PE_ATTR]

    # Switch one live branch off in both directions, keeping its admittance.
    edge_attr = data[BRANCH].edge_attr
    half = edge_attr.size(0) // 2
    branch = int((edge_attr[:half, B_ON] == 1).nonzero()[0])
    switched = data.clone()
    switched[BRANCH].edge_attr[[branch, branch + half], B_ON] = 0
    assert (switched[BRANCH].edge_attr[branch, YFT_TF_I] != 0).all()

    before = EffectiveResistancePE()(switched.clone())
    after = Compose([RemoveInactiveBranches(), EffectiveResistancePE()])(switched)
    torch.testing.assert_close(
        before["bus"][ER_PE_ATTR],
        after["bus"][ER_PE_ATTR],
        atol=0,
        rtol=0,
    )
    assert not torch.allclose(before["bus"][ER_PE_ATTR], original)


def test_disk_cache_is_shared_between_instances(datamodule, tmp_path, monkeypatch):
    dataset = datamodule.datasets[0]
    samples = [_without_task_transforms(dataset, idx) for idx in range(20)]

    first = EffectiveResistancePE(cache_dir=str(tmp_path))
    expected = [first(s)["bus"][ER_PE_ATTR] for s in samples]
    cache_files = list(tmp_path.glob("pe_cache_*.pt"))
    assert len(cache_files) == len(first._memory_cache)
    assert not list(tmp_path.glob(".pe_cache_tmp_*"))

    def fail(*args, **kwargs):
        raise AssertionError("encoding recomputed despite the disk cache")

    monkeypatch.setattr(effective_resistance, "effective_resistance_pe", fail)
    second = EffectiveResistancePE(cache_dir=str(tmp_path))
    for sample, pe in zip(samples, expected):
        torch.testing.assert_close(second(sample)["bus"][ER_PE_ATTR], pe)


def test_memory_cache_is_bounded(datamodule):
    dataset = datamodule.datasets[0]
    transform = EffectiveResistancePE(memory_cache_size=2)
    for idx in range(len(dataset)):
        transform(_without_task_transforms(dataset, idx))
    assert len(transform._memory_cache) == 2


def test_cached_encoding_cannot_be_modified_through_a_sample(datamodule):
    dataset = datamodule.datasets[0]
    transform = EffectiveResistancePE()
    first = transform(_without_task_transforms(dataset, 0))
    expected = first["bus"][ER_PE_ATTR].clone()
    first["bus"][ER_PE_ATTR].zero_()
    again = transform(_without_task_transforms(dataset, 0))
    torch.testing.assert_close(again["bus"][ER_PE_ATTR], expected)


# --------------------------------------------------------------------------- #
# Use from a config
# --------------------------------------------------------------------------- #
def _args_with_transforms(transforms):
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["transforms"] = transforms
    return NestedNamespace(**cfg)


def test_config_passes_cache_dir(tmp_path):
    args = _args_with_transforms(
        [{"name": "EffectiveResistancePE", "cache_dir": str(tmp_path / "pe")}],
    )
    (transform,) = get_task_transforms(args).transforms
    assert isinstance(transform, EffectiveResistancePE)
    assert transform.cache_dir == str(tmp_path / "pe")
    assert (tmp_path / "pe").is_dir()


def test_config_with_unknown_argument_is_rejected():
    args = _args_with_transforms(
        [{"name": "EffectiveResistancePE", "cache_directory": "somewhere"}],
    )
    with pytest.raises(ValueError, match="cache_directory"):
        get_task_transforms(args)
