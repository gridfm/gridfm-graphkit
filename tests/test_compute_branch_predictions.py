"""Tests for compute_branch_predictions in gridfm_graphkit.tasks.utils."""

import numpy as np
import pytest
import torch
import yaml
from torch_geometric.data import HeteroData

from gridfm_graphkit.datasets.globals import (
    ANG_MAX,
    ANG_MIN,
    P_E,
    Q_E,
    RATE_A,
    VM_H,
    VA_H,
    YFF_TT_I,
    YFF_TT_R,
    YFT_TF_I,
    YFT_TF_R,
)
from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer
from gridfm_graphkit.io.param_handler import NestedNamespace
from gridfm_graphkit.tasks.utils import compute_branch_predictions

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# edge_attr has 11 columns (indices 0–10); build a minimal valid tensor.
_NUM_EDGE_FEATURES = 11


def _make_edge_attr(
    num_edges: int,
    *,
    yff_r: float = 1.0,
    yff_i: float = 0.0,
    yft_r: float = -1.0,
    yft_i: float = 0.0,
    ang_min_deg: float = -30.0,
    ang_max_deg: float = 30.0,
    rate_a: float = 100.0,
) -> torch.Tensor:
    """Return a [num_edges, 11] edge_attr tensor with controlled values."""
    attr = torch.zeros(num_edges, _NUM_EDGE_FEATURES)
    attr[:, YFF_TT_R] = yff_r
    attr[:, YFF_TT_I] = yff_i
    attr[:, YFT_TF_R] = yft_r
    attr[:, YFT_TF_I] = yft_i
    attr[:, ANG_MIN] = ang_min_deg
    attr[:, ANG_MAX] = ang_max_deg
    attr[:, RATE_A] = rate_a
    return attr


def _make_inputs(
    num_bus: int = 3,
    num_edges: int = 4,
    *,
    vm: float = 1.0,
    va: float = 0.0,
    rate_a: float = 100.0,
    ang_min_deg: float = -30.0,
    ang_max_deg: float = 30.0,
):
    """Return (eval_bus, target, bus_edge_index, bus_edge_attr, scenario_ids, local_bus_idx)."""
    # bus state: [VM, VA, PG, QG]
    bus = torch.zeros(num_bus, 4)
    bus[:, 0] = vm   # VM_OUT = 0
    bus[:, 1] = va   # VA_OUT = 1

    # simple directed edges: 0->1, 1->2, 2->0, 1->0
    src = torch.tensor([0, 1, 2, 1])[:num_edges]
    dst = torch.tensor([1, 2, 0, 0])[:num_edges]
    edge_index = torch.stack([src, dst])

    edge_attr = _make_edge_attr(
        num_edges,
        rate_a=rate_a,
        ang_min_deg=ang_min_deg,
        ang_max_deg=ang_max_deg,
    )

    scenario_ids = torch.zeros(num_bus, dtype=torch.long)
    local_bus_idx = torch.arange(num_bus)

    return bus.clone(), bus.clone(), edge_index, edge_attr, scenario_ids, local_bus_idx


# ---------------------------------------------------------------------------
# Test 1 — output keys are exactly as expected
# ---------------------------------------------------------------------------

EXPECTED_KEYS = {
    "scenario", "from_bus", "to_bus",
    "Pft", "Qft", "Pft_target", "Qft_target",
    "angle_diff", "angle_excess_low", "angle_excess_high",
    "angle_diff_target", "angle_excess_low_target", "angle_excess_high_target",
    "thermal_excess", "thermal_excess_target",
    "rate_a", "Yff_r", "Yff_i", "Yft_r", "Yft_i",
}


def test_output_keys():
    result = compute_branch_predictions(*_make_inputs())
    assert set(result.keys()) == EXPECTED_KEYS


# ---------------------------------------------------------------------------
# Test 2 — all values are numpy arrays
# ---------------------------------------------------------------------------

def test_all_values_are_numpy():
    result = compute_branch_predictions(*_make_inputs())
    for key, val in result.items():
        assert isinstance(val, np.ndarray), f"{key!r} is {type(val)}, expected np.ndarray"


# ---------------------------------------------------------------------------
# Test 3 — all arrays have length == num_edges
# ---------------------------------------------------------------------------

def test_array_lengths():
    num_edges = 4
    result = compute_branch_predictions(*_make_inputs(num_edges=num_edges))
    for key, val in result.items():
        assert len(val) == num_edges, f"{key!r} has length {len(val)}, expected {num_edges}"


# ---------------------------------------------------------------------------
# Test 4 — thermal excess is non-negative everywhere
# ---------------------------------------------------------------------------

def test_thermal_excess_non_negative():
    result = compute_branch_predictions(*_make_inputs())
    assert (result["thermal_excess"] >= 0).all()
    assert (result["thermal_excess_target"] >= 0).all()


# ---------------------------------------------------------------------------
# Test 5 — angle excess is non-negative everywhere
# ---------------------------------------------------------------------------

def test_angle_excess_non_negative():
    result = compute_branch_predictions(*_make_inputs())
    assert (result["angle_excess_low"] >= 0).all()
    assert (result["angle_excess_high"] >= 0).all()
    assert (result["angle_excess_low_target"] >= 0).all()
    assert (result["angle_excess_high_target"] >= 0).all()


# ---------------------------------------------------------------------------
# Test 6 — angle_diff is wrapped to [-pi, pi]
# ---------------------------------------------------------------------------

def test_angle_diff_wrapped():
    result = compute_branch_predictions(*_make_inputs())
    assert (result["angle_diff"] >= -np.pi).all()
    assert (result["angle_diff"] <= np.pi).all()
    assert (result["angle_diff_target"] >= -np.pi).all()
    assert (result["angle_diff_target"] <= np.pi).all()


# ---------------------------------------------------------------------------
# Test 7 — when eval_bus == target, predicted fields equal ground-truth fields
# ---------------------------------------------------------------------------

def test_perfect_prediction_equals_target():
    args = _make_inputs()
    result = compute_branch_predictions(*args)
    np.testing.assert_array_equal(result["Pft"], result["Pft_target"])
    np.testing.assert_array_equal(result["Qft"], result["Qft_target"])
    np.testing.assert_array_equal(result["thermal_excess"], result["thermal_excess_target"])
    np.testing.assert_array_equal(result["angle_diff"], result["angle_diff_target"])
    np.testing.assert_array_equal(result["angle_excess_low"], result["angle_excess_low_target"])
    np.testing.assert_array_equal(result["angle_excess_high"], result["angle_excess_high_target"])


# ---------------------------------------------------------------------------
# Test 8 — thermal excess is zero when apparent flow is well below rate_a
# ---------------------------------------------------------------------------

def test_no_thermal_excess_when_within_limits():
    # VM=0.001 -> very small flows -> well below rate_a=100
    result = compute_branch_predictions(*_make_inputs(vm=0.001, rate_a=100.0))
    np.testing.assert_array_equal(result["thermal_excess"], 0.0)
    np.testing.assert_array_equal(result["thermal_excess_target"], 0.0)


# ---------------------------------------------------------------------------
# Test 9 — branch flows match stored P_E/Q_E on real case14 data
# ---------------------------------------------------------------------------

def test_branch_flows_match_stored_values():
    data_dict = torch.load(
        "tests/data/case14_ieee/processed/data_index_0.pt",
        weights_only=True,
    )
    data = HeteroData.from_dict(data_dict)

    node_stats = torch.load(
        "tests/data/case14_ieee/processed/data_stats_HeteroDataMVANormalizer.pt",
        weights_only=True,
    )
    with open("tests/config/datamodule_test_base_config.yaml", "r") as f:
        args = NestedNamespace(**yaml.safe_load(f))

    normalizer = HeteroDataMVANormalizer(args)
    normalizer.fit_from_dict(node_stats)
    normalizer.transform(data)

    bus_edge_index = data[("bus", "connects", "bus")].edge_index
    bus_edge_attr = data[("bus", "connects", "bus")].edge_attr
    num_bus = data["bus"].x.size(0)

    # Use ground-truth VM/VA as both eval_bus and target
    bus_state = torch.zeros(num_bus, 4)
    bus_state[:, 0] = data["bus"].x[:, VM_H]  # VM_OUT = 0
    bus_state[:, 1] = data["bus"].x[:, VA_H]  # VA_OUT = 1

    scenario_ids = torch.zeros(num_bus, dtype=torch.long)
    local_bus_idx = torch.arange(num_bus)

    result = compute_branch_predictions(
        bus_state,
        bus_state,
        bus_edge_index,
        bus_edge_attr,
        scenario_ids,
        local_bus_idx,
    )

    stored_P = bus_edge_attr[:, P_E].numpy()
    stored_Q = bus_edge_attr[:, Q_E].numpy()

    np.testing.assert_allclose(result["Pft"], stored_P, atol=1e-4)
    np.testing.assert_allclose(result["Qft"], stored_Q, atol=1e-4)
