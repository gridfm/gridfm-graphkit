# Copyright 2026 Massimiliano Brighindi
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MATPOWER's zero-rating sentinel across evaluation paths."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from gridfm_graphkit.datasets.globals import (
    ANG_MAX,
    ANG_MIN,
    C0_H,
    RATE_A,
    VM_H,
    YFF_TT_R,
)
from gridfm_graphkit.tasks.opf_ac_dc_baseline import _compute_branch_violations
from gridfm_graphkit.tasks.opf_task import OptimalPowerFlowTask
from gridfm_graphkit.tasks.utils import compute_branch_predictions

# A zero rating disables the model constraint. Missing values must stay missing.
_CASES = [
    (0.0, 0.0),
    (-0.0, 0.0),
    (5.0, 5.0),
    (10.0, 0.0),
    (20.0, 0.0),
    (np.inf, 0.0),
    (np.nan, np.nan),
]


def _assert_excess(actual, expected):
    if np.isnan(expected):
        assert np.isnan(actual)
    else:
        assert actual == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("rating,expected", _CASES)
def test_baseline_zero_rating_is_unbounded(rating, expected):
    """Preserve the sentinel and finite/missing limit semantics in AC and DC."""
    branch = pd.DataFrame(
        {
            "scenario": [0],
            "from_bus": [0],
            "to_bus": [1],
            "rate_a": [rating],
            "ang_min": [-360.0],
            "ang_max": [360.0],
            "pf": [10.0],
            "qf": [0.0],
            "pt": [-10.0],
            "qt": [0.0],
            "pf_dc_computed": [10.0],
            "qf_dc_computed": [0.0],
            "pt_dc_computed": [-10.0],
            "qt_dc_computed": [0.0],
        },
    )
    bus = pd.DataFrame(
        {"scenario": [0, 0], "bus": [0, 1], "Va": [0.0, 0.0], "Va_dc": [0.0, 0.0]},
    )
    before = branch.copy(deep=True)
    metrics = _compute_branch_violations(branch, bus)
    for prefix in ("AC", "DC"):
        for end in ("from", "to"):
            _assert_excess(
                metrics[f"{prefix} Mean branch thermal violation {end} (MVA)"],
                expected,
            )
    pd.testing.assert_frame_equal(branch, before)


def _inputs(rating, dtype):
    bus = torch.zeros((3, 4), dtype=dtype)
    bus[:, 0] = 1.0
    edges = torch.tensor([[0, 1, 1, 2], [1, 2, 0, 1]])
    attributes = torch.zeros((4, 11), dtype=dtype)
    attributes[:, YFF_TT_R] = 10.0
    attributes[:, RATE_A] = rating
    attributes[:, ANG_MIN] = -360.0
    attributes[:, ANG_MAX] = 360.0
    return bus, edges, attributes


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rating,expected", _CASES)
def test_branch_predictions_zero_rating_is_unbounded(rating, expected, dtype):
    """Apply sentinel semantics to predictions and labels, keeping raw exports."""
    bus, edges, attributes = _inputs(rating, dtype)
    result = compute_branch_predictions(
        bus,
        bus.clone(),
        edges,
        attributes,
        torch.zeros(3, dtype=torch.long),
        torch.arange(3),
    )
    for name in ("thermal_excess", "thermal_excess_target"):
        for value in result[name]:
            _assert_excess(value, expected)
    np.testing.assert_array_equal(result["rate_a"], np.full(4, rating))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rating,expected", _CASES)
def test_opf_test_step_zero_rating_is_unbounded(rating, expected, dtype):
    """Exercise the full native test_step using controlled model outputs."""
    bus, edges, attributes = _inputs(rating, dtype)
    x_bus = torch.zeros((3, 15), dtype=dtype)
    x_gen = torch.zeros((2, 7), dtype=dtype)
    x_gen[:, C0_H] = 1.0
    y_bus = torch.zeros((3, 5), dtype=dtype)
    y_bus[:, VM_H] = 1.0
    y_gen = torch.ones((2, 1), dtype=dtype)
    batch = SimpleNamespace(
        x_dict={"bus": x_bus, "gen": x_gen},
        y_dict={"bus": y_bus, "gen": y_gen},
        edge_index_dict={
            ("bus", "connects", "bus"): edges,
            ("gen", "connected_to", "bus"): torch.tensor([[0, 1], [0, 1]]),
        },
        edge_attr_dict={("bus", "connects", "bus"): attributes},
        batch_dict={
            "bus": torch.zeros(3, dtype=torch.long),
            "gen": torch.zeros(2, dtype=torch.long),
        },
        mask_dict={
            "PQ": torch.tensor([False, False, True]),
            "PV": torch.tensor([False, True, False]),
            "REF": torch.tensor([True, False, False]),
        },
        num_graphs=1,
    )
    metrics = {}
    task = SimpleNamespace(
        args=SimpleNamespace(data=SimpleNamespace(networks=["fixture"]), verbose=False),
        data_normalizers=[
            SimpleNamespace(
                inverse_transform=lambda batch: None,
                inverse_output=lambda output, batch: None,
            ),
        ],
        shared_step=lambda batch: (
            {"bus": bus.clone(), "gen": y_gen.clone()},
            {"loss": torch.tensor(0.0)},
        ),
        log=lambda name, value, **kwargs: metrics.update({name: value}),
    )
    OptimalPowerFlowTask.test_step(task, batch, 0)
    for end in ("from", "to"):
        _assert_excess(
            metrics[f"fixture/Branch thermal violation {end}"].item(),
            expected,
        )
