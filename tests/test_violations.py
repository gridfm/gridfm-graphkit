import numpy as np
import pandas as pd
import pytest
import torch
from gridfm_graphkit.datasets.globals import ANG_MIN, ANG_MAX
from gridfm_graphkit.tasks.opf_ac_dc_baseline import _compute_branch_violations
from gridfm_graphkit.tasks.utils import compute_angle_violation


def test_angle_violation_zero_within_limits():
    """Angle differences within limits produce zero violation."""
    # limits: [-30°, 30°] on both edges
    # angle_diff on edge 0: 0.1 rad ≈ 5.7° — well within [-30°, 30°]
    # angle_diff on edge 1: 0.0 rad — exactly at centre
    edge_attr = torch.zeros(2, 9)
    edge_attr[:, ANG_MIN] = -30.0
    edge_attr[:, ANG_MAX] = 30.0

    bus_angles = torch.tensor([0.1, 0.0, 0.0])  # radians
    edge_index = torch.tensor([[0, 1], [1, 2]])  # shape (2, 2)

    violation = compute_angle_violation(
        bus_edge_attr=edge_attr,
        bus_angles=bus_angles,
        bus_edge_index=edge_index,
        ang_min_col=ANG_MIN,
        ang_max_col=ANG_MAX,
    )

    assert violation == pytest.approx(0.0), (
        f"Expected zero violation within limits, got {violation}"
    )


def test_angle_violation_positive_outside_limits():
    """Regression test: ANG_MIN/ANG_MAX are in degrees and must be converted to
    radians before comparison with bus voltage angles (which are in radians).

    Without the ``* torch.pi / 180.0`` conversion in ``compute_angle_violation``,
    a raw degree value of 5.0 would be compared against an angle_diff of 0.1 rad,
    producing zero violation (false negative). With the conversion, the limit becomes
    ~0.0873 rad and the excess of ~0.0127 rad is correctly detected.
    """
    # limits: [-5°, 5°] → [-0.08727 rad, 0.08727 rad]
    # angle_diff on edge 0: 0.1 rad — just exceeds the 5° max
    # angle_diff on edge 1: 0.0 rad — within limits, contributes zero
    edge_attr = torch.zeros(2, 9)
    edge_attr[:, ANG_MIN] = -5.0
    edge_attr[:, ANG_MAX] = 5.0

    bus_angles = torch.tensor([0.1, 0.0, 0.0])  # radians
    edge_index = torch.tensor([[0, 1], [1, 2]])  # shape (2, 2)

    violation = compute_angle_violation(
        bus_edge_attr=edge_attr,
        bus_angles=bus_angles,
        bus_edge_index=edge_index,
        ang_min_col=ANG_MIN,
        ang_max_col=ANG_MAX,
    )

    assert violation > 0.0, (
        "Expected positive violation when angle_diff exceeds degree-converted limit"
    )

    # Exact regression value:
    #   edge 0: excess = 0.1 - (5 * pi / 180) ≈ 0.01273 rad
    #   edge 1: excess = 0.0 (within limits)
    #   mean over 2 edges = 0.01273 / 2 ≈ 0.00637 rad
    expected = (0.1 - 5.0 * torch.pi / 180.0) / 2.0
    assert violation == pytest.approx(expected, abs=1e-6), (
        f"Violation value mismatch: got {violation:.8f}, "
        f"expected {expected:.8f}. "
        "Check that ANG_MIN/ANG_MAX are being converted from degrees to radians."
    )


def _make_branch_violation_inputs(
    va_from_deg,
    va_to_deg,
    ang_min_deg,
    ang_max_deg,
    va_dc_from_deg=None,
    va_dc_to_deg=None,
):
    """Build minimal branch_df and bus_df for _compute_branch_violations."""
    n = len(va_from_deg)
    if va_dc_from_deg is None:
        va_dc_from_deg = va_from_deg
    if va_dc_to_deg is None:
        va_dc_to_deg = va_to_deg

    branch_df = pd.DataFrame(
        {
            "scenario": [0] * n,
            "from_bus": list(range(n)),
            "to_bus": [i + n for i in range(n)],
            "pf": [0.0] * n,
            "qf": [0.0] * n,
            "pt": [0.0] * n,
            "qt": [0.0] * n,
            "pf_dc_computed": [0.0] * n,
            "pt_dc_computed": [0.0] * n,
            "qf_dc_computed": [0.0] * n,
            "qt_dc_computed": [0.0] * n,
            "rate_a": [999.0] * n,  # large — no thermal violation
            "ang_min": ang_min_deg,
            "ang_max": ang_max_deg,
        },
    )

    from_buses = pd.DataFrame(
        {
            "scenario": [0] * n,
            "bus": list(range(n)),
            "Va": va_from_deg,
            "Va_dc": va_dc_from_deg,
        },
    )
    to_buses = pd.DataFrame(
        {
            "scenario": [0] * n,
            "bus": [i + n for i in range(n)],
            "Va": va_to_deg,
            "Va_dc": va_dc_to_deg,
        },
    )
    bus_df = pd.concat([from_buses, to_buses], ignore_index=True)
    return branch_df, bus_df


def test_baseline_angle_violation_zero_within_limits():
    """Angles within limits produce zero violation in the AC/DC baseline."""
    # angle_diff = 0.1 rad ≈ 5.73°, limits [-30°, 30°] → no violation
    branch_df, bus_df = _make_branch_violation_inputs(
        va_from_deg=[5.729577951],  # 0.1 rad expressed in degrees
        va_to_deg=[0.0],
        ang_min_deg=[-30.0],
        ang_max_deg=[30.0],
    )
    result = _compute_branch_violations(branch_df, bus_df)
    assert result[
        "AC Mean branch angle difference violation (radians)"
    ] == pytest.approx(0.0, abs=1e-6)
    assert result[
        "DC Mean branch angle difference violation (radians)"
    ] == pytest.approx(0.0, abs=1e-6)


def test_baseline_angle_violation_regression_degrees_vs_radians():
    """Regression: without degrees→radians conversion a 5° limit compared against
    a 0.1 rad diff (~5.73°) would produce zero violation (false negative).
    With the fix the excess is correctly detected and matches compute_angle_violation.
    """
    # angle_diff = 0.1 rad ≈ 5.73°, limit = 5° (0.08727 rad) → excess ≈ 0.01273 rad
    branch_df, bus_df = _make_branch_violation_inputs(
        va_from_deg=[5.729577951],  # 0.1 rad expressed in degrees
        va_to_deg=[0.0],
        ang_min_deg=[-5.0],
        ang_max_deg=[5.0],
    )
    result = _compute_branch_violations(branch_df, bus_df)
    expected = 0.1 - 5.0 * np.pi / 180.0  # one branch, mean == the single value

    assert result[
        "AC Mean branch angle difference violation (radians)"
    ] == pytest.approx(expected, abs=1e-6), (
        "AC: check that ang_min/ang_max are converted from degrees to radians"
    )
    assert result[
        "DC Mean branch angle difference violation (radians)"
    ] == pytest.approx(expected, abs=1e-6), (
        "DC: check that ang_min/ang_max are converted from degrees to radians"
    )
