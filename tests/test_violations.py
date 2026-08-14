import pytest
import torch
from gridfm_graphkit.datasets.globals import ANG_MIN, ANG_MAX
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
