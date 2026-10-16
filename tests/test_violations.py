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


# ---------------------------------------------------------------------------
# Fix #2 – edge_is_forward flag survives batching and splits thermal violations
# ---------------------------------------------------------------------------


def test_edge_is_forward_splits_correctly_in_batch():
    """edge_is_forward must correctly identify forward/reverse edges after batching.

    Regression for issue #136 bug #2: the old ``[:half_edges]`` split cut the
    batch between the first and second *half of the graphs*, not between forward
    and reverse directions, when batch_size > 1.
    """
    from torch_geometric.data import HeteroData
    from torch_geometric.data.batch import Batch

    def _make_graph(n_branches):
        data = HeteroData()
        # bus nodes (not used in this test, just required for a valid HeteroData)
        data["bus"].x = torch.zeros(n_branches + 1, 1)
        # edges: n_branches forward + n_branches reverse
        fwd = torch.stack(
            [torch.arange(n_branches), torch.arange(1, n_branches + 1)],
            dim=0,
        )
        rev = torch.stack(
            [torch.arange(1, n_branches + 1), torch.arange(n_branches)],
            dim=0,
        )
        data["bus", "connects", "bus"].edge_index = torch.cat([fwd, rev], dim=1)
        data["bus", "connects", "bus"].edge_attr = torch.zeros(2 * n_branches, 11)
        data["bus", "connects", "bus"].edge_is_forward = torch.cat(
            [torch.ones(n_branches), torch.zeros(n_branches)],
        ).bool()
        return data

    # Two graphs with *different* branch counts to expose the old half-split bug.
    g1 = _make_graph(3)  # 6 edges total
    g2 = _make_graph(5)  # 10 edges total

    batch = Batch.from_data_list([g1, g2])
    edge_is_forward = batch["bus", "connects", "bus"].edge_is_forward

    # Total edges: 16; forward edges should be exactly 8 (3 from g1, 5 from g2).
    assert edge_is_forward.sum().item() == 8, (
        "Expected 8 forward edges (3 + 5) after batching two graphs."
    )
    assert (~edge_is_forward).sum().item() == 8, (
        "Expected 8 reverse edges after batching two graphs."
    )

    # The old half-split would give indices [:8] and [8:].  Because PyG interleaves
    # graphs as [g1_fwd, g1_rev, g2_fwd, g2_rev], the true layout is:
    #   [fwd0, fwd1, fwd2, rev0, rev1, rev2, fwd0, fwd1, fwd2, fwd3, fwd4,
    #    rev0, rev1, rev2, rev3, rev4]
    # Half-split [:8] picks g1_fwd + g1_rev + g2_fwd[:2] — mixed directions.
    half_split_forward = edge_is_forward[:8]
    assert not half_split_forward.all(), (
        "Half-split incorrectly classifies all edges as forward — "
        "batching interleaves graphs, not directions."
    )


# ---------------------------------------------------------------------------
# Fix #4 – QgViolationPenalty is extensive and restricted to PV/REF buses
# ---------------------------------------------------------------------------


def test_qg_violation_penalty_is_extensive():
    """QgViolationPenalty loss must scale with violation count, not be constant.

    Regression for issue #136 bug #4: the old code called .mean() over the
    *violating* buses only, so adding more violations didn't change the loss.
    """
    from gridfm_graphkit.training.loss import QgViolationPenaltyLoss
    from gridfm_graphkit.datasets.globals import QG_OUT, MAX_QG_H, MIN_QG_H

    loss_fn = QgViolationPenaltyLoss(loss_args={}, args=None)

    N = 6  # 3 PV + 3 REF
    mask_PV = torch.tensor([True, True, True, False, False, False])
    mask_REF = torch.tensor([False, False, False, True, True, True])

    x_dict = {"bus": torch.zeros(N, 12)}
    x_dict["bus"][:, MAX_QG_H] = 1.0
    x_dict["bus"][:, MIN_QG_H] = -1.0

    # One violator: pred=2.0 on bus 0, rest within limits.
    pred_one = {"bus": torch.zeros(N, 4)}
    pred_one["bus"][0, QG_OUT] = 2.0

    # All six violating: pred=2.0 everywhere.
    pred_all = {"bus": torch.full((N, 4), 2.0)}

    mask = {"PV": mask_PV, "REF": mask_REF}
    loss_one = loss_fn(pred_one, None, mask=mask, x_dict=x_dict)["loss"]
    loss_all = loss_fn(pred_all, None, mask=mask, x_dict=x_dict)["loss"]

    assert loss_all > loss_one, (
        "QgViolationPenalty must be larger when more buses violate; "
        "old .mean()-over-violators made it constant."
    )


def test_qg_violation_penalty_excludes_pq_buses():
    """QgViolationPenalty must not penalise PQ buses (Qg is forced to 0 there)."""
    from gridfm_graphkit.training.loss import QgViolationPenaltyLoss
    from gridfm_graphkit.datasets.globals import MAX_QG_H, MIN_QG_H

    loss_fn = QgViolationPenaltyLoss(loss_args={}, args=None)

    N = 3
    # All PQ buses — none are PV or REF.
    mask = {
        "PV": torch.zeros(N, dtype=torch.bool),
        "REF": torch.zeros(N, dtype=torch.bool),
    }
    x_dict = {"bus": torch.zeros(N, 12)}
    x_dict["bus"][:, MAX_QG_H] = 0.0
    x_dict["bus"][:, MIN_QG_H] = 0.0
    pred = {"bus": torch.full((N, 4), 5.0)}  # massive violation on PQ buses

    loss = loss_fn(pred, None, mask=mask, x_dict=x_dict)["loss"]
    assert loss.item() == pytest.approx(0.0), (
        "QgViolationPenalty should be zero when all buses are PQ."
    )


# ---------------------------------------------------------------------------
# Fix #6 – compute_angle_violation with edge_is_forward uses forward edges only
# ---------------------------------------------------------------------------


def test_angle_violation_forward_only_ignores_reverse():
    """When edge_is_forward is supplied, only forward edges are evaluated.

    Regression for issue #136 bug #6: on reverse edges the angle difference has
    the opposite sign, so asymmetric limits give wrong violations unless only
    forward edges are used.

    Setup: two directed edges sharing buses 0 and 1.
      - Forward:  0 → 1  (angle_diff =  0.1 rad, limit [−5°, 5°] → excess ≈ 0.01273 rad)
      - Reverse:  1 → 0  (angle_diff = −0.1 rad)
    Asymmetric limits: [−2°, 5°] so the reverse edge violates the lower bound
    (−0.1 rad < −2° = −0.0349 rad) whereas the forward edge only violates the
    upper bound.  Without the forward-only filter both violations are averaged and
    the result is inflated.
    """
    edge_attr = torch.zeros(2, 9)
    # Asymmetric: lower limit much tighter than upper
    edge_attr[:, ANG_MIN] = -2.0  # -2° = -0.03491 rad
    edge_attr[:, ANG_MAX] = 5.0  # 5° =  0.08727 rad

    bus_angles = torch.tensor([0.1, 0.0])  # radians
    edge_index = torch.tensor([[0, 1], [1, 0]])  # fwd: 0→1, rev: 1→0
    edge_is_forward = torch.tensor([True, False])

    violation_forward_only = compute_angle_violation(
        bus_edge_attr=edge_attr,
        bus_angles=bus_angles,
        bus_edge_index=edge_index,
        ang_min_col=ANG_MIN,
        ang_max_col=ANG_MAX,
        edge_is_forward=edge_is_forward,
    )
    # Forward edge only: angle_diff = 0.1 rad, exceeds ang_max = 5° = 0.08727 rad
    expected_fwd = 0.1 - 5.0 * torch.pi / 180.0  # ≈ 0.01273 rad
    assert violation_forward_only == pytest.approx(expected_fwd, abs=1e-6), (
        "Forward-only violation should match the single forward-edge excess."
    )

    # Without the filter: reverse edge has angle_diff = −0.1 rad which also
    # violates ang_min = −2° = −0.03491 rad (excess ≈ 0.06509 rad).
    # Mean over two edges = (0.01273 + 0.06509) / 2 ≈ 0.03891, which is larger.
    violation_both = compute_angle_violation(
        bus_edge_attr=edge_attr,
        bus_angles=bus_angles,
        bus_edge_index=edge_index,
        ang_min_col=ANG_MIN,
        ang_max_col=ANG_MAX,
    )
    expected_both = (
        (0.1 - 5.0 * torch.pi / 180.0) + (0.1 - 2.0 * torch.pi / 180.0)
    ) / 2.0
    assert violation_both == pytest.approx(expected_both, abs=1e-6), (
        "All-edges violation should include both forward and reverse excesses."
    )
    assert violation_forward_only != pytest.approx(violation_both.item(), abs=1e-4), (
        "Forward-only and all-edges results must differ with asymmetric limits."
    )
