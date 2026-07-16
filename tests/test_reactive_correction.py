import torch
import pytest
from torch_geometric.data import HeteroData

from gridfm_graphkit.datasets.hetero_preprocess import (
    reconcile_reactive_balance,
    REACTIVE_CORRECTION_MODES,
)
from gridfm_graphkit.datasets.globals import (
    QD_H,
    QG_H,
    VM_H,
    VA_H,
    PV_H,
    REF_H,
    VM_OUT,
    VA_OUT,
    QG_OUT,
)
from gridfm_graphkit.models.utils import (
    ComputeBranchFlow,
    ComputeNodeInjection,
    ComputeNodeResiduals,
)

PROCESSED = "tests/data/case14_ieee/processed/data_index_0.pt"


def _load_scenario() -> HeteroData:
    # Processed files are stored pre-normalization: physical units, angles in degrees,
    # which is exactly the regime reconcile_reactive_balance operates in.
    return HeteroData.from_dict(torch.load(PROCESSED, weights_only=True))


def _residual_Q(data: HeteroData) -> torch.Tensor:
    """Per-bus reactive residual via the same layer chain used by the training loss."""
    bus = data["bus"].x
    ei = data["bus", "connects", "bus"].edge_index
    ea = data["bus", "connects", "bus"].edge_attr
    n = bus.size(0)
    out = torch.zeros((n, 4), dtype=bus.dtype)
    out[:, VM_OUT] = bus[:, VM_H]
    out[:, VA_OUT] = bus[:, VA_H] * torch.pi / 180.0  # deg -> rad for the physics
    out[:, QG_OUT] = bus[:, QG_H]
    Pft, Qft = ComputeBranchFlow()(out, ei, ea)
    _, Q_in = ComputeNodeInjection()(Pft, Qft, ei, n)
    _, residual_Q = ComputeNodeResiduals()(torch.zeros(n), Q_in, out, bus)
    return residual_Q


@pytest.mark.parametrize("mode", REACTIVE_CORRECTION_MODES)
def test_reconcile_drives_reactive_residual_to_zero(mode):
    data = _load_scenario()

    # Corrupt Qd with a large offset to create a "crazy big" reactive imbalance.
    torch.manual_seed(0)
    corruption = torch.randn(data["bus"].x.size(0)) * 200.0
    data["bus"].x[:, QD_H] = data["bus"].x[:, QD_H] + corruption
    assert _residual_Q(data).abs().max() > 1.0  # imbalance is real and large

    vm_before = data["bus"].x[:, VM_H].clone()
    va_before = data["bus"].x[:, VA_H].clone()

    reconcile_reactive_balance(data, mode=mode)

    # Residual is driven to ~0 on every bus.
    assert _residual_Q(data).abs().max() < 1e-2

    # Voltages and angles are untouched.
    assert torch.equal(data["bus"].x[:, VM_H], vm_before)
    assert torch.equal(data["bus"].x[:, VA_H], va_before)


def test_qd_all_absorbs_only_into_qd():
    data = _load_scenario()
    qg_before = data["bus"].x[:, QG_H].clone()

    torch.manual_seed(1)
    data["bus"].x[:, QD_H] += torch.randn(data["bus"].x.size(0)) * 200.0

    reconcile_reactive_balance(data, mode="qd_all")

    # Mode 'qd_all' never touches Qg.
    assert torch.equal(data["bus"].x[:, QG_H], qg_before)


def test_qd_pq_qg_pvref_routes_by_bus_type():
    data = _load_scenario()
    bus = data["bus"].x
    pv_ref = (bus[:, PV_H] > 0.5) | (bus[:, REF_H] > 0.5)
    assert pv_ref.any() and (~pv_ref).any()  # case14 has both

    qd_before = bus[:, QD_H].clone()
    qg_before = bus[:, QG_H].clone()

    torch.manual_seed(2)
    bus[:, QD_H] += torch.randn(bus.size(0)) * 200.0
    qd_corrupted = bus[:, QD_H].clone()

    reconcile_reactive_balance(data, mode="qd_pq_qg_pvref")

    # PV/REF buses: Qd unchanged (from corrupted value), correction went into Qg.
    assert torch.allclose(bus[pv_ref, QD_H], qd_corrupted[pv_ref])
    assert not torch.allclose(bus[pv_ref, QG_H], qg_before[pv_ref])
    # PQ buses: Qg unchanged, correction went into Qd.
    assert torch.equal(bus[~pv_ref, QG_H], qg_before[~pv_ref])


def test_y_columns_synced_with_corrected_x():
    data = _load_scenario()
    torch.manual_seed(3)
    data["bus"].x[:, QD_H] += torch.randn(data["bus"].x.size(0)) * 200.0

    reconcile_reactive_balance(data, mode="qd_all")

    # Targets (y) must match the corrected inputs for the load/gen columns.
    assert torch.equal(data["bus"].y[:, QD_H], data["bus"].x[:, QD_H])
    assert torch.equal(data["bus"].y[:, QG_H], data["bus"].x[:, QG_H])


def test_unknown_mode_raises():
    data = _load_scenario()
    with pytest.raises(ValueError):
        reconcile_reactive_balance(data, mode="nonsense")
