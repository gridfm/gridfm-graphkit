import torch
from torch_geometric.data import HeteroData
from gridfm_graphkit.datasets.masking import AddPFHeteroMask
from gridfm_graphkit.datasets.task_transforms import PowerFlowTransforms
from gridfm_graphkit.io.param_handler import NestedNamespace
from gridfm_graphkit.datasets.globals import QG_H, VA_H, VM_H, PG_H


def test_pf_mask_bus_types():
    """PF knowns: PQ -> (PD, QD), PV -> (PG, VM), REF -> (VM, VA)."""
    data_dict = torch.load(
        "tests/data/case14_ieee/processed/data_index_0.pt",
        weights_only=True,
    )
    data = AddPFHeteroMask()(HeteroData.from_dict(data_dict))
    mask = data.mask_dict
    bus = mask["bus"]
    pq, pv, ref = mask["PQ"], mask["PV"], mask["REF"]
    assert pq.any() and pv.any() and ref.any()

    assert bus[pq][:, [VM_H, VA_H]].all()

    assert not bus[pv][:, VM_H].any()
    assert bus[pv][:, [VA_H, QG_H]].all()

    assert not bus[ref][:, [VM_H, VA_H]].any()
    assert bus[ref][:, QG_H].all()

    # Only generators on the slack bus have PG unknown.
    gen_idx, bus_idx = data.edge_index_dict[("gen", "connected_to", "bus")]
    expected_pg = torch.zeros_like(mask["gen"][:, PG_H])
    expected_pg[gen_idx] = ref[bus_idx]
    assert torch.equal(mask["gen"][:, PG_H], expected_pg)


def test_pf_mask_ref_vm_legacy_flag():
    """`data.mask_ref_vm: true` restores the old mask for older checkpoints."""
    data_dict = torch.load(
        "tests/data/case14_ieee/processed/data_index_0.pt",
        weights_only=True,
    )
    data = HeteroData.from_dict(data_dict)
    args = NestedNamespace(data={"mask_ref_vm": True, "mask_value": 0.0})
    mask = PowerFlowTransforms(args).transforms[2](data).mask_dict
    assert mask["bus"][mask["REF"]][:, VM_H].all()
    assert not mask["bus"][mask["REF"]][:, VA_H].any()
