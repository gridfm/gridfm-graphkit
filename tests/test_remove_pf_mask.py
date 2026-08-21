"""Tests that the .static attribute is correctly saved at dataset build time
and that RemovePFMask restores the zeroed limit columns after inverse_transform.
"""

import copy

import pytest
import torch
import yaml
from torch_geometric.data import HeteroData

from gridfm_graphkit.datasets.globals import (
    ANG_MAX,
    ANG_MIN,
    MAX_QG_H,
    MAX_VM_H,
    MIN_QG_H,
    MIN_VM_H,
    RATE_A,
    VN_KV,
)
from gridfm_graphkit.datasets.masking import AddPFHeteroMask, RemovePFMask
from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer
from gridfm_graphkit.datasets.transforms import ApplyMasking
from gridfm_graphkit.io.param_handler import NestedNamespace

_DATA_PATH = "tests/data/case14_ieee/processed/data_index_0.pt"
_STATS_PATH = "tests/data/case14_ieee/processed/data_stats_HeteroDataMVANormalizer.pt"
_CONFIG_PATH = "tests/config/datamodule_test_base_config.yaml"

BUS_LIMIT_COLS = [MIN_VM_H, MAX_VM_H, MIN_QG_H, MAX_QG_H, VN_KV]
BRANCH_LIMIT_COLS = [ANG_MIN, ANG_MAX, RATE_A]


@pytest.fixture(scope="module")
def raw_data():
    """Load a single processed scenario, un-normalised."""
    data_dict = torch.load(_DATA_PATH, weights_only=True)
    return HeteroData.from_dict(data_dict)


@pytest.fixture(scope="module")
def args():
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return NestedNamespace(**cfg)


@pytest.fixture(scope="module")
def normalizer(args):
    stats = torch.load(_STATS_PATH, weights_only=True)
    n = HeteroDataMVANormalizer(args)
    n.fit_from_dict(stats)
    return n


def test_static_attribute_saved_at_build_time(raw_data):
    """Dataset build saves .static with the raw limit values before masking."""
    assert hasattr(raw_data["bus"], "static"), (
        "data['bus'].static not found — was the dataset reprocessed after the fix?"
    )
    assert hasattr(raw_data[("bus", "connects", "bus")], "static"), (
        "data[('bus','connects','bus')].static not found — was the dataset reprocessed?"
    )

    bus_static = raw_data["bus"].static
    branch_static = raw_data[("bus", "connects", "bus")].static

    # static must match the corresponding columns of x / edge_attr exactly
    assert torch.equal(
        bus_static,
        raw_data["bus"].x[:, BUS_LIMIT_COLS],
    ), "bus.static does not match bus.x limit columns"
    assert torch.equal(
        branch_static,
        raw_data[("bus", "connects", "bus")].edge_attr[:, BRANCH_LIMIT_COLS],
    ), "edge static does not match edge_attr limit columns"

    # sanity: limits should not all be zero in real data
    assert bus_static.abs().sum() > 0, (
        "bus.static is all zeros — raw data may be missing limit values"
    )
    assert branch_static.abs().sum() > 0, (
        "branch.static is all zeros — raw data may be missing limit values"
    )


def test_remove_pf_mask_restores_limits(raw_data, args, normalizer):
    """RemovePFMask restores correct limit values after the full transform pipeline."""
    data = copy.deepcopy(raw_data)

    # Replicate exactly what happens at inference time.
    # Call .forward() directly (not __call__) to avoid BaseTransform's
    # internal copy.copy() which would discard mask_dict between steps.
    # 1. normalise
    normalizer.transform(data)
    # 2. add PF mask (marks limit cols)
    AddPFHeteroMask().forward(data)
    # 3. apply masking (zeros limit cols)
    ApplyMasking(args).forward(data)

    # At this point limits should be zero
    assert data.x_dict["bus"][:, BUS_LIMIT_COLS].abs().sum() == 0, (
        "Expected bus limit columns to be zero after ApplyMasking"
    )
    assert (
        data.edge_attr_dict[("bus", "connects", "bus")][:, BRANCH_LIMIT_COLS]
        .abs()
        .sum()
        == 0
    ), "Expected branch limit columns to be zero after ApplyMasking"

    # 4. inverse transform (zeros × baseMVA = still zeros — the original bug)
    normalizer.inverse_transform(data)

    assert data.x_dict["bus"][:, BUS_LIMIT_COLS].abs().sum() == 0, (
        "Sanity: bus limits should still be zero before RemovePFMask"
    )

    # 5. RemovePFMask restores them
    RemovePFMask()(data)

    restored_bus = data.x_dict["bus"][:, BUS_LIMIT_COLS]
    restored_branch = data.edge_attr_dict[("bus", "connects", "bus")][
        :,
        BRANCH_LIMIT_COLS,
    ]

    assert restored_bus.abs().sum() > 0, (
        "Bus limit columns are still zero after RemovePFMask"
    )
    assert restored_branch.abs().sum() > 0, (
        "Branch limit columns are still zero after RemovePFMask"
    )

    # Values must match the original raw limits stored in .static
    assert torch.allclose(restored_bus, raw_data["bus"].static, atol=1e-5), (
        "Restored bus limits do not match the original raw values in .static"
    )
    assert torch.allclose(
        restored_branch,
        raw_data[("bus", "connects", "bus")].static,
        atol=1e-5,
    ), "Restored branch limits do not match the original raw values in .static"
