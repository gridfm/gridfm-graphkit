from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

from matplotlib.collections import PathCollection

from gridfm_graphkit.datasets.globals import (
    PG_H,
    PG_OUT_GEN,
    PQ_H,
    PV_H,
    REF_H,
    VM_H,
)
from gridfm_graphkit.utils import visualization


def _data():
    bus_x = torch.zeros((3, 15))
    bus_x[0, PQ_H] = 1
    bus_x[1, PV_H] = 1
    bus_x[2, REF_H] = 1

    bus_y = torch.zeros((3, 15))
    bus_y[:, VM_H] = torch.tensor([0.9, 1.0, 1.1])
    gen_y = torch.zeros((3, 7))
    gen_y[:, PG_H] = torch.tensor([1.0, 2.0, 3.0])

    bus_mask = torch.zeros_like(bus_y, dtype=torch.bool)
    bus_mask[:, VM_H] = torch.tensor([True, False, True])
    gen_mask = torch.zeros_like(gen_y, dtype=torch.bool)
    gen_mask[:, PG_H] = torch.tensor([True, True, False])

    return SimpleNamespace(
        x_dict={"bus": bus_x, "gen": torch.zeros((3, 7))},
        y_dict={"bus": bus_y, "gen": gen_y},
        edge_index_dict={
            ("bus", "connects", "bus"): torch.tensor(
                [[0, 1, 1, 2, 0], [1, 0, 2, 1, 0]],
            ),
            ("gen", "connected_to", "bus"): torch.tensor(
                [[0, 1, 2], [0, 0, 1]],
            ),
        },
        edge_attr_dict={("bus", "connects", "bus"): torch.zeros((5, 11))},
        mask_dict={"bus": bus_mask, "gen": gen_mask},
    )


class _ScalingNormalizer:
    def inverse_transform(self, data):
        data.y_dict["bus"][:, :3] *= 10
        data.y_dict["gen"][:, PG_H] *= 10

    def inverse_output(self, output, _data):
        output["bus"][:, 2:] *= 10
        output["gen"][:, PG_OUT_GEN] *= 10


def _collection_values(ax):
    return np.concatenate(
        [
            collection.get_array()
            for collection in ax.collections
            if isinstance(collection, PathCollection)
            and collection.get_array() is not None
        ],
    )


def test_visualize_error_uses_heterogeneous_pbe_inputs_and_returns_figure(
    monkeypatch,
):
    data = _data()
    calls = {}

    class StubPBELoss:
        def __init__(self, loss_args, args):
            calls["visualization"] = loss_args.visualization
            calls["args"] = args

        def __call__(
            self,
            output,
            target,
            edge_index,
            edge_attr,
            mask,
            x_dict=None,
        ):
            calls["inputs"] = (output, target, edge_index, edge_attr, mask, x_dict)
            return {
                "Nodal Active Power Loss in p.u.": torch.tensor([0.01, 0.02, 0.03]),
            }

    monkeypatch.setattr(visualization, "PBELoss", StubPBELoss)
    monkeypatch.setattr(
        visualization.plt,
        "show",
        lambda: pytest.fail("visualization should return figures without showing"),
    )
    output = {"bus": torch.zeros((3, 6)), "gen": torch.zeros((3, 1))}

    fig = visualization.visualize_error(
        data,
        output,
        node_normalizer=SimpleNamespace(baseMVA=1000),
    )

    assert calls["visualization"] is True
    assert calls["args"] is None
    assert calls["inputs"][0] is output
    assert calls["inputs"][1] is data.y_dict
    assert calls["inputs"][2] is data.edge_index_dict
    assert calls["inputs"][3] is data.edge_attr_dict
    assert calls["inputs"][4] is data.mask_dict
    assert calls["inputs"][5] is data.x_dict
    assert len(fig.axes) == 2
    assert fig.axes[0].get_title() == "Nodal Active Power Residuals"
    assert fig.axes[1].get_ylabel() == "Nodal Active Power Residual (MW)"
    assert {label.get_text() for label in fig.axes[0].texts} == {"PQ", "PV", "REF"}
    visualization.plt.close(fig)


def test_visualize_quantity_heatmap_uses_bus_columns_and_does_not_mutate_inputs(
    monkeypatch,
):
    data = _data()
    output = {
        "bus": torch.tensor(
            [
                [1.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.3, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.4, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ),
        "gen": torch.tensor([[4.0], [5.0], [6.0]]),
    }
    original_bus_target = data.y_dict["bus"].clone()
    original_gen_target = data.y_dict["gen"].clone()
    original_bus_output = output["bus"].clone()
    monkeypatch.setattr(
        visualization.plt,
        "show",
        lambda: pytest.fail("visualization should return figures without showing"),
    )

    fig = visualization.visualize_quantity_heatmap(
        data,
        output,
        quantity="Vm",
        quantity_name="Voltage magnitude",
        unit="p.u.",
        node_normalizer=_ScalingNormalizer(),
    )

    # Nodes are drawn grouped by type in NODE_SHAPES order (REF=bus 2, PV=bus 1,
    # PQ=bus 0), so the collection order is the reverse of the bus order.
    np.testing.assert_allclose(_collection_values(fig.axes[0]), [1.1, 1.0, 0.9])
    np.testing.assert_allclose(_collection_values(fig.axes[2]), [1.4, 1.0, 1.2])
    torch.testing.assert_close(data.y_dict["bus"], original_bus_target)
    torch.testing.assert_close(data.y_dict["gen"], original_gen_target)
    torch.testing.assert_close(output["bus"], original_bus_output)
    assert len(fig.axes) == 4
    visualization.plt.close(fig)


def test_visualize_quantity_heatmap_aggregates_masked_generator_pg():
    data = _data()
    output = {
        "bus": torch.zeros((3, 6)),
        "gen": torch.tensor([[4.0], [5.0], [6.0]]),
    }

    fig = visualization.visualize_quantity_heatmap(
        data,
        output,
        quantity="Pg",
        quantity_name="Active generation",
        unit="MW",
        node_normalizer=_ScalingNormalizer(),
    )

    np.testing.assert_allclose(_collection_values(fig.axes[0]), [0.0, 30.0, 30.0])
    np.testing.assert_allclose(_collection_values(fig.axes[2]), [0.0, 30.0, 90.0])
    visualization.plt.close(fig)


def test_visualize_quantity_heatmap_rejects_unsupported_quantity():
    with pytest.raises(ValueError, match="Unsupported quantity"):
        visualization._resolve_quantity("branch flow")
