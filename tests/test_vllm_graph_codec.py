"""Unit tests for the vLLM graph codec (no vLLM dependency)."""

import torch
from torch_geometric.data import HeteroData

from gridfm_graphkit.vllm import graph_codec


def _toy_graph() -> HeteroData:
    data = HeteroData()
    data["bus"].x = torch.randn(4, 15)
    data["gen"].x = torch.randn(2, 7)
    data[graph_codec.BUS_BUS].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 3]],
        dtype=torch.long,
    )
    data[graph_codec.BUS_BUS].edge_attr = torch.randn(3, 11)
    data[graph_codec.GEN_BUS].edge_index = torch.tensor(
        [[0, 1], [0, 3]],
        dtype=torch.long,
    )
    data[graph_codec.BUS_GEN].edge_index = torch.tensor(
        [[0, 3], [0, 1]],
        dtype=torch.long,
    )
    data.mask_dict = {
        "bus": torch.zeros(4, 15, dtype=torch.bool),
        "gen": torch.ones(2, 7, dtype=torch.bool),
        "branch": torch.ones(3, 11, dtype=torch.bool),
        "PQ": torch.tensor([True, False, True, False]),
        "PV": torch.tensor([False, True, False, False]),
        "REF": torch.tensor([False, False, False, True]),
    }
    return data


def test_encode_decode_round_trip() -> None:
    data = _toy_graph()
    fields = graph_codec.encode_hetero_data(data)

    assert set(fields) == set(graph_codec.GRAPH_FIELDS)

    rebuilt = graph_codec.decode_hetero_data(fields)

    torch.testing.assert_close(rebuilt["bus"].x, data["bus"].x)
    torch.testing.assert_close(rebuilt["gen"].x, data["gen"].x)
    torch.testing.assert_close(
        rebuilt[graph_codec.BUS_BUS].edge_index,
        data[graph_codec.BUS_BUS].edge_index,
    )
    torch.testing.assert_close(
        rebuilt[graph_codec.BUS_BUS].edge_attr,
        data[graph_codec.BUS_BUS].edge_attr,
    )
    torch.testing.assert_close(
        rebuilt[graph_codec.GEN_BUS].edge_index,
        data[graph_codec.GEN_BUS].edge_index,
    )
    torch.testing.assert_close(
        rebuilt[graph_codec.BUS_GEN].edge_index,
        data[graph_codec.BUS_GEN].edge_index,
    )
    for key in ("bus", "gen", "branch", "PQ", "PV", "REF"):
        assert torch.equal(rebuilt.mask_dict[key], data.mask_dict[key])


def test_decode_of_per_item_slice_of_batched_fields() -> None:
    """vLLM's multimodal collation prepends an item dimension to every field.

    The model wrapper slices one item off before decoding (see
    ``GridFMForPooling.forward``); ``decode_hetero_data`` itself expects each
    field at its natural rank. This mirrors that split: batch the fields, take
    item ``0``, and confirm the decoded graph matches the original.
    """
    data = _toy_graph()
    batched = {
        k: v.unsqueeze(0) for k, v in graph_codec.encode_hetero_data(data).items()
    }
    item_fields = {k: v[0] for k, v in batched.items()}

    rebuilt = graph_codec.decode_hetero_data(item_fields)

    assert rebuilt["bus"].x.shape == (4, 15)
    assert rebuilt[graph_codec.BUS_BUS].edge_index.shape == (2, 3)


def test_pack_unpack_round_trip() -> None:
    n_bus, n_gen, emb_dim = 4, 2, 8
    bus_pred = torch.randn(n_bus, 2)
    gen_pred = torch.randn(n_gen, 1)
    bus_emb = torch.randn(n_bus, emb_dim)
    gen_emb = torch.randn(n_gen, emb_dim)

    packed = graph_codec.pack_outputs(bus_pred, gen_pred, bus_emb, gen_emb)
    assert packed.ndim == 1

    out = graph_codec.unpack_outputs(packed)
    torch.testing.assert_close(out["bus_pred"], bus_pred)
    torch.testing.assert_close(out["gen_pred"], gen_pred)
    torch.testing.assert_close(out["bus_emb"], bus_emb)
    torch.testing.assert_close(out["gen_emb"], gen_emb)
