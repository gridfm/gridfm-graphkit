"""Serialization between PyG ``HeteroData`` graphs and flat tensor dicts.

vLLM ships model inputs to the worker process as a dict of named tensors under
a single multimodal modality. A power-grid :class:`~torch_geometric.data.HeteroData`
graph is a nested structure of node/edge stores, so it must be flattened into
such a dict on the way in and rebuilt on the way out. This module is the single
source of truth for that mapping.

It is deliberately free of any vLLM import: the encode side runs in the API
server (inside the IO processor), the decode side runs in the vLLM worker
(inside the model), and both share these functions so the layouts can never
drift. The output packing helpers do the same job for the model's return value,
which travels back as one opaque tensor via vLLM's identity pooler.
"""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

# Relation triples used by GNS_heterogeneous, in a fixed order.
BUS_BUS = ("bus", "connects", "bus")
GEN_BUS = ("gen", "connected_to", "bus")
BUS_GEN = ("bus", "connected_to", "gen")

# Flat multimodal field names. GNS_heterogeneous.forward consumes x_dict,
# edge_index_dict, edge_attr_dict and mask_dict. The PowerFlow reconstruction
# task reads mask_dict["bus"]/["gen"] in the forward loop and the physics
# decoder additionally reads mask_dict["PQ"]/["PV"]/["REF"], so the full mask
# set produced by AddPFHeteroMask must cross the wire (branch mask included so
# the layout matches other tasks and future decoders).
FIELD_BUS_X = "bus_x"
FIELD_GEN_X = "gen_x"
FIELD_BUS_BUS_EDGE_INDEX = "bus_bus_edge_index"
FIELD_BUS_BUS_EDGE_ATTR = "bus_bus_edge_attr"
FIELD_GEN_BUS_EDGE_INDEX = "gen_bus_edge_index"
FIELD_BUS_GEN_EDGE_INDEX = "bus_gen_edge_index"
FIELD_MASK_BUS = "mask_bus"
FIELD_MASK_GEN = "mask_gen"
FIELD_MASK_BRANCH = "mask_branch"
FIELD_MASK_PQ = "mask_pq"
FIELD_MASK_PV = "mask_pv"
FIELD_MASK_REF = "mask_ref"

# Mapping from flat field name -> mask_dict key, preserving AddPFHeteroMask's set.
_MASK_FIELDS = {
    FIELD_MASK_BUS: "bus",
    FIELD_MASK_GEN: "gen",
    FIELD_MASK_BRANCH: "branch",
    FIELD_MASK_PQ: "PQ",
    FIELD_MASK_PV: "PV",
    FIELD_MASK_REF: "REF",
}

GRAPH_FIELDS = (
    FIELD_BUS_X,
    FIELD_GEN_X,
    FIELD_BUS_BUS_EDGE_INDEX,
    FIELD_BUS_BUS_EDGE_ATTR,
    FIELD_GEN_BUS_EDGE_INDEX,
    FIELD_BUS_GEN_EDGE_INDEX,
    *_MASK_FIELDS.keys(),
)


def encode_hetero_data(data: HeteroData) -> dict[str, torch.Tensor]:
    """Flatten a masked, normalized ``HeteroData`` into named tensors.

    Args:
        data: A graph that has already been through the normalizer and the
            task masking transform, so ``data.mask_dict`` is populated.

    Returns:
        A dict keyed by :data:`GRAPH_FIELDS`, holding the tensors the model
        needs to rebuild the graph and run a forward pass.
    """
    mask_dict = data.mask_dict
    fields = {
        FIELD_BUS_X: data["bus"].x,
        FIELD_GEN_X: data["gen"].x,
        FIELD_BUS_BUS_EDGE_INDEX: data[BUS_BUS].edge_index,
        FIELD_BUS_BUS_EDGE_ATTR: data[BUS_BUS].edge_attr,
        FIELD_GEN_BUS_EDGE_INDEX: data[GEN_BUS].edge_index,
        FIELD_BUS_GEN_EDGE_INDEX: data[BUS_GEN].edge_index,
    }
    for field_name, mask_key in _MASK_FIELDS.items():
        fields[field_name] = mask_dict[mask_key]
    return fields


def decode_hetero_data(fields: dict[str, torch.Tensor]) -> HeteroData:
    """Rebuild a ``HeteroData`` (with ``mask_dict``) from flat named tensors.

    Inverse of :func:`encode_hetero_data`. Expects each field at its natural
    rank (a single graph, no leading item dimension). vLLM's multimodal
    collation adds a leading "items" dimension across the batch; the model
    wrapper slices one item off before calling this, so the tensors here always
    describe exactly one graph.
    """
    bus_x = fields[FIELD_BUS_X]
    gen_x = fields[FIELD_GEN_X]
    bus_bus_edge_index = fields[FIELD_BUS_BUS_EDGE_INDEX].long()
    bus_bus_edge_attr = fields[FIELD_BUS_BUS_EDGE_ATTR]
    gen_bus_edge_index = fields[FIELD_GEN_BUS_EDGE_INDEX].long()
    bus_gen_edge_index = fields[FIELD_BUS_GEN_EDGE_INDEX].long()

    data = HeteroData()
    data["bus"].x = bus_x
    data["gen"].x = gen_x
    data[BUS_BUS].edge_index = bus_bus_edge_index
    data[BUS_BUS].edge_attr = bus_bus_edge_attr
    data[GEN_BUS].edge_index = gen_bus_edge_index
    data[BUS_GEN].edge_index = bus_gen_edge_index

    # bus/gen/branch masks are 2D [n, feat]; PQ/PV/REF are 1D [n_bus].
    mask_dict: dict[str, torch.Tensor] = {}
    for field_name, mask_key in _MASK_FIELDS.items():
        mask_dict[mask_key] = fields[field_name].bool()
    data.mask_dict = mask_dict
    return data


# --- Model output packing ------------------------------------------------
#
# The model returns one opaque tensor (vLLM's identity pooler passes it through
# unchanged as ``PoolingRequestOutput.outputs.data``). We flatten the four
# result tensors — bus/gen predictions and bus/gen embeddings — into a single 1D
# float tensor prefixed by a small integer header describing their shapes, then
# unpack on the IO-processor side. This keeps the wire format a single tensor of
# a single dtype regardless of differing node counts and feature widths.

_HEADER_LEN = 5  # n_bus, n_gen, bus_pred_dim, gen_pred_dim, emb_dim


def pack_outputs(
    bus_pred: torch.Tensor,
    gen_pred: torch.Tensor,
    bus_emb: torch.Tensor,
    gen_emb: torch.Tensor,
) -> torch.Tensor:
    """Flatten predictions + embeddings into one 1D float tensor.

    Args:
        bus_pred: ``[n_bus, bus_pred_dim]`` per-bus predictions (Vm, Va, ...).
        gen_pred: ``[n_gen, gen_pred_dim]`` per-generator predictions (Pg, ...).
        bus_emb: ``[n_bus, emb_dim]`` per-bus latent embeddings.
        gen_emb: ``[n_gen, emb_dim]`` per-generator latent embeddings.

    Returns:
        A 1D float tensor: ``[header(5) | bus_pred | gen_pred | bus_emb | gen_emb]``.
        ``bus_emb`` and ``gen_emb`` share ``emb_dim``.
    """
    n_bus, bus_pred_dim = bus_pred.shape
    n_gen, gen_pred_dim = gen_pred.shape
    emb_dim = bus_emb.shape[1]
    if gen_emb.shape[1] != emb_dim:
        raise ValueError(
            f"bus and gen embeddings must share emb_dim, got {emb_dim} and "
            f"{gen_emb.shape[1]}",
        )
    header = torch.tensor(
        [n_bus, n_gen, bus_pred_dim, gen_pred_dim, emb_dim],
        dtype=torch.float32,
        device=bus_pred.device,
    )
    return torch.cat(
        [
            header,
            bus_pred.reshape(-1).float(),
            gen_pred.reshape(-1).float(),
            bus_emb.reshape(-1).float(),
            gen_emb.reshape(-1).float(),
        ],
    )


def unpack_outputs(packed: torch.Tensor) -> dict[str, torch.Tensor]:
    """Inverse of :func:`pack_outputs`.

    Returns:
        A dict with keys ``bus_pred``, ``gen_pred``, ``bus_emb``, ``gen_emb``.
    """
    packed = packed.reshape(-1)
    header = packed[:_HEADER_LEN].round().long().tolist()
    n_bus, n_gen, bus_pred_dim, gen_pred_dim, emb_dim = header

    offset = _HEADER_LEN
    sizes = {
        "bus_pred": (n_bus, bus_pred_dim),
        "gen_pred": (n_gen, gen_pred_dim),
        "bus_emb": (n_bus, emb_dim),
        "gen_emb": (n_gen, emb_dim),
    }
    out: dict[str, torch.Tensor] = {}
    for name, (rows, cols) in sizes.items():
        count = rows * cols
        out[name] = packed[offset : offset + count].reshape(rows, cols)
        offset += count
    return out
