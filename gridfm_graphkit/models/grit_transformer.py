import copy

from gridfm_graphkit.io.registries import MODELS_REGISTRY
import torch
from torch import nn
from torch_geometric.data import Data

from gridfm_graphkit.models.rrwp_encoder import (
    RRWPLinearNodeEncoder,
    RRWPLinearEdgeEncoder,
)
from gridfm_graphkit.models.grit_layer import GritTransformerLayer
from gridfm_graphkit.models.kernel_pos_encoder import RWSENodeEncoder

try:
    from torch_scatter import scatter_add
except ImportError:
    scatter_add = None

from gridfm_graphkit.datasets.globals import PG_H


class BatchNorm1dNode(torch.nn.Module):
    r"""A batch normalization layer for node-level features.

    Args:
        dim_in (int): BatchNorm input dimension.
        eps (float): BatchNorm eps.
        momentum (float): BatchNorm momentum.
    """

    def __init__(self, dim_in, eps, momentum):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(
            dim_in,
            eps=eps,
            momentum=momentum,
        )

    def forward(self, batch):
        batch.x = self.bn(batch.x)
        return batch


class BatchNorm1dEdge(torch.nn.Module):
    r"""A batch normalization layer for edge-level features.

    Args:
        dim_in (int): BatchNorm input dimension.
        eps (float): BatchNorm eps.
        momentum (float): BatchNorm momentum.
    """

    def __init__(self, dim_in, eps, momentum):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(
            dim_in,
            eps=eps,
            momentum=momentum,
        )

    def forward(self, batch):
        batch.edge_attr = self.bn(batch.edge_attr)
        return batch


class LinearNodeEncoder(torch.nn.Module):
    def __init__(self, dim_in, emb_dim):
        super().__init__()

        self.encoder = torch.nn.Linear(dim_in, emb_dim)

    def forward(self, batch):
        batch.x = self.encoder(batch.x)
        return batch


class LinearEdgeEncoder(torch.nn.Module):
    def __init__(self, edge_dim, emb_dim):
        super().__init__()

        self.in_dim = edge_dim

        self.encoder = torch.nn.Linear(self.in_dim, emb_dim)

    def forward(self, batch):
        batch.edge_attr = self.encoder(batch.edge_attr.view(-1, self.in_dim))
        return batch


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension

    """

    def __init__(self, dim_in, dim_inner, args):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if args.encoder.node_encoder:
            # Encode integer node features via nn.Embeddings
            if "RWSE" in args.encoder.node_encoder_name:
                self.node_encoder = RWSENodeEncoder(
                    self.dim_in,
                    dim_inner,
                    args.encoder.posenc_RWSE,
                )
            else:
                self.node_encoder = LinearNodeEncoder(self.dim_in, dim_inner)
            if args.encoder.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(dim_inner, 1e-5, 0.1)
            # Update dim_in to reflect the new dimension fo the node features
            self.dim_in = dim_inner
        if args.encoder.edge_encoder:
            edge_dim = args.edge_dim
            enc_dim_edge = dim_inner
            # Encode integer edge features via nn.Embeddings
            self.edge_encoder = LinearEdgeEncoder(edge_dim, enc_dim_edge)
            if args.encoder.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(enc_dim_edge, 1e-5, 0.1)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class GraphHead(nn.Module):
    """
    Prediction head for decoding tasks.
    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        L (int): Number of hidden layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.FC_layers = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.LeakyReLU(),
            nn.Linear(dim_in, dim_out),
        )

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        graph_emb = self.FC_layers(batch.x)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred


class GritTransformer(torch.nn.Module):
    """
    The GritTransformer (Graph Inductive Bias Transformer) from
    Graph Inductive Biases in Transformers without Message Passing, L. Ma et al.,
    2023.

    """

    def __init__(self, args, include_decoder=True):
        super().__init__()

        dim_in = args.model.input_dim
        dim_out = args.model.output_dim
        dim_inner = args.model.hidden_size
        num_heads = args.model.attention_head
        dropout = args.model.dropout
        num_layers = args.model.num_layers
        self.mask_dim = getattr(args.data, "mask_dim", 6)
        mask_val = getattr(args.data, "mask_value", -1.0)
        self.register_buffer(
            "mask_value",
            torch.zeros(self.mask_dim) + mask_val,
        )

        self.encoder = FeatureEncoder(dim_in, dim_inner, args.model)
        dim_in = self.encoder.dim_in

        if args.data.posenc_RRWP.enable:
            self.rrwp_abs_encoder = RRWPLinearNodeEncoder(
                args.data.posenc_RRWP.ksteps,
                dim_inner,
            )
            rel_pe_dim = args.data.posenc_RRWP.ksteps
            self.rrwp_rel_encoder = RRWPLinearEdgeEncoder(
                rel_pe_dim,
                dim_inner,
                pad_to_full_graph=args.model.gt.attn.full_attn,
                add_node_attr_as_self_loop=False,
                fill_value=0.0,
            )

        assert args.model.hidden_size == dim_inner == dim_in, (
            "The inner and hidden dims must match."
        )

        layers = []
        for ll in range(num_layers):
            # The last layer's edge output is never consumed downstream
            # (only node features feed into the output heads), so skip
            # creating O_e / norm_e parameters to avoid DDP unused-parameter
            # errors.
            is_last = ll == num_layers - 1
            layers.append(
                GritTransformerLayer(
                    in_dim=args.model.gt.dim_hidden,
                    out_dim=args.model.gt.dim_hidden,
                    num_heads=num_heads,
                    dropout=dropout,
                    act=args.model.act,
                    attn_dropout=args.model.gt.attn_dropout,
                    layer_norm=args.model.gt.layer_norm,
                    batch_norm=args.model.gt.batch_norm,
                    residual=True,
                    norm_e=False if is_last else args.model.gt.attn.norm_e,
                    O_e=False if is_last else args.model.gt.attn.O_e,
                    cfg=args.model.gt,
                ),
            )

        self.layers = nn.Sequential(*layers)

        if include_decoder:
            self.decoder = GraphHead(dim_inner, dim_out)

    def forward(self, batch):
        """
        Forward pass for GRIT.

        Args:
            batch (Batch): Pytorch Geometric Batch object, with x, y encodings, etc.

        Returns:
            output (Tensor): Output node features of shape [num_nodes, output_dim].
        """
        # print('xxxx',batch.x.min(), batch.x.max())
        # print('yyyyy',batch.y.min(), batch.y.max())
        # print('>>>>', batch)
        for module in self.children():
            batch = module(batch)

        return batch


def aggregate_pg(batch, mask_value=-1.0):
    """Aggregate per-generator active power (PG) onto bus nodes.

    In the homogeneous reference, PG is a direct bus feature visible to the
    transformer alongside Pd, Qd, Vm, Va, etc.  In the heterogeneous
    representation PG lives on separate generator nodes, so it must be
    aggregated onto buses before the transformer can learn voltage-power
    coupling.

    Masked generators (where PG has been replaced by the mask value) are
    excluded from the sum to avoid corrupting the aggregated signal.  Buses
    where *all* connected generators are masked receive the mask value
    instead, preserving a consistent "unknown" indicator.
    """
    if scatter_add is None:
        raise ImportError(
            "torch-scatter is required for the GRIT modules but is not installed. "
            "Install it with: pip install torch-scatter",
        )

    gen_to_bus = batch["gen", "connected_to", "bus"].edge_index
    gen_pg = batch["gen"].x[:, PG_H]
    gen_masked = batch.mask_dict["gen"][:, PG_H]  # True = masked

    # Zero out masked generators so they don't contribute to the sum
    pg_clean = torch.where(gen_masked, torch.zeros_like(gen_pg), gen_pg)

    pg_per_bus = scatter_add(
        pg_clean,
        gen_to_bus[1],
        dim=0,
        dim_size=batch["bus"].x.size(0),
    )

    # Check which buses have ALL generators masked (or no generators at all)
    unmasked_count = scatter_add(
        (~gen_masked).float(),
        gen_to_bus[1],
        dim=0,
        dim_size=batch["bus"].x.size(0),
    )
    all_masked = unmasked_count == 0

    # Set mask_value for fully-masked buses
    pg_per_bus[all_masked] = mask_value

    return pg_per_bus


@MODELS_REGISTRY.register("GRIT")
class GritHeteroAdapter(torch.nn.Module):
    """Adapter that enables the homogeneous GRIT transformer to operate on
    heterogeneous power-grid graphs.

    Extracts the bus-only homogeneous subgraph using PyG's native HeteroData
    accessors, runs it through the GRIT encoder and transformer layers, and
    produces per-node-type predictions.  Generator output comes from a
    lightweight standalone MLP (generators are not seen by the transformer).

    Returns:
        dict: ``{"bus": Tensor[num_bus, output_bus_dim],
                  "gen": Tensor[num_gen, output_gen_dim]}``
    """

    def __init__(self, args):
        super().__init__()
        # Work on a local copy to avoid mutating the shared config object,
        # which would be a problem if multiple models are constructed from
        # the same args (e.g. ensembles, teacher-student, A/B comparisons).
        args = copy.deepcopy(args)

        dim_inner = args.model.hidden_size
        output_bus_dim = args.model.output_bus_dim
        output_gen_dim = args.model.output_gen_dim
        input_gen_dim = args.model.input_gen_dim

        # Ensure config keys expected by GritTransformer are present.
        # input_dim  = bus feature dimension  (used by FeatureEncoder)
        # output_dim = bus output dimension   (used by the unused GraphHead)
        if not hasattr(args.model, "input_dim"):
            args.model.input_dim = args.model.input_bus_dim
        if not hasattr(args.model, "output_dim"):
            args.model.output_dim = output_bus_dim

        # Sync PE kernel.times from data config into model encoder config so
        # users only need to specify it once (under data.posenc_RWSE).
        if (
            hasattr(args.data, "posenc_RWSE")
            and args.data.posenc_RWSE.enable
            and hasattr(args.model, "encoder")
            and hasattr(args.model.encoder, "posenc_RWSE")
        ):
            from gridfm_graphkit.io.param_handler import NestedNamespace

            enc_rwse = args.model.encoder.posenc_RWSE
            if not hasattr(enc_rwse, "kernel"):
                enc_rwse.kernel = NestedNamespace()
            enc_rwse.kernel.times = args.data.posenc_RWSE.kernel.times

        # Sync gt.dim_hidden from model.hidden_size so it is specified once.
        if hasattr(args.model, "gt"):
            args.model.gt.dim_hidden = args.model.hidden_size

        # The original homogeneous GRIT
        # (encoder + optional PE encoders + transformer layers)
        # Decoder is excluded — this adapter provides its own per-type heads.
        self.grit = GritTransformer(args, include_decoder=False)

        # Per-node-type output heads (replace GraphHead for hetero output)
        self.bus_head = nn.Sequential(
            nn.Linear(dim_inner, dim_inner),
            nn.LeakyReLU(),
            nn.Linear(dim_inner, output_bus_dim),
        )
        # gen_head is only needed for tasks that require per-generator
        # predictions (e.g. OPF cost computation). When output_gen_dim is 0
        # or not set, skip it to avoid DDP unused-parameter errors.
        if output_gen_dim and output_gen_dim > 0:
            self.gen_head = nn.Sequential(
                nn.Linear(input_gen_dim, dim_inner),
                nn.LeakyReLU(),
                nn.Linear(dim_inner, output_gen_dim),
            )
        else:
            self.gen_head = None

    def forward(self, batch, return_embeddings: bool = False):
        """Forward pass on a heterogeneous power-grid batch.

        Args:
            batch: A batched ``HeteroData`` with node types ``"bus"`` and
                ``"gen"``, and edge type ``("bus", "connects", "bus")``.

        Returns:
            dict with keys ``"bus"`` and ``"gen"``, each mapping to the
            predicted output features. When ``return_embeddings`` is ``True``,
            also returns a ``{"bus": ...}`` embedding dictionary, where the bus
            embedding is the latent tensor (``homo.x``) fed into the prediction
            heads. This model does not expose a separate ``"gen"`` embedding, so
            downstream gen-embedding export is skipped for GRIT.
        """
        # --- Extract bus-only homogeneous subgraph ---
        # Aggregate generator PG onto buses
        pg_per_bus = aggregate_pg(batch, mask_value=self.grit.mask_value[0].item())
        bus_x = torch.cat(
            [batch["bus"].x, pg_per_bus.unsqueeze(-1)],
            dim=-1,
        )  # 15 → 16D

        homo = Data(
            x=bus_x,
            y=batch["bus"].y,
            edge_index=batch["bus", "connects", "bus"].edge_index,
            edge_attr=batch["bus", "connects", "bus"].edge_attr,
            batch=batch["bus"].batch,
        )

        # Forward positional-encoding attributes if present
        for attr in ("pestat_RWSE", "rrwp", "log_deg", "deg"):
            if hasattr(batch["bus"], attr):
                setattr(homo, attr, getattr(batch["bus"], attr))

        # RRWP relative PE is stored as a dedicated edge type for correct batching
        if ("bus", "rrwp", "bus") in batch.edge_types:
            homo.rrwp_index = batch["bus", "rrwp", "bus"].edge_index
            homo.rrwp_val = batch["bus", "rrwp", "bus"].edge_attr

        # --- Run GRIT encoder + PE encoders + transformer layers ---
        homo = self.grit.encoder(homo)
        if hasattr(self.grit, "rrwp_abs_encoder"):
            homo = self.grit.rrwp_abs_encoder(homo)
        if hasattr(self.grit, "rrwp_rel_encoder"):
            homo = self.grit.rrwp_rel_encoder(homo)
        homo = self.grit.layers(homo)

        # --- Per-type decoding ---
        bus_out = self.bus_head(homo.x)
        gen_out = (
            self.gen_head(batch["gen"].x)
            if self.gen_head is not None
            else batch["gen"].x
        )

        predictions = {"bus": bus_out, "gen": gen_out}
        if not return_embeddings:
            return predictions
        return predictions, {"bus": homo.x}
