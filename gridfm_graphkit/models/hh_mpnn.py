"""HH-MPNN: Hybrid Heterogeneous Message Passing Neural Network.

Implements the architecture of Arowolo & Cremer, "Towards Generalization of
Graph Neural Networks for AC Optimal Power Flow" (arXiv:2510.06860) on the
gridfm-graphkit heterogeneous graph.

Each layer combines a local step, a heterogeneous interaction network with
type-specific edge and node update functions, with a global step, Performer
attention over all nodes of a graph regardless of type. Node states are
initialised from type-specific encoders concatenated with an
effective-resistance positional encoding.
"""

import math

import torch
from torch import nn
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn.attention.performer import generalized_kernel, linear_attention
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_add

from gridfm_graphkit.datasets.effective_resistance import ER_PE_ATTR, ER_PE_DIM
from gridfm_graphkit.datasets.globals import (
    # Bus feature indices
    PD_H,
    QD_H,
    QG_H,
    VM_H,
    VA_H,
    MIN_VM_H,
    MAX_VM_H,
    MIN_QG_H,
    MAX_QG_H,
    GS,
    BS,
    # Generator feature indices
    PG_H,
    MIN_PG,
    MAX_PG,
)
from gridfm_graphkit.io.param_handler import get_physics_decoder
from gridfm_graphkit.io.registries import MODELS_REGISTRY
from gridfm_graphkit.models.utils import (
    ComputeBranchFlow,
    ComputeNodeInjection,
    bound_with_sigmoid,
)

BRANCH = ("bus", "connects", "bus")
GEN_TO_BUS = ("gen", "connected_to", "bus")
LOAD_TO_BUS = ("load", "connected_to", "bus")
SHUNT_TO_BUS = ("shunt", "connected_to", "bus")

# Bus columns that become the features of the load and shunt nodes.
LOAD_COLS = [PD_H, QD_H]
SHUNT_COLS = [GS, BS]

# OPF voltage angles are squashed into [-VA_BOUND, VA_BOUND] radians.
VA_BOUND = 2.0


def _edge_key(edge_type) -> str:
    """ModuleDict key for an edge type tuple."""
    return "__".join(edge_type)


class MLP(nn.Module):
    """ReLU multi-layer perceptron with an optional final LayerNorm.

    Args:
        input_size: Input width.
        hidden_size: Width of the hidden layers.
        output_size: Output width.
        layers: Number of linear layers.
        layernorm: Whether to normalise the output.
    """

    def __init__(self, input_size, hidden_size, output_size, layers, layernorm=True):
        super().__init__()
        modules = []
        for i in range(layers):
            modules.append(
                nn.Linear(
                    input_size if i == 0 else hidden_size,
                    output_size if i == layers - 1 else hidden_size,
                ),
            )
            if i != layers - 1:
                modules.append(nn.ReLU())
        if layernorm:
            modules.append(nn.LayerNorm(output_size))
        self.network = nn.Sequential(*modules)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialise weights from N(0, 1 / fan_in) and biases to zero."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0, 1 / math.sqrt(layer.in_features))
                layer.bias.data.fill_(0)

    def forward(self, x):
        return self.network(x)


class HeteroInteractionNetwork(nn.Module):
    """One message-passing step over a heterogeneous graph.

    Every edge computes a message from its two end nodes, and from its own
    state if it is a physical edge. The message is summed into both end nodes,
    and each node type then updates its state with its own MLP.

    Args:
        node_types: Node types whose state is updated.
        edge_types: All ``(src, relation, dst)`` edge types.
        physical_edge_types: The edge types that carry an edge state.
        hidden_size: Width of node and edge states.
        layers: Number of linear layers in each update MLP.
    """

    def __init__(
        self,
        node_types,
        edge_types,
        physical_edge_types,
        hidden_size,
        layers,
    ):
        super().__init__()
        self.physical_edge_types = list(physical_edge_types)
        self.edge_updaters = nn.ModuleDict()
        for edge_type in edge_types:
            num_inputs = 3 if edge_type in self.physical_edge_types else 2
            self.edge_updaters[_edge_key(edge_type)] = MLP(
                hidden_size * num_inputs,
                hidden_size,
                hidden_size,
                layers,
            )
        self.node_updaters = nn.ModuleDict(
            {
                node_type: MLP(hidden_size * 2, hidden_size, hidden_size, layers)
                for node_type in node_types
            },
        )

    def forward(self, x_dict, edge_index_dict, edge_state_dict):
        """Run one message-passing step.

        Args:
            x_dict: Node states per node type.
            edge_index_dict: ``[2, E]`` edge index per edge type.
            edge_state_dict: Edge states of the physical edge types.

        Returns:
            Tuple of the updated node states, for the node types given at
            construction, and the updated physical edge states.
        """
        new_edge_states = {}
        messages = {node_type: torch.zeros_like(x) for node_type, x in x_dict.items()}

        for edge_type, (src, dst) in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            inputs = [x_dict[dst_type][dst], x_dict[src_type][src]]
            updater = self.edge_updaters[_edge_key(edge_type)]
            if edge_type in self.physical_edge_types:
                edge_state = edge_state_dict[edge_type]
                message = updater(torch.cat(inputs + [edge_state], dim=-1)) + edge_state
                new_edge_states[edge_type] = message
            else:
                message = updater(torch.cat(inputs, dim=-1))

            messages[dst_type] = messages[dst_type] + scatter_add(
                message,
                dst,
                dim=0,
                dim_size=x_dict[dst_type].size(0),
            )
            messages[src_type] = messages[src_type] + scatter_add(
                message,
                src,
                dim=0,
                dim_size=x_dict[src_type].size(0),
            )

        new_x_dict = {
            node_type: x_dict[node_type]
            + updater(torch.cat((x_dict[node_type], messages[node_type]), dim=-1))
            for node_type, updater in self.node_updaters.items()
        }
        return new_x_dict, new_edge_states


class MaskedPerformerAttention(PerformerAttention):
    """Performer attention in which padded positions contribute nothing.

    ``PerformerAttention`` zeroes the values of padded positions but not their
    keys, which still add a constant to the attention normaliser. The output
    for a graph then depends on how much padding the rest of the batch gives
    it. Zeroing the keys as well makes it independent of the batch; without
    padding both versions are identical.
    """

    def forward(self, x, mask=None):
        """Attend within each row of a dense batch.

        Args:
            x: ``[batch_size, max_nodes, channels]`` node states.
            mask: ``[batch_size, max_nodes]`` bool, ``True`` for real nodes.

        Returns:
            ``[batch_size, max_nodes, channels]`` attention output.
        """
        batch_size, max_nodes, _ = x.shape
        q, k, v = (
            linear(x)
            .reshape(batch_size, max_nodes, self.heads, self.head_channels)
            .permute(0, 2, 1, 3)
            for linear in (self.q, self.k, self.v)
        )
        projection = self.fast_attn.projection_matrix
        q = generalized_kernel(q, projection, self.kernel)
        k = generalized_kernel(k, projection, self.kernel)
        if mask is not None:
            keep = mask[:, None, :, None]
            k = k * keep
            v = v * keep
        out = linear_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, max_nodes, -1)
        return self.dropout(self.attn_out(out))


class HeteroPerformerLayer(nn.Module):
    """Global attention over all nodes of a graph, fused with message passing.

    Args:
        hidden_size: Width of the node states.
        num_heads: Number of Performer attention heads.
    """

    def __init__(self, hidden_size, num_heads=1):
        super().__init__()
        self.attn = MaskedPerformerAttention(channels=hidden_size, heads=num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, x_dict, xm_dict, batch_dict, num_graphs):
        """Attend over all nodes, then merge with the message-passing states.

        Args:
            x_dict: Node states before this layer, per node type.
            xm_dict: Node states after message passing. May cover only some
                node types.
            batch_dict: Graph assignment of each node, per node type.
            num_graphs: Number of graphs in the batch.

        Returns:
            New node states for the node types present in ``xm_dict``.
        """
        node_types = list(x_dict)
        sizes = [x_dict[node_type].size(0) for node_type in node_types]
        x_all = torch.cat([x_dict[node_type] for node_type in node_types], dim=0)
        batch_all = torch.cat([batch_dict[node_type] for node_type in node_types])

        # to_dense_batch needs the nodes of each graph to be contiguous.
        order = torch.argsort(batch_all, stable=True)
        x_dense, mask = to_dense_batch(
            x_all[order],
            batch_all[order],
            batch_size=num_graphs,
        )
        attended = self.norm1(x_dense + self.attn(x_dense, mask=mask))[mask]
        xt_all = torch.empty_like(attended)
        xt_all[order] = attended

        xt_dict = dict(zip(node_types, torch.split(xt_all, sizes)))
        return {
            node_type: self.mlp(self.norm2(xt_dict[node_type] + xm))
            for node_type, xm in xm_dict.items()
        }


@MODELS_REGISTRY.register("HH_MPNN")
class HHMPNN(nn.Module):
    """HH-MPNN on the gridfm-graphkit heterogeneous graph.

    The model reads the ``bus`` and ``gen`` node stores, the
    ``("bus", "connects", "bus")`` branches and the
    ``("gen", "connected_to", "bus")`` connectors. It needs the positional
    encoding ``batch["bus"].pestat_ER``, which the ``EffectiveResistancePE``
    transform provides.

    Config keys under ``model``:

    - ``hidden_size``, ``num_layers``, ``input_bus_dim``, ``input_gen_dim``,
      ``edge_dim``: as for the other models.
    - ``attention_head`` (default 1): number of Performer heads.
    - ``load_shunt_nodes`` (default ``True``): as in the paper, give every bus
      with a non-zero load (Pd, Qd) a ``load`` node and every bus with a
      non-zero shunt (GS, BS) a ``shunt`` node, each linked to its bus by a
      connector edge. Those four columns are then removed from the bus
      features. If ``False``, the model runs on bus and gen nodes only and the
      columns stay on the bus.
    - ``physics_decoder`` (default ``False``): how the outputs are produced.

      - ``False``, OptimalPowerFlow only: as in the paper, Vm, Va and Qg (per
        bus) and Pg (per generator) are predicted directly and squashed into
        their limits with a sigmoid, so these limits hold by construction.
      - ``True``, all tasks: only Vm, Va and Pg are predicted, bounded in the
        same way for OptimalPowerFlow. Qg, and the other task-specific
        quantities, come from the task's physics decoder through the
        power-balance equations, as in ``GNS_heterogeneous``, so Qg can exceed
        its limits.

    Returns:
        ``{"bus": Tensor[num_bus, 4], "gen": Tensor[num_gen, 1]}`` with bus
        columns ``[Vm, Va, Pg, Qg]`` and generator column ``[Pg]``.
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.hidden_dim = args.model.hidden_size
        self.num_layers = args.model.num_layers
        self.task = args.task.task_name
        self.load_shunt_nodes = getattr(args.model, "load_shunt_nodes", True)
        self.use_physics_decoder = getattr(args.model, "physics_decoder", False)
        heads = getattr(args.model, "attention_head", 1)

        if not self.use_physics_decoder and self.task != "OptimalPowerFlow":
            raise ValueError(
                "HH_MPNN with bounded output heads supports only the "
                f"OptimalPowerFlow task, got '{self.task}'. Set "
                "`model.physics_decoder: true` for other tasks.",
            )

        bus_cols = list(range(args.model.input_bus_dim))
        input_dims = {"bus": len(bus_cols), "gen": args.model.input_gen_dim}
        edge_types = [BRANCH, GEN_TO_BUS]
        if self.load_shunt_nodes:
            bus_cols = [c for c in bus_cols if c not in LOAD_COLS + SHUNT_COLS]
            input_dims = {
                "bus": len(bus_cols),
                "gen": args.model.input_gen_dim,
                "load": len(LOAD_COLS),
                "shunt": len(SHUNT_COLS),
            }
            edge_types += [LOAD_TO_BUS, SHUNT_TO_BUS]
        self.register_buffer("bus_cols", torch.tensor(bus_cols), persistent=False)
        node_types = list(input_dims)

        # StateEstimation has no generator output.
        output_dims = {"bus": 2 if self.use_physics_decoder else 3}
        if self.task != "StateEstimation":
            output_dims["gen"] = 1

        self.node_encoders = nn.ModuleDict(
            {
                node_type: MLP(dim, self.hidden_dim, self.hidden_dim - ER_PE_DIM, 2)
                for node_type, dim in input_dims.items()
            },
        )
        self.edge_encoder = MLP(
            args.model.edge_dim,
            self.hidden_dim,
            self.hidden_dim,
            2,
        )

        # Node types without an output head are not updated in the last layer:
        # that update would never reach the loss, and DDP rejects parameters
        # that receive no gradient.
        self.layers = nn.ModuleList(
            [
                HeteroInteractionNetwork(
                    node_types if i < self.num_layers - 1 else list(output_dims),
                    edge_types,
                    [BRANCH],
                    self.hidden_dim,
                    2,
                )
                for i in range(self.num_layers)
            ],
        )
        self.global_attn_layers = nn.ModuleList(
            [
                HeteroPerformerLayer(self.hidden_dim, heads)
                for _ in range(self.num_layers)
            ],
        )
        self.node_decoders = nn.ModuleDict(
            {
                node_type: MLP(
                    self.hidden_dim,
                    self.hidden_dim,
                    dim,
                    2,
                    layernorm=False,
                )
                for node_type, dim in output_dims.items()
            },
        )

        if self.use_physics_decoder:
            self.branch_flow_layer = ComputeBranchFlow()
            self.node_injection_layer = ComputeNodeInjection()
            self.physics_decoder = get_physics_decoder(args)

    def _build_graph(self, batch):
        """Assemble the node features, encodings and edges the layers run on.

        Returns:
            Tuple ``(x_dict, pe_dict, batch_dict, edge_index_dict, gen_bus,
            num_graphs)``, where ``gen_bus`` is the bus index of each generator.
        """
        bus = batch["bus"]
        if not hasattr(bus, ER_PE_ATTR):
            raise ValueError(
                f"HH_MPNN needs the positional encoding `{ER_PE_ATTR}` on the bus "
                "nodes. Add `EffectiveResistancePE` to `data.transforms`.",
            )
        bus_x = bus.x
        bus_pe = getattr(bus, ER_PE_ATTR)
        num_bus = bus_x.size(0)
        bus_batch = getattr(bus, "batch", None)
        if bus_batch is None:
            bus_batch = bus_x.new_zeros(num_bus, dtype=torch.long)
        num_graphs = int(bus_batch.max()) + 1 if num_bus > 0 else 1

        gen_x = batch["gen"].x
        gen_to_bus = batch[GEN_TO_BUS].edge_index
        # Bus of each generator, independent of the order of the connectors.
        gen_bus = gen_to_bus.new_zeros(gen_x.size(0)).scatter_(
            0,
            gen_to_bus[0],
            gen_to_bus[1],
        )

        x_dict = {"bus": bus_x[:, self.bus_cols], "gen": gen_x}
        node_bus = {"gen": gen_bus}
        edge_index_dict = {BRANCH: batch[BRANCH].edge_index, GEN_TO_BUS: gen_to_bus}

        if self.load_shunt_nodes:
            for node_type, cols, edge_type in (
                ("load", LOAD_COLS, LOAD_TO_BUS),
                ("shunt", SHUNT_COLS, SHUNT_TO_BUS),
            ):
                features = bus_x[:, cols]
                at_bus = (features != 0).any(dim=1).nonzero().squeeze(1)
                x_dict[node_type] = features[at_bus]
                node_bus[node_type] = at_bus
                node_index = torch.arange(at_bus.numel(), device=at_bus.device)
                edge_index_dict[edge_type] = torch.stack([node_index, at_bus])

        # Nodes attached to a bus inherit its encoding and graph assignment.
        pe_dict = {"bus": bus_pe}
        batch_dict = {"bus": bus_batch}
        for node_type, at_bus in node_bus.items():
            pe_dict[node_type] = bus_pe[at_bus]
            batch_dict[node_type] = bus_batch[at_bus]
        return x_dict, pe_dict, batch_dict, edge_index_dict, gen_bus, num_graphs

    def _process(
        self,
        x_dict,
        edge_index_dict,
        edge_state_dict,
        batch_dict,
        num_graphs,
    ):
        """Run all message-passing and global-attention layers."""
        for interaction, attention in zip(self.layers, self.global_attn_layers):
            xm_dict, edge_state_dict = interaction(
                x_dict,
                edge_index_dict,
                edge_state_dict,
            )
            x_dict = attention(x_dict, xm_dict, batch_dict, num_graphs)
        return x_dict

    def _decode_bounded(self, batch, bus_out, gen_out, gen_bus):
        """Squash the direct predictions into their limits (OptimalPowerFlow)."""
        bus_x, gen_x = batch.x_dict["bus"], batch.x_dict["gen"]
        bus_mask, gen_mask = batch.mask_dict["bus"], batch.mask_dict["gen"]

        vm = bound_with_sigmoid(bus_out[:, 0], bus_x[:, MIN_VM_H], bus_x[:, MAX_VM_H])
        va = bound_with_sigmoid(bus_out[:, 1], -VA_BOUND, VA_BOUND)
        qg = bound_with_sigmoid(bus_out[:, 2], bus_x[:, MIN_QG_H], bus_x[:, MAX_QG_H])
        pg = bound_with_sigmoid(gen_out[:, 0], gen_x[:, MIN_PG], gen_x[:, MAX_PG])

        # Quantities that are inputs of the task keep their given value.
        vm = torch.where(bus_mask[:, VM_H], vm, bus_x[:, VM_H])
        va = torch.where(bus_mask[:, VA_H], va, bus_x[:, VA_H])
        qg = torch.where(bus_mask[:, QG_H], qg, bus_x[:, QG_H])
        pg = torch.where(gen_mask[:, PG_H], pg, gen_x[:, PG_H])

        pg_bus = scatter_add(pg, gen_bus, dim=0, dim_size=bus_x.size(0))
        return {
            "bus": torch.stack([vm, va, pg_bus, qg], dim=1),
            "gen": pg.unsqueeze(-1),
        }

    def _decode_physics(self, batch, bus_out, gen_out, gen_bus):
        """Complete the predictions with the task's physics decoder."""
        bus_x, gen_x = batch.x_dict["bus"], batch.x_dict["gen"]
        mask_dict = batch.mask_dict
        num_bus = bus_x.size(0)
        branch_index = batch[BRANCH].edge_index
        branch_attr = batch[BRANCH].edge_attr

        if self.task == "StateEstimation":
            p_flow, q_flow = self.branch_flow_layer(bus_out, branch_index, branch_attr)
            p_in, q_in = self.node_injection_layer(
                p_flow,
                q_flow,
                branch_index,
                num_bus,
            )
            bus_pred = self.physics_decoder(p_in, q_in, bus_out, bus_x, None, None)
            # No generator head: pass the generator input through. Cloned,
            # because the normalizer rescales model outputs in place.
            return {"bus": bus_pred, "gen": gen_x[:, : PG_H + 1].clone()}

        vm, va, pg = bus_out[:, 0], bus_out[:, 1], gen_out[:, 0]
        if self.task == "OptimalPowerFlow":
            vm = bound_with_sigmoid(vm, bus_x[:, MIN_VM_H], bus_x[:, MAX_VM_H])
            va = bound_with_sigmoid(va, -VA_BOUND, VA_BOUND)
            pg = bound_with_sigmoid(pg, gen_x[:, MIN_PG], gen_x[:, MAX_PG])
        vm = torch.where(mask_dict["bus"][:, VM_H], vm, bus_x[:, VM_H])
        va = torch.where(mask_dict["bus"][:, VA_H], va, bus_x[:, VA_H])
        pg = torch.where(mask_dict["gen"][:, PG_H], pg, gen_x[:, PG_H])

        voltages = torch.stack([vm, va], dim=1)
        p_flow, q_flow = self.branch_flow_layer(voltages, branch_index, branch_attr)
        p_in, q_in = self.node_injection_layer(p_flow, q_flow, branch_index, num_bus)
        pg_bus = scatter_add(pg, gen_bus, dim=0, dim_size=num_bus)
        bus_pred = self.physics_decoder(p_in, q_in, voltages, bus_x, pg_bus, mask_dict)
        return {"bus": bus_pred, "gen": pg.unsqueeze(-1)}

    def forward(self, batch, return_embeddings: bool = False):
        """Predict the bus and generator quantities of a batch of grids.

        Args:
            batch: ``HeteroData`` batch after the task transforms and
                ``EffectiveResistancePE``.
            return_embeddings: If ``True``, also return the node states fed
                into the output heads.

        Returns:
            The prediction dictionary described in the class docstring. With
            ``return_embeddings``, a tuple ``(predictions, embeddings)`` where
            ``embeddings`` has a ``"bus"`` entry and, except for
            StateEstimation, a ``"gen"`` entry.
        """
        x_dict, pe_dict, batch_dict, edge_index_dict, gen_bus, num_graphs = (
            self._build_graph(batch)
        )
        x_dict = {
            node_type: torch.cat(
                [self.node_encoders[node_type](x), pe_dict[node_type]],
                dim=-1,
            )
            for node_type, x in x_dict.items()
        }
        edge_state_dict = {BRANCH: self.edge_encoder(batch[BRANCH].edge_attr)}

        embeddings = self._process(
            x_dict,
            edge_index_dict,
            edge_state_dict,
            batch_dict,
            num_graphs,
        )

        bus_out = self.node_decoders["bus"](embeddings["bus"])
        gen_out = (
            self.node_decoders["gen"](embeddings["gen"])
            if "gen" in self.node_decoders
            else None
        )
        decode = (
            self._decode_physics if self.use_physics_decoder else self._decode_bounded
        )
        predictions = decode(batch, bus_out, gen_out, gen_bus)

        if not return_embeddings:
            return predictions
        return predictions, embeddings
