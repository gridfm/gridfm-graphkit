import torch
from torch import nn
from torch_geometric.nn import HeteroConv, TransformerConv
from gridfm_graphkit.io.registries import MODELS_REGISTRY
from gridfm_graphkit.io.param_handler import get_physics_decoder
from torch_scatter import scatter_add
from gridfm_graphkit.models.utils import (
    ComputeBranchFlow,
    ComputeNodeInjection,
    ComputeNodeResiduals,
    bound_with_sigmoid,
)
from gridfm_graphkit.datasets.globals import (
    # Bus feature indices
    VM_H,
    VA_H,
    MIN_VM_H,
    MAX_VM_H,
    # Output feature indices
    VM_OUT,
    PG_OUT_GEN,
    # Generator feature indices
    PG_H,
    MIN_PG,
    MAX_PG,
)


@MODELS_REGISTRY.register("GNS_heterogeneous")
class GNS_heterogeneous(nn.Module):
    """
    Heterogeneous version of your Transformer-based GNN for buses and generators.
    - Expects node features as dict: x_dict = {"bus": Tensor[num_bus, bus_feat], "gen": Tensor[num_gen, gen_feat]}
    - Expects edge_index_dict and edge_attr_dict with keys:
        ("bus","connects","bus"), ("gen","connected_to","bus"), ("bus","connected_to","gen")
      (edge_attr only needed for bus-bus currently; other relations can be None)
    - Keeps the physics residual idea but splits it into bus-step and gen-step residuals.
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.num_layers = args.model.num_layers
        self.hidden_dim = args.model.hidden_size
        self.input_bus_dim = args.model.input_bus_dim
        self.input_gen_dim = args.model.input_gen_dim
        self.output_bus_dim = args.model.output_bus_dim
        self.output_gen_dim = args.model.output_gen_dim
        self.edge_dim = args.model.edge_dim
        self.heads = args.model.attention_head
        self.task = args.task.task_name
        self.dropout = getattr(args.model, "dropout", 0.0)

        # projections for each node type
        self.input_proj_bus = nn.Sequential(
            nn.Linear(self.input_bus_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        self.input_proj_gen = nn.Sequential(
            nn.Linear(self.input_gen_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        self.input_proj_edge = nn.Sequential(
            nn.Linear(self.edge_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        # a small physics MLP that will take residuals (real, imag) and return a correction
        self.physics_mlp = nn.Sequential(
            nn.Linear(2, self.hidden_dim * self.heads),
            nn.LeakyReLU(),
        )

        # Build hetero layers: HeteroConv of TransformerConv per relation
        self.layers = nn.ModuleList()
        self.norms_bus = nn.ModuleList()
        self.norms_gen = nn.ModuleList()
        for i in range(self.num_layers):
            # in-channels depend on whether it is first layer (hidden_dim) or subsequent (hidden_dim * heads)
            in_bus = self.hidden_dim if i == 0 else self.hidden_dim * self.heads
            in_gen = self.hidden_dim if i == 0 else self.hidden_dim * self.heads
            out_dim = self.hidden_dim  # TransformerConv will output hidden_dim (per head reduction in HeteroConv call)

            # relation -> conv module mapping
            conv_dict = {
                ("bus", "connects", "bus"): TransformerConv(
                    in_bus,
                    out_dim,
                    heads=self.heads,
                    edge_dim=self.hidden_dim,
                    dropout=self.dropout,
                    beta=True,
                ),
                ("gen", "connected_to", "bus"): TransformerConv(
                    in_gen,
                    out_dim,
                    heads=self.heads,
                    dropout=self.dropout,
                    beta=True,
                ),
                ("bus", "connected_to", "gen"): TransformerConv(
                    in_bus,
                    out_dim,
                    heads=self.heads,
                    dropout=self.dropout,
                    beta=True,
                ),
            }

            hetero_conv = HeteroConv(conv_dict, aggr="sum")
            self.layers.append(hetero_conv)

            # Norms for node representations (note: after HeteroConv each node type will have size out_dim * heads)
            self.norms_bus.append(nn.LayerNorm(out_dim * self.heads))
            self.norms_gen.append(nn.LayerNorm(out_dim * self.heads))

        # Separate shared MLPs to produce final bus/gen outputs (predictions y)
        self.mlp_bus = nn.Sequential(
            nn.Linear(self.hidden_dim * self.heads, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.output_bus_dim),
        )

        self.mlp_gen = nn.Sequential(
            nn.Linear(self.hidden_dim * self.heads, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.output_gen_dim),
        )

        # mask param (kept similar to your original)
        self.activation = nn.LeakyReLU()
        self.branch_flow_layer = ComputeBranchFlow()
        self.node_injection_layer = ComputeNodeInjection()
        self.node_residuals_layer = ComputeNodeResiduals()
        self.physics_decoder = get_physics_decoder(args)

        # In StateEstimation, the gen output head, physics_mlp, and the gen
        # branch of the final hetero conv layer never contribute to the loss.
        # Freeze those parameters so DDP doesn't reject them as unused. The
        # modules stay on the model so existing checkpoints still load.
        if self.task == "StateEstimation":
            for p in self.mlp_gen.parameters():
                p.requires_grad = False
            for p in self.physics_mlp.parameters():
                p.requires_grad = False
            last = self.num_layers - 1
            for p in self.norms_gen[last].parameters():
                p.requires_grad = False
            last_conv = self.layers[last].convs[("bus", "connected_to", "gen")]
            for p in last_conv.parameters():
                p.requires_grad = False

        # container for monitoring residual norms per layer and type
        self.layer_residuals = {}

    def forward(self, batch, return_embeddings: bool = False):
        """
        Accepts a PyG HeteroData batch and extracts the required tensors.

        batch: HeteroData/Batch containing:
            x_dict: {"bus": Tensor[num_bus, bus_feat], "gen": Tensor[num_gen, gen_feat]}
            edge_index_dict: keys like ("bus","connects","bus"), ("gen","connected_to","bus"), ("bus","connected_to","gen")
            edge_attr_dict: same keys -> edge attributes (bus-bus requires G,B)
            mask_dict: dict mapping node/bus types to mask tensors

        When ``return_embeddings`` is ``True``, returns ``(predictions, embeddings)``
        instead of ``predictions`` alone. Each embedding is the latent tensor fed
        directly into the corresponding prediction head: ``embeddings["bus"]`` is
        the ``h_bus`` consumed by ``mlp_bus`` and ``embeddings["gen"]`` is the
        ``h_gen`` consumed by ``mlp_gen`` on the final layer. (Note the last-layer
        ``physics_mlp`` update to ``h_bus`` is intentionally skipped, so the
        returned bus embedding is exactly the head input, not a post-update state.)
        """
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict
        mask_dict = batch.mask_dict

        self.layer_residuals = {}

        # 1) initial projections
        h_bus = self.input_proj_bus(x_dict["bus"])  # [num_bus, hidden_dim]
        h_gen = self.input_proj_gen(x_dict["gen"])  # [num_gen, hidden_dim]

        num_bus = x_dict["bus"].size(0)
        _, gen_to_bus_index = edge_index_dict[("gen", "connected_to", "bus")]
        bus_edge_index = edge_index_dict[("bus", "connects", "bus")]
        bus_edge_attr = edge_attr_dict[("bus", "connects", "bus")]

        edge_attr_proj_dict = {}
        for key, edge_attr in edge_attr_dict.items():
            if edge_attr is not None:
                edge_attr_proj_dict[key] = self.input_proj_edge(edge_attr)
            else:
                edge_attr_proj_dict[key] = None

        bus_mask = mask_dict["bus"][:, VM_H : VA_H + 1]
        gen_mask = mask_dict["gen"][:, : (PG_H + 1)]
        bus_fixed = x_dict["bus"][:, VM_H : VA_H + 1]
        gen_fixed = x_dict["gen"][:, : (PG_H + 1)]

        # iterate layers
        for i, conv in enumerate(self.layers):
            out_dict = conv(
                {"bus": h_bus, "gen": h_gen},
                edge_index_dict,
                edge_attr_proj_dict,
            )
            out_bus = out_dict["bus"]  # [Nb, hidden_dim * heads]
            out_gen = out_dict["gen"]  # [Ng, hidden_dim * heads]

            out_bus = self.activation(self.norms_bus[i](out_bus))
            out_gen = self.activation(self.norms_gen[i](out_gen))

            # skip connection
            h_bus = h_bus + out_bus if out_bus.shape == h_bus.shape else out_bus
            h_gen = h_gen + out_gen if out_gen.shape == h_gen.shape else out_gen

            # Decode bus and generator predictions
            bus_temp = self.mlp_bus(h_bus)  # [Nb, 2]  -> Vm, Va
            gen_temp = self.mlp_gen(h_gen)  # [Ng, 1]  -> Pg

            if self.task == "StateEstimation":
                if i == self.num_layers - 1:
                    Pft, Qft = self.branch_flow_layer(
                        bus_temp,
                        bus_edge_index,
                        bus_edge_attr,
                    )
                    P_in, Q_in = self.node_injection_layer(
                        Pft,
                        Qft,
                        bus_edge_index,
                        num_bus,
                    )
                    output_temp = self.physics_decoder(
                        P_in,
                        Q_in,
                        bus_temp,
                        x_dict["bus"],
                        None,
                        None,
                    )

            else:
                bus_temp = torch.where(bus_mask, bus_temp, bus_fixed)
                gen_temp = torch.where(gen_mask, gen_temp, gen_fixed)

                if self.task == "OptimalPowerFlow":
                    bus_temp[:, VM_OUT] = bound_with_sigmoid(
                        bus_temp[:, VM_OUT],
                        x_dict["bus"][:, MIN_VM_H],
                        x_dict["bus"][:, MAX_VM_H],
                    )
                    gen_temp[:, PG_OUT_GEN] = bound_with_sigmoid(
                        gen_temp[:, PG_OUT_GEN],
                        x_dict["gen"][:, MIN_PG],
                        x_dict["gen"][:, MAX_PG],
                    )

                Pft, Qft = self.branch_flow_layer(
                    bus_temp,
                    bus_edge_index,
                    bus_edge_attr,
                )
                P_in, Q_in = self.node_injection_layer(
                    Pft,
                    Qft,
                    bus_edge_index,
                    num_bus,
                )
                agg_bus = scatter_add(
                    gen_temp.squeeze(),
                    gen_to_bus_index,
                    dim=0,
                    dim_size=num_bus,
                )
                output_temp = self.physics_decoder(
                    P_in,
                    Q_in,
                    bus_temp,
                    x_dict["bus"],
                    agg_bus,
                    mask_dict,
                )
                residual_P, residual_Q = self.node_residuals_layer(
                    P_in,
                    Q_in,
                    output_temp,
                    x_dict["bus"],
                )

                bus_residuals = torch.stack([residual_P, residual_Q], dim=-1)

                # Save and project residuals to latent space.
                # layer_residuals is consumed by LayeredWeightedPhysicsLoss for
                # every layer, so it is recorded unconditionally.
                self.layer_residuals[i] = torch.linalg.norm(
                    bus_residuals,
                    dim=-1,
                ).mean()
                # On the last layer the updated h_bus is never read again: the
                # loop ends, predictions come from output_temp/gen_temp, and the
                # exported embedding is the tensor that fed mlp_bus. Skipping the
                # update avoids a dead forward pass and makes h_bus itself the
                # correct embedding to return (no snapshot/clone needed).
                if i < self.num_layers - 1:
                    h_bus = h_bus + self.physics_mlp(bus_residuals)

        predictions = {"bus": output_temp, "gen": gen_temp}
        if not return_embeddings:
            return predictions
        return predictions, {"bus": h_bus, "gen": h_gen}
