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
    compute_shunt_power,
)
from gridfm_graphkit.datasets.globals import (
    # Bus feature indices
    VM_H,
    VA_H,
    MIN_VM_H,
    MAX_VM_H,
    # Output feature indices
    VM_OUT,
    VA_OUT,
    PG_OUT_GEN,
    # Generator feature indices
    PG_H,
    MIN_PG,
    MAX_PG,
)


@MODELS_REGISTRY.register("GNS_heterogeneous")
class GNS_heterogeneous(nn.Module):
    """
    Heterogeneous Transformer-based GNN for buses and generators.
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

        # StateEstimation only: also predict branch r, x.
        self.parameter_estimation = getattr(args.task, "parameter_estimation", False)

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

        self.activation = nn.LeakyReLU()
        self.branch_flow_layer = ComputeBranchFlow()
        self.node_injection_layer = ComputeNodeInjection()
        self.node_residuals_layer = ComputeNodeResiduals()
        self.physics_decoder = get_physics_decoder(args)

        # container for monitoring residual norms per layer and type
        self.layer_residuals = {}

        if self.parameter_estimation:
            # Outputs r, x, bc per branch. Only r and x are evaluated.
            self.mlp_impedance = nn.Sequential(
                nn.Linear(
                    2 * self.hidden_dim * self.heads + 2 * self.hidden_dim,
                    self.hidden_dim,
                ),
                nn.LayerNorm(self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, 3),
            )

    def forward(
        self,
        x_dict,
        edge_index_dict,
        edge_attr_dict,
        mask_dict,
        batch_dict=None,
    ):
        """
        x_dict: {"bus": Tensor[num_bus, bus_feat], "gen": Tensor[num_gen, gen_feat]}
        edge_index_dict: keys like ("bus","connects","bus"), ("gen","connected_to","bus"), ("bus","connected_to","gen")
        edge_attr_dict: same keys -> edge attributes (bus-bus requires G,B)
        batch_dict: graph index of every node, per node type. Needed only for parameter estimation.
        mask: optional mask per node (applies when computing residuals)
        """

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

            if self.task != "StateEstimation":
                bus_temp = torch.where(bus_mask, bus_temp, bus_fixed)
                gen_temp = torch.where(gen_mask, gen_temp, gen_fixed)
            else:
                # SE predicts voltages only. Generator Pg stays at its measured value.
                gen_temp = gen_fixed.detach().clone()

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

            # Save and project residuals to latent space
            self.layer_residuals[i] = torch.linalg.norm(
                bus_residuals,
                dim=-1,
            ).mean()
            h_bus = h_bus + self.physics_mlp(bus_residuals)

        if self.task == "StateEstimation":
            # Final output: voltages and the injections and flows they imply.
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
            p_shunt, q_shunt = compute_shunt_power(bus_temp, x_dict["bus"])
            Vm_out = bus_temp[:, VM_OUT]
            Va_out = bus_temp[:, VA_OUT]
            output_temp = torch.stack(
                [Vm_out, Va_out, P_in - p_shunt, Q_in - q_shunt],
                dim=1,
            )
            result = {
                "bus": output_temp,
                "gen": gen_temp,
                "branch": torch.stack([Pft, Qft], dim=-1),
            }

            if self.parameter_estimation:
                # One r, x prediction per branch. In each graph the first half of the
                # edges are forward edges and the second half the matching reverse edges.
                edge_proj = edge_attr_proj_dict[("bus", "connects", "bus")]
                device = edge_proj.device
                forward_edge_attr_proj = torch.zeros(
                    int(edge_proj.size(0) / 2),
                    edge_proj.size(1),
                ).to(device)
                reverse_edge_attr_proj = torch.zeros_like(forward_edge_attr_proj).to(
                    device,
                )
                forward_edge_index = torch.zeros(
                    bus_edge_index.size(0),
                    int(bus_edge_index.size(1) / 2),
                    dtype=torch.int32,
                ).to(device)

                current_start = 0
                batch_indices_branch = batch_dict["bus"][bus_edge_index[0]]

                for batch_idx in batch_indices_branch.unique():
                    num_edges = (batch_indices_branch == batch_idx).sum().item()
                    new_start = current_start + num_edges
                    midpoint = int(current_start + num_edges / 2)

                    assert (
                        torch.where(batch_indices_branch == batch_idx)[0].min()
                        == current_start
                    )
                    assert (
                        torch.where(batch_indices_branch == batch_idx)[0].max()
                        == new_start - 1
                    )
                    current_start_out = int(current_start / 2)
                    new_start_out = int(new_start / 2)
                    forward_edge_attr_proj[current_start_out:new_start_out, :] = (
                        edge_proj[current_start:midpoint, :]
                    )
                    reverse_edge_attr_proj[current_start_out:new_start_out, :] = (
                        edge_proj[midpoint:new_start, :]
                    )
                    forward_edge_index[:, current_start_out:new_start_out] = (
                        bus_edge_index[:, current_start:midpoint]
                    )
                    reverse_edge_index = bus_edge_index[:, midpoint:new_start]
                    assert (
                        bus_edge_index[:, current_start:midpoint]
                        == reverse_edge_index[[1, 0], :]
                    ).all()
                    current_start = new_start

                # Edge embeddings come only from the input projection, so the final bus
                # embeddings at both ends are added to the decoder input.
                impedance = self.mlp_impedance(
                    torch.cat(
                        (
                            h_bus[forward_edge_index[0]],
                            h_bus[forward_edge_index[1]],
                            forward_edge_attr_proj,
                            reverse_edge_attr_proj,
                        ),
                        dim=1,
                    ),
                )
                result["impedance"] = impedance  # [num_branches, 3]: r, x, bc

            return result
        return {"bus": output_temp, "gen": gen_temp}
