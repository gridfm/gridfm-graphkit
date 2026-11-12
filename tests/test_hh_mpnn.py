"""Tests for the HH-MPNN model."""

import copy
import math
import sys
from argparse import Namespace
from unittest import mock

import pytest
import torch
import torch.nn as nn
import torch_scatter
import yaml
from torch_geometric.data import Batch
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.utils import to_dense_batch

from gridfm_graphkit.__main__ import main
from gridfm_graphkit.cli import main_cli
from gridfm_graphkit.datasets.effective_resistance import ER_PE_ATTR
from gridfm_graphkit.datasets.globals import (
    BS,
    GS,
    MAX_PG,
    MAX_QG_H,
    MAX_VM_H,
    MIN_PG,
    MIN_QG_H,
    MIN_VM_H,
    PD_H,
    PG_OUT,
    QD_H,
    QG_H,
    QG_OUT,
    VA_H,
    VA_OUT,
    VM_OUT,
)
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.io.param_handler import (
    NestedNamespace,
    get_loss_function,
    get_task,
    load_model,
)
from gridfm_graphkit.models.hh_mpnn import (
    BRANCH,
    GEN_TO_BUS,
    LOAD_TO_BUS,
    SHUNT_TO_BUS,
    VA_BOUND,
    HHMPNN,
    MaskedPerformerAttention,
)

CONFIG_PATH = "tests/config/hh_mpnn_test_config.yaml"
SE_CONFIG_PATH = "tests/config/datamodule_test_base_config3.yaml"

# Every supported (task, load_shunt_nodes, physics_decoder) combination.
MODES = [
    ("OptimalPowerFlow", True, False),
    ("OptimalPowerFlow", False, False),
    ("OptimalPowerFlow", True, True),
    ("OptimalPowerFlow", False, True),
    ("PowerFlow", True, True),
    ("PowerFlow", False, True),
    ("StateEstimation", True, True),
    ("StateEstimation", False, True),
]

_MASKING = {
    "OptimalPowerFlow": "AddOPFHeteroMask",
    "PowerFlow": {"name": "AddPFHeteroMask", "mask_ref_vm": False},
    "StateEstimation": "SimulateMeasurements",
}


def make_config(task="OptimalPowerFlow", load_shunt_nodes=True, physics_decoder=False):
    """The HH-MPNN test config, adapted to a task and a model mode."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["load_shunt_nodes"] = load_shunt_nodes
    cfg["model"]["physics_decoder"] = physics_decoder
    cfg["data"]["transforms"] = [
        "RemoveInactiveBranches",
        "RemoveInactiveGenerators",
        _MASKING[task],
        "ApplyMasking",
        "EffectiveResistancePE",
    ]
    if task == "StateEstimation":
        with open(SE_CONFIG_PATH) as f:
            se_cfg = yaml.safe_load(f)
        cfg["task"] = se_cfg["task"]
        for key in ("losses", "loss_weights", "loss_args"):
            cfg["training"][key] = se_cfg["training"][key]
    cfg["task"]["task_name"] = task
    return cfg


def make_batch(cfg, num_graphs=8):
    """Build the datamodule for a config and return its args and one batch."""
    args = NestedNamespace(**cfg)
    dm = LitGridHeteroDataModule(args, data_dir="tests/data")

    class DummyTrainer:
        is_global_zero = True
        logger = None

    dm.trainer = DummyTrainer()
    dm.setup("fit")
    dataset = dm.datasets[0]
    return args, Batch.from_data_list([dataset[i] for i in range(num_graphs)])


@pytest.fixture(scope="module")
def opf():
    """Args, batch and model for the default (paper) configuration."""
    args, batch = make_batch(make_config())
    torch.manual_seed(0)
    return args, batch, load_model(args).eval()


# --------------------------------------------------------------------------- #
# Output contract, in every mode
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task,load_shunt_nodes,physics_decoder", MODES)
def test_forward_and_gradients(task, load_shunt_nodes, physics_decoder):
    """Outputs have the task's shape and every parameter gets a gradient.

    A parameter without gradient would make DDP fail, because the CLI runs it
    with ``find_unused_parameters=False``.
    """
    args, batch = make_batch(make_config(task, load_shunt_nodes, physics_decoder))
    torch.manual_seed(0)
    model = load_model(args)
    assert isinstance(model, HHMPNN)

    output = model(batch)
    assert output["bus"].shape == (batch["bus"].num_nodes, 4)
    assert output["gen"].shape == (batch["gen"].num_nodes, 1)
    assert torch.isfinite(output["bus"]).all() and torch.isfinite(output["gen"]).all()

    loss = get_loss_function(args)(
        output,
        batch.y_dict,
        batch.edge_index_dict,
        batch.edge_attr_dict,
        batch.mask_dict,
        model=model,
        x_dict=batch.x_dict,
    )["loss"]
    loss.backward()
    without_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert without_grad == []


@pytest.mark.parametrize("task", ["PowerFlow", "StateEstimation"])
def test_bounded_heads_reject_other_tasks(task):
    args = NestedNamespace(**make_config(task, physics_decoder=False))
    with pytest.raises(ValueError, match="physics_decoder"):
        load_model(args)


def test_missing_positional_encoding_is_reported(opf):
    _, batch, model = opf
    batch = copy.copy(batch)
    del batch["bus"][ER_PE_ATTR]
    with pytest.raises(ValueError, match="EffectiveResistancePE"):
        model(batch)


def test_return_embeddings(opf):
    args, batch, model = opf
    predictions, embeddings = model(batch, return_embeddings=True)
    assert embeddings["bus"].shape == (batch["bus"].num_nodes, args.model.hidden_size)
    assert embeddings["gen"].shape == (batch["gen"].num_nodes, args.model.hidden_size)
    torch.testing.assert_close(predictions["bus"], model(batch)["bus"])


def test_runs_in_bfloat16(opf):
    """The CLI's --bfloat16 flag casts model and batch to bfloat16."""
    _, batch, model = opf
    model = copy.deepcopy(model).to(torch.bfloat16)
    batch = copy.copy(batch)
    for store in batch.stores:
        for key, value in store.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                store[key] = value.to(torch.bfloat16)
    output = model(batch)
    assert output["bus"].dtype == torch.bfloat16
    assert torch.isfinite(output["bus"].float()).all()


# --------------------------------------------------------------------------- #
# Bounded output heads (the paper's decoding)
# --------------------------------------------------------------------------- #
def _saturate_decoders(model):
    """Make the output heads produce large values of both signs."""
    for decoder in model.node_decoders.values():
        nn.init.normal_(decoder.network[-1].weight, std=50.0)


def test_bounded_outputs_stay_within_limits(opf):
    _, batch, model = opf
    model = copy.deepcopy(model)
    _saturate_decoders(model)
    output = model(batch)
    bus_x, gen_x = batch.x_dict["bus"], batch.x_dict["gen"]
    eps = 1e-6

    vm, va = output["bus"][:, VM_OUT], output["bus"][:, VA_OUT]
    qg, pg = output["bus"][:, QG_OUT], output["gen"][:, 0]
    assert (vm >= bus_x[:, MIN_VM_H] - eps).all() and (
        vm <= bus_x[:, MAX_VM_H] + eps
    ).all()
    assert (qg >= bus_x[:, MIN_QG_H] - eps).all() and (
        qg <= bus_x[:, MAX_QG_H] + eps
    ).all()
    assert (pg >= gen_x[:, MIN_PG] - eps).all() and (pg <= gen_x[:, MAX_PG] + eps).all()
    assert (va.abs() <= VA_BOUND + eps).all()
    # The saturated heads really do reach both ends of the ranges.
    assert (vm - bus_x[:, MIN_VM_H]).abs().min() < 1e-3
    assert (vm - bus_x[:, MAX_VM_H]).abs().min() < 1e-3


def test_task_inputs_are_passed_through(opf):
    """Unmasked quantities keep their input value, e.g. the slack angle."""
    _, batch, model = opf
    output = model(batch)
    bus_x, bus_mask = batch.x_dict["bus"], batch.mask_dict["bus"]
    ref = batch.mask_dict["REF"]
    assert ref.any() and not bus_mask[ref, VA_H].any()
    torch.testing.assert_close(output["bus"][ref, VA_OUT], bus_x[ref, VA_H])
    known_qg = ~bus_mask[:, QG_H]
    torch.testing.assert_close(output["bus"][known_qg, QG_OUT], bus_x[known_qg, QG_H])


def test_bus_pg_is_the_sum_of_its_generators(opf):
    _, batch, model = opf
    output = model(batch)
    gen_index, bus_index = batch[GEN_TO_BUS].edge_index
    expected = torch.zeros(batch["bus"].num_nodes).index_add(
        0,
        bus_index,
        output["gen"][gen_index, 0],
    )
    torch.testing.assert_close(output["bus"][:, PG_OUT], expected)


def test_generator_order_does_not_matter(opf):
    """Shuffling the gen-bus connectors leaves the predictions unchanged."""
    _, batch, model = opf
    shuffled = copy.copy(batch)
    perm = torch.randperm(batch[GEN_TO_BUS].edge_index.size(1))
    shuffled[GEN_TO_BUS].edge_index = batch[GEN_TO_BUS].edge_index[:, perm]
    torch.testing.assert_close(model(shuffled)["bus"], model(batch)["bus"])


# --------------------------------------------------------------------------- #
# Physics decoder mode
# --------------------------------------------------------------------------- #
def test_physics_decoder_closes_reactive_balance():
    """With the physics decoder, Qg at PV and REF buses balances exactly."""
    from gridfm_graphkit.models.utils import (
        ComputeBranchFlow,
        ComputeNodeInjection,
        ComputeNodeResiduals,
    )

    args, batch = make_batch(make_config(physics_decoder=True))
    torch.manual_seed(0)
    output = load_model(args).eval()(batch)

    p_flow, q_flow = ComputeBranchFlow()(
        output["bus"],
        batch[BRANCH].edge_index,
        batch[BRANCH].edge_attr,
    )
    p_in, q_in = ComputeNodeInjection()(
        p_flow,
        q_flow,
        batch[BRANCH].edge_index,
        batch["bus"].num_nodes,
    )
    _, residual_q = ComputeNodeResiduals()(
        p_in,
        q_in,
        output["bus"],
        batch.x_dict["bus"],
    )
    pv_ref = batch.mask_dict["PV"] | batch.mask_dict["REF"]
    assert residual_q[pv_ref].abs().max() < 1e-5


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def test_load_and_shunt_nodes_follow_the_bus_columns(opf):
    _, batch, model = opf
    x_dict, pe_dict, batch_dict, edge_index_dict, _, _ = model._build_graph(batch)
    bus_x = batch.x_dict["bus"]

    has_load = (bus_x[:, [PD_H, QD_H]] != 0).any(dim=1)
    has_shunt = (bus_x[:, [GS, BS]] != 0).any(dim=1)
    assert 0 < has_load.sum() < bus_x.size(0)
    assert has_shunt.any()

    torch.testing.assert_close(x_dict["load"], bus_x[has_load][:, [PD_H, QD_H]])
    torch.testing.assert_close(x_dict["shunt"], bus_x[has_shunt][:, [GS, BS]])
    assert (
        edge_index_dict[LOAD_TO_BUS][1].tolist()
        == has_load.nonzero().flatten().tolist()
    )
    assert (
        edge_index_dict[SHUNT_TO_BUS][1].tolist()
        == has_shunt.nonzero().flatten().tolist()
    )

    # Load and shunt columns leave the bus; the nodes inherit PE and graph id.
    assert x_dict["bus"].size(1) == bus_x.size(1) - 4
    torch.testing.assert_close(pe_dict["load"], batch["bus"][ER_PE_ATTR][has_load])
    assert batch_dict["shunt"].tolist() == batch["bus"].batch[has_shunt].tolist()
    assert batch_dict["gen"].tolist() == batch["gen"].batch.tolist()


def test_native_graph_keeps_all_bus_columns():
    args, batch = make_batch(make_config(load_shunt_nodes=False), num_graphs=2)
    model = load_model(args)
    x_dict, _, _, edge_index_dict, _, _ = model._build_graph(batch)
    assert set(x_dict) == {"bus", "gen"}
    assert set(edge_index_dict) == {BRANCH, GEN_TO_BUS}
    torch.testing.assert_close(x_dict["bus"], batch.x_dict["bus"])


def test_batch_without_loads_or_shunts(opf):
    _, batch, model = opf
    batch = copy.copy(batch)
    batch["bus"].x = batch["bus"].x.clone()
    batch["bus"].x[:, [PD_H, QD_H, GS, BS]] = 0
    x_dict = model._build_graph(batch)[0]
    assert x_dict["load"].size(0) == 0 and x_dict["shunt"].size(0) == 0
    assert torch.isfinite(model(batch)["bus"]).all()


def test_single_graph_without_batch_vector(opf):
    """A plain HeteroData (no batch vector) gives the same result as a batch."""
    _, batch, model = opf
    single = batch.get_example(0)
    expected = model(Batch.from_data_list([single]))
    torch.testing.assert_close(model(single)["bus"], expected["bus"])


@pytest.mark.parametrize("load_shunt_nodes", [True, False])
def test_prediction_is_independent_of_the_rest_of_the_batch(load_shunt_nodes):
    """Graphs of different sizes in one batch do not influence each other."""
    args, batch = make_batch(make_config(load_shunt_nodes=load_shunt_nodes), 12)
    torch.manual_seed(0)
    model = load_model(args).eval()
    graphs = [batch.get_example(i) for i in range(12)]
    # Graph sizes differ, so the dense attention batch is padded.
    assert len({graph["gen"].num_nodes for graph in graphs}) > 1

    together = model(batch)["bus"]
    bus_ptr = batch["bus"].ptr
    for i, graph in enumerate(graphs):
        alone = model(Batch.from_data_list([graph]))["bus"]
        torch.testing.assert_close(
            together[bus_ptr[i] : bus_ptr[i + 1]],
            alone,
            atol=1e-5,
            rtol=0,
        )


def test_masked_attention_ignores_padding_and_matches_pyg_without_it():
    torch.manual_seed(0)
    reference = PerformerAttention(channels=32, heads=2).eval()
    attention = MaskedPerformerAttention(channels=32, heads=2).eval()
    attention.load_state_dict(reference.state_dict())

    x = torch.randn(2, 9, 32)
    mask = torch.ones(2, 9, dtype=torch.bool)
    torch.testing.assert_close(attention(x, mask), reference(x.clone(), mask))

    padded = torch.cat([x, torch.zeros(2, 50, 32)], dim=1)
    padded_mask = torch.cat([mask, torch.zeros(2, 50, dtype=torch.bool)], dim=1)
    torch.testing.assert_close(
        attention(padded, padded_mask)[:, :9],
        attention(x, mask),
    )


# --------------------------------------------------------------------------- #
# Equivalence with the original HH-MPNN training script
#
# The three classes below are the layers of the script the model was ported
# from, kept verbatim apart from formatting. With the same weights, the ported
# layers must produce the same node states.
# --------------------------------------------------------------------------- #
class _OriginalMLP(nn.Module):
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
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0, 1 / math.sqrt(layer.in_features))
                layer.bias.data.fill_(0)

    def forward(self, x):
        return self.network(x)


class _OriginalPerformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=1):
        super().__init__()
        self.attn = PerformerAttention(channels=hidden_dim, heads=num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x_dict, xm_dict, batch_dict):
        flat_x, flat_xm, flat_batch = [], [], []
        slices = {}
        offset = 0
        for ntype in x_dict:
            x = x_dict[ntype]
            slices[ntype] = slice(offset, offset + x.size(0))
            flat_x.append(x)
            flat_xm.append(xm_dict[ntype])
            flat_batch.append(batch_dict[ntype])
            offset += x.size(0)

        x_all = torch.cat(flat_x, dim=0)
        xm_all = torch.cat(flat_xm, dim=0)
        global_batch_all = torch.cat(flat_batch, dim=0)

        _, batch_all = torch.unique(global_batch_all, return_inverse=True)
        sorted_indices = torch.argsort(batch_all)
        batch_all_sorted = batch_all[sorted_indices]
        x_all_sorted = x_all[sorted_indices]
        xm_all_sorted = xm_all[sorted_indices]
        x_dense, mask = to_dense_batch(x_all_sorted, batch_all_sorted)

        x_attn = self.attn(x_dense, mask=mask)
        xt_out = self.norm1(x_dense + x_attn)
        xt_out = xt_out[mask]
        x_comb = self.norm2(xt_out + xm_all_sorted)
        x_final = self.mlp(x_comb)

        unsorted_x = torch.empty_like(x_final)
        unsorted_x[sorted_indices] = x_final
        return {ntype: unsorted_x[slices[ntype]] for ntype in x_dict.keys()}


class _OriginalInteractionNetwork(nn.Module):
    def __init__(
        self,
        node_types,
        edge_types,
        physical_edge_types,
        hidden_size,
        layers,
    ):
        super().__init__()
        self.physical_edge_types = physical_edge_types
        self.edge_updaters = nn.ModuleDict()
        for src, rel, dst in edge_types:
            key = f"{src}_{rel}_{dst}"
            edge_type = (src, rel, dst)
            in_dim = (
                hidden_size * 3 if edge_type in physical_edge_types else hidden_size * 2
            )
            self.edge_updaters[key] = _OriginalMLP(
                in_dim,
                hidden_size,
                hidden_size,
                layers,
            )
        self.node_updaters = nn.ModuleDict(
            {
                nt: _OriginalMLP(hidden_size * 2, hidden_size, hidden_size, layers)
                for nt in node_types
            },
        )

    def forward(self, x_dict, edge_indices_dict, edge_features_dict):
        updated_edge_features = {}
        aggregated_messages = {
            nt: torch.zeros_like(feat) for nt, feat in x_dict.items()
        }

        for edge_type, edge_index in edge_indices_dict.items():
            src_type, rel_type, dst_type = edge_type
            edge_key = f"{src_type}_{rel_type}_{dst_type}"
            src, dst = edge_index
            x_i = x_dict[dst_type][dst]
            x_j = x_dict[src_type][src]

            if edge_type in self.physical_edge_types:
                edge_feature = edge_features_dict[edge_type]
                updated_edge = self.edge_updaters[edge_key](
                    torch.cat((x_i, x_j, edge_feature), dim=-1),
                )
                updated_edge = updated_edge + edge_feature
                updated_edge_features[edge_type] = updated_edge
            else:
                updated_edge = self.edge_updaters[edge_key](
                    torch.cat((x_i, x_j), dim=-1),
                )
                updated_edge_features[edge_type] = updated_edge

            aggregated_messages[dst_type] = torch_scatter.scatter_add(
                updated_edge,
                dst,
                dim=0,
                out=aggregated_messages[dst_type],
            )
            aggregated_messages[src_type] = torch_scatter.scatter_add(
                updated_edge,
                src,
                dim=0,
                out=aggregated_messages[src_type],
            )

        updated_nodes = {}
        for nt, x in x_dict.items():
            node_update = self.node_updaters[nt](
                torch.cat((x, aggregated_messages[nt]), dim=-1),
            )
            updated_nodes[nt] = x + node_update
        return updated_nodes, updated_edge_features


def _copy_layer_weights(model, interactions, attentions):
    """Load the ported layers' weights into the original layers."""
    for ported, original in zip(model.layers, interactions):
        state = {}
        for name, value in ported.state_dict().items():
            # Edge-type keys are joined with "__" in the port, "_" originally.
            state[name.replace("__", "_")] = value
        # The port has no last-layer updaters for node types without an output
        # head; the original keeps them, and their result is never used.
        missing, unexpected = original.load_state_dict(state, strict=False)
        assert not unexpected
        assert all(name.startswith("node_updaters.") for name in missing)
    for ported, original in zip(model.global_attn_layers, attentions):
        original.load_state_dict(ported.state_dict())


def _graphs_of_equal_size(batch, model):
    """The largest group of graphs in a batch with identical node counts."""
    groups = {}
    for i in range(batch.num_graphs):
        graph = batch.get_example(i)
        sizes = tuple(x.size(0) for x in model._build_graph(graph)[0].values())
        groups.setdefault(sizes, []).append(graph)
    return Batch.from_data_list(max(groups.values(), key=len))


def test_layers_match_the_original_implementation(opf):
    args, batch, model = opf
    hidden, num_layers = args.model.hidden_size, args.model.num_layers
    # Equal-sized graphs need no padding, which is the one place where the
    # port deliberately differs from the original (MaskedPerformerAttention).
    batch = _graphs_of_equal_size(batch, model)
    assert batch.num_graphs > 1
    x_dict, pe_dict, batch_dict, edge_index_dict, _, num_graphs = model._build_graph(
        batch,
    )
    node_types, edge_types = list(x_dict), list(edge_index_dict)
    assert node_types == ["bus", "gen", "load", "shunt"]

    torch.manual_seed(1)
    interactions = [
        _OriginalInteractionNetwork(node_types, edge_types, [BRANCH], hidden, 2)
        for _ in range(num_layers)
    ]
    attentions = [_OriginalPerformerLayer(hidden) for _ in range(num_layers)]
    _copy_layer_weights(model, interactions, attentions)

    with torch.no_grad():
        states = {
            node_type: torch.cat(
                [model.node_encoders[node_type](x), pe_dict[node_type]],
                dim=-1,
            )
            for node_type, x in x_dict.items()
        }
        edge_states = {BRANCH: model.edge_encoder(batch[BRANCH].edge_attr)}
        ported = model._process(
            states,
            edge_index_dict,
            edge_states,
            batch_dict,
            num_graphs,
        )

        original = states
        for interaction, attention in zip(interactions, attentions):
            xm_dict, edge_states = interaction(original, edge_index_dict, edge_states)
            original = attention(original, xm_dict, batch_dict)

    for node_type in ("bus", "gen"):
        torch.testing.assert_close(
            ported[node_type],
            original[node_type],
            atol=1e-5,
            rtol=1e-5,
        )


# --------------------------------------------------------------------------- #
# End to end through the CLI
# --------------------------------------------------------------------------- #
def _cli_args(command, config, tmp_path, **extra):
    return Namespace(
        command=command,
        config=str(config),
        data_path="tests/data",
        log_dir=str(tmp_path / "mlruns"),
        exp_name="pytest_hh_mpnn",
        run_name="pytest_run",
        **extra,
    )


@pytest.mark.parametrize("task,load_shunt_nodes,physics_decoder", MODES)
def test_cli_train(tmp_path, task, load_shunt_nodes, physics_decoder):
    """`train` runs fit and test for one epoch in every mode."""
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(make_config(task, load_shunt_nodes, physics_decoder)),
    )
    main_cli(_cli_args("train", config, tmp_path))


def test_cli_evaluate_and_predict_from_saved_weights(tmp_path):
    cfg = make_config()
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    task = get_task(NestedNamespace(**cfg), data_normalizers=[])
    model_path = tmp_path / "model.pt"
    torch.save(task.state_dict(), model_path)

    main_cli(_cli_args("evaluate", config, tmp_path, model_path=str(model_path)))

    output_path = tmp_path / "predictions"
    argv = [
        "gridfm_graphkit",
        "predict",
        "--config",
        str(config),
        "--data_path",
        "tests/data",
        "--log_dir",
        str(tmp_path / "mlruns"),
        "--model_path",
        str(model_path),
        "--output_path",
        str(output_path),
        "--get_embeddings",
    ]
    with mock.patch.object(sys, "argv", argv):
        main()
    written = {p.name for p in output_path.iterdir()}
    assert "case14_ieee_predictions.parquet" in written
    assert "case14_ieee_bus_embeddings.parquet" in written
    assert "case14_ieee_gen_embeddings.parquet" in written


def test_example_config_uses_the_paper_setup():
    """The shipped example enables the paper's graph and output heads."""
    with open("examples/config/HHMPNN_OPF_datakit_case14.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["model"]["type"] == "HH_MPNN"
    assert cfg["model"]["load_shunt_nodes"] is True
    assert cfg["model"]["physics_decoder"] is False
    assert "EffectiveResistancePE" in cfg["data"]["transforms"]
    assert cfg["data"]["transforms"].index("RemoveInactiveBranches") == 0
