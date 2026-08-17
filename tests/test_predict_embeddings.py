import numpy as np
import torch
import yaml
from unittest import mock
from types import SimpleNamespace

from gridfm_graphkit.__main__ import main
from gridfm_graphkit.cli import _prediction_output_filename, main_cli
from gridfm_graphkit.models.gnn_heterogeneous_gns import GNS_heterogeneous
from gridfm_graphkit.models.grit_transformer import GritHeteroAdapter
from gridfm_graphkit.tasks.utils import (
    embedding_table_from_tensor,
    local_index_per_graph,
)


def test_prediction_output_filename_handles_embeddings() -> None:
    assert _prediction_output_filename("case14", "bus") == "case14_predictions.parquet"
    assert (
        _prediction_output_filename("case14", "gen") == "case14_gen_predictions.parquet"
    )
    assert (
        _prediction_output_filename("case14", "bus_embeddings")
        == "case14_bus_embeddings.parquet"
    )
    assert (
        _prediction_output_filename("case14", "gen_embeddings")
        == "case14_gen_embeddings.parquet"
    )


def test_local_index_per_graph_builds_local_entity_ids() -> None:
    batch_index = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.long)
    torch.testing.assert_close(
        local_index_per_graph(batch_index),
        torch.tensor([0, 1, 2, 0, 1, 0], dtype=torch.long),
    )


def test_embedding_table_from_tensor_formats_columns() -> None:
    embedding = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    table = embedding_table_from_tensor(
        embedding,
        id_columns={"scenario": np.array([10, 11]), "bus": np.array([0, 1])},
    )
    assert list(table) == ["scenario", "bus", "emb_000", "emb_001"]
    np.testing.assert_allclose(table["emb_000"], np.array([1.0, 3.0]))
    np.testing.assert_allclose(table["emb_001"], np.array([2.0, 4.0]))


def test_grit_adapter_returns_bus_embeddings_when_requested() -> None:
    adapter = object.__new__(GritHeteroAdapter)
    torch.nn.Module.__init__(adapter)
    adapter.grit = SimpleNamespace(
        mask_value=torch.tensor([-1.0], dtype=torch.float32),
        encoder=lambda homo: homo,
        layers=lambda homo: homo,
    )
    adapter.bus_head = torch.nn.Identity()
    adapter.gen_head = torch.nn.Identity()

    batch = {
        "bus": SimpleNamespace(
            x=torch.arange(30, dtype=torch.float32).reshape(2, 15),
            y=torch.zeros((2, 2), dtype=torch.float32),
            batch=torch.tensor([0, 0], dtype=torch.long),
        ),
        "gen": SimpleNamespace(
            x=torch.arange(12, dtype=torch.float32).reshape(2, 6),
        ),
        ("bus", "connects", "bus"): SimpleNamespace(
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.zeros((2, 1), dtype=torch.float32),
        ),
    }
    batch = type("FakeBatch", (dict,), {"edge_types": ()})(batch)

    aggregated_pg = torch.tensor([100.0, 200.0], dtype=torch.float32)
    with mock.patch(
        "gridfm_graphkit.models.grit_transformer.aggregate_pg",
        return_value=aggregated_pg,
    ):
        predictions, embeddings = adapter(batch, return_embeddings=True)

    expected_bus_features = torch.cat(
        [batch["bus"].x, aggregated_pg.unsqueeze(-1)],
        dim=-1,
    )
    torch.testing.assert_close(predictions["bus"], expected_bus_features)
    torch.testing.assert_close(embeddings["bus"], expected_bus_features)
    torch.testing.assert_close(predictions["gen"], batch["gen"].x)


def test_predict_parser_accepts_get_embeddings_flag() -> None:
    test_argv = [
        "gridfm_graphkit",
        "predict",
        "--config",
        "examples/config/HGNS_PF_118Bus.yaml",
        "--model_path",
        "tests/models/dummy_model.pt",
        "--get_embeddings",
    ]

    with (
        mock.patch("sys.argv", test_argv),
        mock.patch(
            "gridfm_graphkit.__main__.main_cli",
        ) as mocked_main_cli,
    ):
        main()

    parsed_args = mocked_main_cli.call_args.args[0]
    assert parsed_args.command == "predict"
    assert parsed_args.get_embeddings is True


def test_main_cli_propagates_get_embeddings_to_task_args(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 0,
                "task": {"task_name": "PowerFlow"},
                "data": {
                    "networks": ["case14"],
                    "workers": 0,
                    "baseMVA": 100,
                },
                "training": {
                    "accelerator": "cpu",
                    "devices": 1,
                    "strategy": "auto",
                    "epochs": 1,
                    "batch_size": 1,
                },
                "callbacks": {"tol": 0.0, "patience": 1},
                "version": 1,
                "optimizer": {
                    "type": "AdamW",
                    "learning_rate": 1e-3,
                    "optimizer_params": {"betas": [0.9, 0.999]},
                    "scheduler_type": "ReduceLROnPlateau",
                    "scheduler_params": {"mode": "min", "factor": 0.5, "patience": 1},
                },
            },
        ),
    )

    args = SimpleNamespace(
        tf32=False,
        log_dir=str(tmp_path / "mlruns"),
        exp_name="tests",
        run_name="predict-embeddings",
        config=str(config_path),
        data_path=str(tmp_path / "data"),
        model_path=str(tmp_path / "model.pt"),
        command="predict",
        output_path=str(tmp_path / "out"),
        get_embeddings=True,
        num_workers=0,
        batch_size=None,
        plugins=[],
        dataset_wrapper=None,
        dataset_wrapper_cache_dir=None,
        mp_context=None,
        normalizer_stats=None,
        bfloat16=False,
        compile=None,
        profiler=None,
        report_performance=False,
        deterministic=False,
        compute_dc_ac_metrics=False,
        save_output=False,
    )

    logger = SimpleNamespace(
        save_dir=str(tmp_path / "mlruns"),
        experiment_id="0",
        run_id="0",
    )
    captured = {}

    class DummyModel:
        def __init__(self):
            self.model = self

        def load_state_dict(self, _state_dict):
            return None

    class DummyDataModule:
        def __init__(self, *args, **kwargs):
            self.data_normalizers = [object()]

    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, model=None, datamodule=None):
            return [
                {
                    "bus": {
                        "scenario": np.array([0]),
                        "vm_pu": np.array([1.0]),
                    },
                    "bus_embeddings": {
                        "scenario": np.array([0]),
                        "bus": np.array([0]),
                        "emb_000": np.array([0.1]),
                    },
                },
            ]

    def fake_get_task(config_args, data_normalizers):
        captured["get_embeddings"] = getattr(config_args, "get_embeddings", None)
        return DummyModel()

    with (
        mock.patch("gridfm_graphkit.cli.MLFlowLogger", return_value=logger),
        mock.patch(
            "gridfm_graphkit.cli.L.seed_everything",
        ),
        mock.patch(
            "gridfm_graphkit.cli.LitGridHeteroDataModule",
            DummyDataModule,
        ),
        mock.patch(
            "gridfm_graphkit.cli.get_task",
            side_effect=fake_get_task,
        ),
        mock.patch(
            "gridfm_graphkit.cli.L.Trainer",
            DummyTrainer,
        ),
        mock.patch(
            "gridfm_graphkit.cli.torch.load",
            return_value={},
        ),
        mock.patch(
            "pandas.DataFrame.to_parquet",
        ),
    ):
        main_cli(args)

    assert captured["get_embeddings"] is True


def test_main_cli_predict_saves_pf_embedding_tables(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 0,
                "task": {"task_name": "PowerFlow"},
                "data": {
                    "networks": ["case14"],
                    "workers": 0,
                    "baseMVA": 100,
                },
                "training": {
                    "accelerator": "cpu",
                    "devices": 1,
                    "strategy": "auto",
                    "epochs": 1,
                    "batch_size": 1,
                },
                "callbacks": {"tol": 0.0, "patience": 1},
                "version": 1,
                "optimizer": {
                    "type": "AdamW",
                    "learning_rate": 1e-3,
                    "optimizer_params": {"betas": [0.9, 0.999]},
                    "scheduler_type": "ReduceLROnPlateau",
                    "scheduler_params": {"mode": "min", "factor": 0.5, "patience": 1},
                },
            },
        ),
    )

    args = SimpleNamespace(
        tf32=False,
        log_dir=str(tmp_path / "mlruns"),
        exp_name="tests",
        run_name="predict-pf-embeddings",
        config=str(config_path),
        data_path=str(tmp_path / "data"),
        model_path=str(tmp_path / "model.pt"),
        command="predict",
        output_path=str(tmp_path / "out"),
        get_embeddings=True,
        num_workers=0,
        batch_size=None,
        plugins=[],
        dataset_wrapper=None,
        dataset_wrapper_cache_dir=None,
        mp_context=None,
        normalizer_stats=None,
        bfloat16=False,
        compile=None,
        profiler=None,
        report_performance=False,
        deterministic=False,
        compute_dc_ac_metrics=False,
        save_output=False,
    )

    logger = SimpleNamespace(
        save_dir=str(tmp_path / "mlruns"),
        experiment_id="0",
        run_id="0",
    )
    writes = []

    class DummyModel:
        def __init__(self):
            self.model = self

        def load_state_dict(self, _state_dict):
            return None

    class DummyDataModule:
        def __init__(self, *args, **kwargs):
            self.data_normalizers = [object()]

    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, model=None, datamodule=None):
            return [
                {
                    "bus": {
                        "scenario": np.array([0]),
                        "bus": np.array([0]),
                        "Vm_pred": np.array([1.0]),
                    },
                    "bus_embeddings": {
                        "scenario": np.array([0]),
                        "bus": np.array([0]),
                        "emb_000": np.array([0.1]),
                    },
                },
                {
                    "bus": {
                        "scenario": np.array([1]),
                        "bus": np.array([1]),
                        "Vm_pred": np.array([1.1]),
                    },
                    "bus_embeddings": {
                        "scenario": np.array([1]),
                        "bus": np.array([1]),
                        "emb_000": np.array([0.2]),
                    },
                },
            ]

    def fake_to_parquet(df, path, index=False):
        writes.append((str(path).split("/")[-1], list(df.columns), df.shape, index))

    with (
        mock.patch("gridfm_graphkit.cli.MLFlowLogger", return_value=logger),
        mock.patch(
            "gridfm_graphkit.cli.L.seed_everything",
        ),
        mock.patch(
            "gridfm_graphkit.cli.LitGridHeteroDataModule",
            DummyDataModule,
        ),
        mock.patch(
            "gridfm_graphkit.cli.get_task",
            return_value=DummyModel(),
        ),
        mock.patch(
            "gridfm_graphkit.cli.L.Trainer",
            DummyTrainer,
        ),
        mock.patch(
            "gridfm_graphkit.cli.torch.load",
            return_value={},
        ),
        mock.patch(
            "pandas.DataFrame.to_parquet",
            new=fake_to_parquet,
        ),
    ):
        main_cli(args)

    saved = {name: (columns, shape, index) for name, columns, shape, index in writes}
    assert set(saved) == {
        "case14_predictions.parquet",
        "case14_bus_embeddings.parquet",
    }
    assert saved["case14_predictions.parquet"][1] == (2, 3)
    assert saved["case14_bus_embeddings.parquet"][0] == ["scenario", "bus", "emb_000"]
    assert all(not index for _, _, index in saved.values())


def test_main_cli_predict_saves_opf_embedding_tables(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 0,
                "task": {"task_name": "OptimalPowerFlow"},
                "data": {
                    "networks": ["case14"],
                    "workers": 0,
                    "baseMVA": 100,
                },
                "training": {
                    "accelerator": "cpu",
                    "devices": 1,
                    "strategy": "auto",
                    "epochs": 1,
                    "batch_size": 1,
                },
                "callbacks": {"tol": 0.0, "patience": 1},
                "version": 1,
                "optimizer": {
                    "type": "AdamW",
                    "learning_rate": 1e-3,
                    "optimizer_params": {"betas": [0.9, 0.999]},
                    "scheduler_type": "ReduceLROnPlateau",
                    "scheduler_params": {"mode": "min", "factor": 0.5, "patience": 1},
                },
            },
        ),
    )

    args = SimpleNamespace(
        tf32=False,
        log_dir=str(tmp_path / "mlruns"),
        exp_name="tests",
        run_name="predict-opf-embeddings",
        config=str(config_path),
        data_path=str(tmp_path / "data"),
        model_path=str(tmp_path / "model.pt"),
        command="predict",
        output_path=str(tmp_path / "out"),
        get_embeddings=True,
        num_workers=0,
        batch_size=None,
        plugins=[],
        dataset_wrapper=None,
        dataset_wrapper_cache_dir=None,
        mp_context=None,
        normalizer_stats=None,
        bfloat16=False,
        compile=None,
        profiler=None,
        report_performance=False,
        deterministic=False,
        compute_dc_ac_metrics=False,
        save_output=False,
    )

    logger = SimpleNamespace(
        save_dir=str(tmp_path / "mlruns"),
        experiment_id="0",
        run_id="0",
    )
    writes = []

    class DummyModel:
        def __init__(self):
            self.model = self

        def load_state_dict(self, _state_dict):
            return None

    class DummyDataModule:
        def __init__(self, *args, **kwargs):
            self.data_normalizers = [object()]

    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, model=None, datamodule=None):
            return [
                {
                    "bus": {
                        "scenario": np.array([0]),
                        "bus": np.array([0]),
                        "Vm_pred": np.array([1.0]),
                    },
                    "gen": {
                        "scenario": np.array([0]),
                        "idx": np.array([0]),
                        "bus": np.array([0]),
                        "p_mw_pred": np.array([10.0]),
                    },
                    "bus_embeddings": {
                        "scenario": np.array([0]),
                        "bus": np.array([0]),
                        "emb_000": np.array([0.1]),
                    },
                    "gen_embeddings": {
                        "scenario": np.array([0]),
                        "idx": np.array([0]),
                        "bus": np.array([0]),
                        "emb_000": np.array([0.2]),
                    },
                },
                {
                    "bus": {
                        "scenario": np.array([1]),
                        "bus": np.array([1]),
                        "Vm_pred": np.array([1.1]),
                    },
                    "gen": {
                        "scenario": np.array([1]),
                        "idx": np.array([1]),
                        "bus": np.array([1]),
                        "p_mw_pred": np.array([11.0]),
                    },
                    "bus_embeddings": {
                        "scenario": np.array([1]),
                        "bus": np.array([1]),
                        "emb_000": np.array([0.3]),
                    },
                    "gen_embeddings": {
                        "scenario": np.array([1]),
                        "idx": np.array([1]),
                        "bus": np.array([1]),
                        "emb_000": np.array([0.4]),
                    },
                },
            ]

    def fake_to_parquet(df, path, index=False):
        writes.append((str(path).split("/")[-1], list(df.columns), df.shape, index))

    with (
        mock.patch("gridfm_graphkit.cli.MLFlowLogger", return_value=logger),
        mock.patch(
            "gridfm_graphkit.cli.L.seed_everything",
        ),
        mock.patch(
            "gridfm_graphkit.cli.LitGridHeteroDataModule",
            DummyDataModule,
        ),
        mock.patch(
            "gridfm_graphkit.cli.get_task",
            return_value=DummyModel(),
        ),
        mock.patch(
            "gridfm_graphkit.cli.L.Trainer",
            DummyTrainer,
        ),
        mock.patch(
            "gridfm_graphkit.cli.torch.load",
            return_value={},
        ),
        mock.patch(
            "pandas.DataFrame.to_parquet",
            new=fake_to_parquet,
        ),
    ):
        main_cli(args)

    saved = {name: (columns, shape, index) for name, columns, shape, index in writes}
    assert set(saved) == {
        "case14_predictions.parquet",
        "case14_gen_predictions.parquet",
        "case14_bus_embeddings.parquet",
        "case14_gen_embeddings.parquet",
    }
    assert saved["case14_predictions.parquet"][1] == (2, 3)
    assert saved["case14_gen_predictions.parquet"][1] == (2, 4)
    assert saved["case14_bus_embeddings.parquet"][0] == ["scenario", "bus", "emb_000"]
    assert saved["case14_gen_embeddings.parquet"][0] == [
        "scenario",
        "idx",
        "bus",
        "emb_000",
    ]
    assert all(not index for _, _, index in saved.values())


class _Const(torch.nn.Module):
    """Returns a constant tensor of width ``out_dim``, one row per input row."""

    def __init__(self, out_dim: int, value: float = 0.0):
        super().__init__()
        self.out_dim = out_dim
        self.value = value

    def forward(self, x):
        return torch.full((x.shape[0], self.out_dim), self.value)


def test_gns_bus_embedding_excludes_final_physics_update() -> None:
    """The exported bus embedding must be the tensor that fed ``mlp_bus``.

    The physics-residual update is skipped on the last layer, so ``h_bus`` at
    the return is exactly that tensor. ``physics_mlp`` returns a huge constant,
    so if the final update were reapplied the embedding would blow up.
    """
    n_bus, n_gen, hidden = 3, 2, 4
    model = object.__new__(GNS_heterogeneous)
    torch.nn.Module.__init__(model)

    # Two layers: layer 0 applies the physics update, layer 1 must not.
    model.task = "PowerFlow"
    model.num_layers = 2
    model.activation = torch.nn.Identity()
    model.layer_residuals = {}
    model.input_proj_bus = _Const(hidden, 2.0)
    model.input_proj_gen = _Const(hidden, 3.0)
    model.input_proj_edge = torch.nn.Identity()
    model.layers = [lambda h, ei, ea: dict(h)] * 2
    model.norms_bus = [torch.nn.Identity()] * 2
    model.norms_gen = [torch.nn.Identity()] * 2
    model.mlp_bus = _Const(2, 1.0)
    model.mlp_gen = _Const(1, 1.0)
    model.physics_mlp = _Const(hidden, 1e6)
    model.branch_flow_layer = lambda bus, ei, ea: (
        torch.zeros(ei.shape[1]),
        torch.zeros(ei.shape[1]),
    )
    model.node_injection_layer = lambda p, q, ei, nb: (
        torch.zeros(nb),
        torch.zeros(nb),
    )
    model.physics_decoder = lambda p, q, bus_temp, bus_x, agg, md: torch.zeros(
        (n_bus, 4),
    )
    model.node_residuals_layer = lambda p, q, out, bus_x: (
        torch.zeros(n_bus),
        torch.zeros(n_bus),
    )

    batch = SimpleNamespace(
        x_dict={"bus": torch.zeros((n_bus, 16)), "gen": torch.zeros((n_gen, 8))},
        edge_index_dict={
            ("bus", "connects", "bus"): torch.tensor([[0, 1], [1, 2]]),
            ("gen", "connected_to", "bus"): torch.tensor([[0, 1], [0, 1]]),
        },
        edge_attr_dict={
            ("bus", "connects", "bus"): torch.zeros((2, 2)),
            ("gen", "connected_to", "bus"): None,
        },
        mask_dict={
            "bus": torch.ones((n_bus, 16), dtype=torch.bool),
            "gen": torch.ones((n_gen, 8), dtype=torch.bool),
        },
    )

    _, embeddings = model(batch, return_embeddings=True)

    # layer 0: h_bus = 2.0 (proj) + 2.0 (identity conv) = 4.0, then += 1e6.
    # layer 1: h_bus = 1e6+4 doubled by the skip connection, and NO further update.
    expected = torch.full((n_bus, hidden), (1e6 + 4.0) * 2)
    torch.testing.assert_close(embeddings["bus"], expected)

    # The final layer's physics update must not be included.
    assert not torch.allclose(
        embeddings["bus"],
        expected + 1e6,
    ), "bus embedding includes the final-layer physics update"

    # layer_residuals must still be recorded for every layer (the loss reads all).
    assert sorted(model.layer_residuals) == [0, 1]
