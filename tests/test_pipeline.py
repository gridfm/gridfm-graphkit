import sys
from argparse import ArgumentParser
from unittest import mock

import pytest
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from gridfm_graphkit.cli import DEFAULT_MONITOR, get_training_callbacks, main_cli
from gridfm_graphkit.__main__ import main
from gridfm_graphkit.io.param_handler import NestedNamespace
from gridfm_graphkit.training.callbacks import SaveBestModelStateDict


# -------------------------------------------------
# Test configurations
# -------------------------------------------------
CONFIGS = [
    "tests/config/datamodule_test_base_config.yaml",
    "tests/config/datamodule_test_base_config2.yaml",
    "tests/config/datamodule_test_base_config3.yaml",
]

DATA_PATH = "tests/data"
LOG_DIR = "tests/mlruns"
MODEL_PATH = "tests/models/dummy_model.pt"
EXP_NAME = "pytest_exp"
RUN_NAME = "pytest_run"


# -------------------------------------------------
# Fixtures
# -------------------------------------------------
@pytest.fixture
def parser():
    """
    Argument parser matching the CLI.
    """
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    for cmd in ["train", "finetune", "evaluate"]:
        sp = subparsers.add_parser(cmd)
        sp.add_argument("--config")
        sp.add_argument("--data_path")
        sp.add_argument("--log_dir")
        sp.add_argument("--model_path")
        sp.add_argument("--exp_name")
        sp.add_argument("--run_name")

    return parser


# -------------------------------------------------
# CLI command tests
# -------------------------------------------------
@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("command", ["train", "finetune", "evaluate"])
def test_cli_commands(parser, config, command):
    """
    Test main_cli() directly for all commands and configs.
    """
    args_list = [
        command,
        "--config",
        config,
        "--data_path",
        DATA_PATH,
        "--log_dir",
        LOG_DIR,
        "--exp_name",
        EXP_NAME,
        "--run_name",
        RUN_NAME,
    ]

    if command in ["finetune", "evaluate"]:
        args_list += ["--model_path", MODEL_PATH]

    args = parser.parse_args(args_list)

    # Should run without raising
    main_cli(args)


# -------------------------------------------------
# Entrypoint (__main__) test
# -------------------------------------------------
@pytest.mark.parametrize("config", CONFIGS)
def test_entrypoint_train(config):
    """
    Test the console entrypoint: python -m gridfm_graphkit train ...
    """
    test_argv = [
        "gridfm_graphkit",
        "train",
        "--config",
        config,
        "--data_path",
        DATA_PATH,
        "--log_dir",
        LOG_DIR,
        "--exp_name",
        EXP_NAME,
        "--run_name",
        RUN_NAME,
    ]

    with mock.patch.object(sys, "argv", test_argv):
        main()


# -------------------------------------------------
# Callback monitor wiring (from YAML callbacks section)
# -------------------------------------------------
def _callbacks_by_type(callbacks):
    return {type(cb): cb for cb in callbacks}


def test_get_training_callbacks_reads_config_monitors():
    """Each callback tracks its configured metric; direction is always 'min'."""
    args = NestedNamespace(
        callbacks={
            "patience": 5,
            "tol": 0,
            "early_stopping_monitor": "Validation PBE Mean",
            "checkpoint_monitor": "Validation layer_11_residual",
        },
    )
    by_type = _callbacks_by_type(get_training_callbacks(args))

    assert by_type[EarlyStopping].monitor == "Validation PBE Mean"
    assert by_type[SaveBestModelStateDict].monitor == "Validation layer_11_residual"

    assert all(cb.mode == "min" for cb in by_type.values())


def test_get_training_callbacks_defaults_when_monitors_absent():
    """Omitted monitor keys fall back to the default 'Validation loss'."""
    args = NestedNamespace(callbacks={"patience": 5, "tol": 0})
    by_type = _callbacks_by_type(get_training_callbacks(args))

    assert by_type[EarlyStopping].monitor == DEFAULT_MONITOR
    assert by_type[SaveBestModelStateDict].monitor == DEFAULT_MONITOR
