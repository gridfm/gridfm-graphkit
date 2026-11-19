import sys
from argparse import ArgumentParser
from unittest import mock

import pytest

from gridfm_graphkit.cli import main_cli
from gridfm_graphkit.__main__ import main


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
# State estimation with the WLS baseline
# -------------------------------------------------
def test_state_estimation_evaluate_with_wls(parser, tmp_path):
    """Train a tiny SE model, then evaluate it together with the WLS baseline."""
    pytest.importorskip("pandapower")
    import glob
    import yaml

    with open("scripts/state_estimation/configs/case14_ieee_base_0.yaml") as f:
        config = yaml.safe_load(f)
    config["data"]["scenarios"] = [72]
    config["data"]["workers"] = 0
    config["model"].update({"hidden_size": 8, "num_layers": 2, "attention_head": 2})
    config["training"].update(
        {"epochs": 1, "batch_size": 8, "accelerator": "cpu", "devices": 1},
    )
    config["verbose"] = False
    train_config = tmp_path / "train.yaml"
    train_config.write_text(yaml.safe_dump(config))

    log_dir = str(tmp_path / "mlruns")
    main_cli(
        parser.parse_args(
            [
                "train",
                "--config",
                str(train_config),
                "--data_path",
                DATA_PATH,
                "--log_dir",
                log_dir,
                "--exp_name",
                "se_train",
                "--run_name",
                "train",
            ],
        ),
    )
    model_path = glob.glob(f"{log_dir}/*/*/artifacts/model/best_model_state_dict.pt")[0]

    config["task"]["run_pp"] = True
    config["training"]["batch_size"] = 1
    config["data"]["val_ratio"] = 0.0
    config["data"]["test_ratio"] = 0.99
    config["data"]["scenarios"] = [6]
    eval_config = tmp_path / "eval.yaml"
    eval_config.write_text(yaml.safe_dump(config))
    main_cli(
        parser.parse_args(
            [
                "evaluate",
                "--config",
                str(eval_config),
                "--data_path",
                DATA_PATH,
                "--log_dir",
                log_dir,
                "--model_path",
                model_path,
                "--exp_name",
                "se_eval",
                "--run_name",
                "eval",
            ],
        ),
    )

    csvs = {
        p.rsplit("_", 1)[-1]
        for p in glob.glob(f"{log_dir}/*/*/artifacts/test/case14_ieee_metrics_*.csv")
    }
    assert {"c.csv", "pp.csv", "cnc.csv"} <= csvs
