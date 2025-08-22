import unittest
from argparse import ArgumentParser
from gridfm_graphkit.cli import main_cli
from gridfm_graphkit.__main__ import main
from unittest import mock
import sys


class TestMLPipeline(unittest.TestCase):
    def setUp(self):
        # Default paths for tests
        self.config = "tests/config/gridFMv0.1_dummy.yaml"
        self.data_path = "tests/data"
        self.log_dir = "tests/mlruns"
        self.model_path = "examples/models/GridFM_v0_1.pth"
        self.output_path = "tests/output"
        self.exp_name = "pytest_exp"
        self.run_name = "pytest_run"

        # Setup parser once
        self.parser = ArgumentParser()
        subparsers = self.parser.add_subparsers(dest="command", required=True)
        for cmd in ["train", "finetune", "evaluate", "predict"]:
            sp = subparsers.add_parser(cmd)
            sp.add_argument("--config")
            sp.add_argument("--data_path")
            sp.add_argument("--log_dir")
            sp.add_argument("--model_path")
            sp.add_argument("--output_path")
            sp.add_argument("--exp_name")
            sp.add_argument("--run_name")

    def _run_cli(self, command):
        args_list = [
            command,
            "--config",
            self.config,
            "--data_path",
            self.data_path,
            "--log_dir",
            self.log_dir,
            "--exp_name",
            self.exp_name,
            "--run_name",
            self.run_name,
        ]
        if command in ["finetune", "evaluate", "predict"]:
            args_list += ["--model_path", self.model_path]
        if command == "predict":
            args_list += ["--output_path", self.output_path]
        args = self.parser.parse_args(args_list)
        main_cli(args)

    def test_train(self):
        self._run_cli("train")

    def test_finetune(self):
        self._run_cli("finetune")

    def test_evaluate(self):
        self._run_cli("evaluate")

    def test_predict(self):
        self._run_cli("predict")

    def test_entrypoint_train(self):
        test_argv = [
            "gridfm_graphkit",
            "train",
            "--config",
            self.config,
            "--data_path",
            self.data_path,
            "--log_dir",
            self.log_dir,
            "--exp_name",
            self.exp_name,
            "--run_name",
            self.run_name,
        ]
        with mock.patch.object(sys, "argv", test_argv):
            main()
