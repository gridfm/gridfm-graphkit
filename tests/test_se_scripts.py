"""The state estimation configs and scripts used for the paper."""

import glob
import os
import subprocess
import sys

import pytest
import yaml

from gridfm_graphkit.io.param_handler import (
    NestedNamespace,
    get_loss_function,
    get_task_transforms,
    load_model,
)

SCRIPTS = "scripts/state_estimation"
CONFIGS = sorted(glob.glob(f"{SCRIPTS}/configs/*.yaml"))


def test_all_paper_configs_exist():
    grids = ["case14_ieee", "case30_ieee", "case57_ieee", "case118_ieee"]
    expected = {f"{g}_{kind}_0.yaml" for g in grids for kind in ["base", "pe"]}
    assert expected == {os.path.basename(c) for c in CONFIGS}


def test_generator_reproduces_configs(tmp_path):
    for name in ["base.yaml", "generate_configs.py"]:
        (tmp_path / name).write_text(open(f"{SCRIPTS}/{name}").read())
    subprocess.run([sys.executable, str(tmp_path / "generate_configs.py")], check=True)
    for config in CONFIGS:
        with open(config) as f:
            committed = yaml.safe_load(f)
        with open(tmp_path / "configs" / os.path.basename(config)) as f:
            generated = yaml.safe_load(f)
        assert committed == generated, config


@pytest.mark.parametrize("config_path", CONFIGS)
def test_config_builds(config_path):
    with open(config_path) as f:
        args = NestedNamespace(**yaml.safe_load(f))
    assert args.task.task_name == "StateEstimation"
    assert args.task.run_pp is False
    load_model(args)
    get_loss_function(args)
    get_task_transforms(args)


def test_eval_settings(tmp_path):
    sys.path.insert(0, SCRIPTS)
    from make_eval_configs import make_settings

    with open(f"{SCRIPTS}/configs/case57_ieee_base_0.yaml") as f:
        train_config = yaml.safe_load(f)

    settings = make_settings(train_config, "case57_ieee", str(tmp_path), 500)
    names = [name for name, _, _ in settings]
    # The run on the unperturbed dataset comes first. It processes the grid the unk_pert runs use.
    assert names[:3] == [
        "unk_pert_perturbation_0.0_0",
        "unk_pert_perturbation_0.1_0",
        "unk_pert_perturbation_0.3_0",
    ]
    sweep = {n for n in names if n.startswith("swype_")}
    assert sweep == {
        f"swype_{m}_{o}" for m in [0.1, 0.2, 0.4] for o in [0.0, 0.1, 0.2]
    } | {"swype_0.4_0.3"}

    config = dict((n, c) for n, c, _ in settings)["swype_0.4_0.3"]
    assert config["task"]["run_pp"] is True
    assert config["training"]["batch_size"] == 1
    assert config["seed"] == 10
    assert config["data"]["scenarios"] == [500]
    for m in ["vm", "power_inj", "power_flow"]:
        assert config["task"]["measurements"][m]["mask_ratio"] == 0.4
        assert config["task"]["measurements"][m]["outlier_ratio"] == 0.3

    data = dict((n, d) for n, _, d in settings)
    assert data["swype_0.1_0.0"].endswith(
        os.path.join("case57_ieee", "perturbation_0.2_5"),
    )
    assert data["unk_pert_perturbation_0.3_0"].endswith(
        os.path.join("case57_ieee", "perturbation_0.3_0"),
    )


def test_datakit_configs(tmp_path):
    raw = tmp_path / "train" / "case118_ieee" / "raw"
    raw.mkdir(parents=True)
    train = {
        "admittance_perturbation": {"sigma": 0.2, "type": "random_perturbation"},
        "topology_perturbation": {"type": "random", "k": 10, "n_topology_variants": 24},
        "load": {"start_scaling_factor": 1.0, "max_scaling_factor": 4.0},
        "settings": {"data_dir": "x"},
    }
    (raw / "args.log").write_text("\nNew generation started\n" + yaml.safe_dump(train))
    (raw / "scenarios_agg_load_profile.log").write_text(
        "\nNew generation started\nu=1.3\nl=0.78\n",
    )
    subprocess.run(
        [
            sys.executable,
            f"{SCRIPTS}/generate_datakit_configs.py",
            "--train_data_dir", str(tmp_path / "train"),
            "--out_dir", str(tmp_path / "out"),
            "--grids", "case118_ieee",
        ],
        check=True,
    )  # fmt: skip
    folders = sorted(os.listdir(tmp_path / "out" / "case118_ieee"))
    assert folders == [
        "perturbation_0.0_0",
        "perturbation_0.1_0",
        "perturbation_0.2_10",
        "perturbation_0.3_0",
    ]
    with open(
        tmp_path / "out" / "case118_ieee" / "perturbation_0.2_10" / "config.yaml",
    ) as f:
        config = yaml.safe_load(f)
    assert config["admittance_perturbation"]["sigma"] == 0.2
    assert config["topology_perturbation"]["k"] == 10
    assert config["topology_perturbation"]["n_topology_variants"] == 1
    assert config["load"]["start_scaling_factor"] == 1.3
    with open(
        tmp_path / "out" / "case118_ieee" / "perturbation_0.0_0" / "config.yaml",
    ) as f:
        assert yaml.safe_load(f)["topology_perturbation"]["type"] == "none"
