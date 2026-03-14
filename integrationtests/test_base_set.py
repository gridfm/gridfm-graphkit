import pytest
import subprocess
import os
import glob
import pandas as pd
import yaml
import urllib.request
import shutil


def execute_and_live_output(cmd) -> None:
    result = subprocess.run(
        cmd,
        text=True,
        shell=True,
        check=True
    )


def prepare_config():
    """
    Download default.yaml from gridfm-datakit repo and modify it with test parameters.
    """
    config_url = "https://raw.githubusercontent.com/gridfm/gridfm-datakit/refs/heads/main/scripts/config/default.yaml"
    config_path = "integrationtests/default.yaml"

    print(f"Downloading config from {config_url}...")
    with urllib.request.urlopen(config_url) as response:
        config_content = response.read().decode("utf-8")

    config = yaml.safe_load(config_content)

    config["network"]["name"] = "case14_ieee"
    config["load"]["scenarios"] = 10000
    config["topology_perturbation"]["n_topology_variants"] = 2

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Config prepared at {config_path} with:")
    print(f"  - network.name: {config['network']['name']}")
    print(f"  - load.scenarios: {config['load']['scenarios']}")
    print(
        f"  - topology_perturbation.n_topology_variants: "
        f"{config['topology_perturbation']['n_topology_variants']}"
    )

    return config_path


def prepare_training_config():
    """
    Modify the training config to set epochs to 2 for testing.
    """
    config_path = "examples/config/HGNS_PF_datakit_case14.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "training" not in config:
        config["training"] = {}

    config["training"]["epochs"] = 2

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Training config updated: epochs set to {config['training']['epochs']}")

    return config_path


@pytest.fixture
def cleanup_test_artifacts():
    """
    Backup modified files and remove generated artifacts after the test.
    """
    training_config = "examples/config/HGNS_PF_datakit_case14.yaml"
    backup_config = training_config + ".bak"

    if os.path.exists(training_config):
        shutil.copy2(training_config, backup_config)

    yield

    # Restore training config
    if os.path.exists(backup_config):
        shutil.move(backup_config, training_config)

    # Remove downloaded config
    config_file = "integrationtests/default.yaml"
    if os.path.exists(config_file):
        os.remove(config_file)

    # Remove generated directories
    for d in ["data_out", "logs"]:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)


def test_train(cleanup_test_artifacts):
    """
    Integration test for gridfm-datakit data generation and gridfm-graphkit training.

    Steps:
    1. Generate power grid data using gridfm-datakit
    2. Train a model using gridfm-graphkit
    3. Validate the PBE Mean metric
    """

    data_dir = "data_out"

    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print("Data directory not found or empty, generating data...")

        config_path = prepare_config()

        execute_and_live_output(
            f"gridfm_datakit generate {config_path}"
        )
    else:
        print(f"Data directory '{data_dir}' already exists, skipping generation.")

    training_config_path = prepare_training_config()

    execute_and_live_output(
        f"gridfm_graphkit train "
        f"--config {training_config_path} "
        f"--data_path data_out/ "
        f"--exp_name exp1 "
        f"--run_name run1 "
        f"--log_dir logs"
    )

    log_base = "logs"

    exp_dirs = glob.glob(os.path.join(log_base, "*"))
    assert len(exp_dirs) > 0, "No experiment directories found in logs/"

    latest_exp_dir = sorted(exp_dirs, key=os.path.getctime)[-1]

    run_dirs = glob.glob(os.path.join(latest_exp_dir, "*"))
    assert len(run_dirs) > 0, f"No run directories found in {latest_exp_dir}"

    latest_run_dir = max(run_dirs, key=os.path.getmtime)

    metrics_file = os.path.join(
        latest_run_dir,
        "artifacts",
        "test",
        "case14_ieee_metrics.csv"
    )

    assert os.path.exists(metrics_file), f"Metrics file not found: {metrics_file}"

    df = pd.read_csv(metrics_file)

    pbe_mean_row = df[df["Metric"] == "PBE Mean"]
    assert len(pbe_mean_row) > 0, "PBE Mean metric not found in CSV"

    pbe_mean_value = float(pbe_mean_row.iloc[0]["Value"])

    assert 1.1 <= pbe_mean_value <= 2.9, (
        f"PBE Mean value {pbe_mean_value} is outside acceptable range [1.1, 2.9]"
    )

    print(f"PBE Mean value {pbe_mean_value} is within acceptable range [1.1, 2.9]")