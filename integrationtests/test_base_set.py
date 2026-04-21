import pytest
import subprocess
import os
import glob
import zipfile
import pandas as pd
import yaml
import shutil
import gdown


def execute_and_live_output(cmd) -> None:
    subprocess.run(cmd, text=True, shell=True, check=True)


# Data generation via gridfm-datakit is replaced by a pre-built dataset
# hosted on Google Drive (case14_ieee, 10 000 scenarios, 2 topology variants).
# def prepare_config():
#     ...


def download_dataset() -> None:
    """Download the pre-built dataset from Google Drive and extract it to the repo root."""
    gdrive_file_id = "1NnNKOPqZU8yL6H-V3hnLFLkDyooH7B5D"
    zip_path = "integrationtests/case14_ieee_dataset.zip"

    print(f"Downloading dataset (file id={gdrive_file_id}) from Google Drive...")
    gdown.download(id=gdrive_file_id, output=zip_path, quiet=False)

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(".")

    print("Dataset ready.")


def prepare_training_config():
    """
    Modify the training config to set epochs to 2 for a quick integration test run.
    """
    config_path = "examples/config/HGNS_PF_datakit_case14.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "training" not in config:
        config["training"] = {}

    config["training"]["epochs"] = 2

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Training config updated: epochs={config['training']['epochs']}")

    return config_path


@pytest.fixture
def cleanup_test_artifacts():
    """
    Backup modified files and remove generated artifacts after the test.
    """
    training_config = " "
    backup_config = training_config + ".bak"

    if os.path.exists(training_config):
        shutil.copy2(training_config, backup_config)

    yield

    # Restore training config
    if os.path.exists(backup_config):
        shutil.move(backup_config, training_config)

    # Remove downloaded zip
    zip_file = "integrationtests/case14_ieee_dataset.zip"
    if os.path.exists(zip_file):
        os.remove(zip_file)

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
        print("Data directory not found or empty, downloading pre-built dataset...")
        download_dataset()
    else:
        print(f"Data directory '{data_dir}' already exists, skipping download.")

    training_config_path = prepare_training_config()

    execute_and_live_output(
        f"gridfm_graphkit train "
        f"--config {training_config_path} "
        f"--data_path data_out/ "
        f"--exp_name exp1 "
        f"--run_name run1 "
        f"--log_dir logs",
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
        "case14_ieee_metrics.csv",
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
