import pytest
import subprocess
import os
import glob
import pandas as pd
import yaml
import shutil
import zipfile
import gdown
import tempfile
import numpy as np
from scipy import stats


def execute_and_live_output(cmd) -> None:
    subprocess.run(cmd, text=True, shell=True, check=True)


def collect_metrics_from_log(log_base: str, metric_keys: list) -> dict:
    """Find the latest run's metrics CSV and return a dict of {metric: value}."""
    exp_dirs = glob.glob(os.path.join(log_base, "*"))
    assert len(exp_dirs) > 0, f"No experiment directories found in {log_base}/"
    latest_exp_dir = sorted(exp_dirs, key=os.path.getctime)[-1]
    run_dirs = glob.glob(os.path.join(latest_exp_dir, "*"))
    assert len(run_dirs) > 0, f"No run directories found in {latest_exp_dir}"
    latest_run_dir = max(run_dirs, key=os.path.getmtime)
    metrics_file = os.path.join(latest_run_dir, "artifacts", "test", "case14_ieee_metrics.csv")
    assert os.path.exists(metrics_file), f"Metrics file not found: {metrics_file}"
    df = pd.read_csv(metrics_file)
    return dict(zip(df["Metric"], df["Value"].astype(float)))


def print_calibration_stats(all_runs: list, metric_keys: list, confidence_interval: float = 0.995) -> None:
    """
    Print per-metric stats across calibration runs:
      - std with Bessel's correction (ddof=1)
      - two-sided CI using Student-t distribution

    Args:
        all_runs: list of per-run metric dicts
        metric_keys: list of metric names to report
        confidence_interval: desired confidence level (default 0.995).
            Example with higher confidence:
                print_calibration_stats(all_runs, metric_keys, confidence_interval=0.995)
    """
    n = len(all_runs)
    alpha_half = (1 + confidence_interval) / 2
    t_crit = stats.t.ppf(alpha_half, df=max(n - 1, 1))
    ci_pct = f"{confidence_interval * 100:g}"
    col_w = max(len(k) for k in metric_keys) + 2
    header = f"  {'Metric':<{col_w}}  {'Mean':>10}  {'Std(ddof=1)':>12}  {f'CI {ci_pct}% lo':>10}  {f'CI {ci_pct}% hi':>10}"
    print(f"\n===== Calibration Results (n={n}, CI={confidence_interval}, t_crit={t_crit:.4f}) =====")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in metric_keys:
        values = [run[key] for run in all_runs if key in run]
        if not values:
            print(f"  {key:<{col_w}}  {'no data':>10}")
            continue
        arr = np.array(values, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        me = t_crit * std / np.sqrt(len(arr))  # margin of error
        lo, hi = mean - me, mean + me
        print(
            f"  {key:<{col_w}}  {mean:>10.4f}  {std:>12.4f}  {lo:>10.4f}  {hi:>10.4f}"
        )
    print("=" * (len(header)) + "\n")


def prepare_training_config():
    """
    Modify the PF training config to set epochs to 20 and hidden_size to 12 for testing.
    """
    config_path = "examples/config/HGNS_PF_datakit_case14.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "training" not in config:
        config["training"] = {}
    if "model" not in config:
        config["model"] = {}

    config["training"]["epochs"] = 20
    config["model"]["hidden_size"] = 12

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Training config updated: epochs set to {config['training']['epochs']}, hidden_size set to {config['model']['hidden_size']}")

    return config_path


def prepare_opf_training_config():
    """
    Modify the OPF training config to set epochs to 20 and hidden_size to 12 for testing.
    """
    config_path = "examples/config/HGNS_OPF_datakit_case14.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "training" not in config:
        config["training"] = {}
    if "model" not in config:
        config["model"] = {}

    config["training"]["epochs"] = 20
    config["model"]["hidden_size"] = 12

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"OPF training config updated: epochs set to {config['training']['epochs']}, hidden_size set to {config['model']['hidden_size']}")

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

    # Remove downloaded config
    config_file = "integrationtests/default.yaml"
    if os.path.exists(config_file):
        os.remove(config_file)

    # Remove generated directories
    for d in ["data_out", "logs"]:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)


def test_train_pf(cleanup_test_artifacts, calibrate_runs, ci_level):
    """
    Integration test for power flow (PF): gridfm-datakit data generation and gridfm-graphkit training.

    Steps:
    1. Generate power flow grid data using gridfm-datakit
    2. Train a PF model using gridfm-graphkit
    3. Validate the PBE Mean metric

    Pass --calibrate N to pytest (e.g. pytest --calibrate 5) to run N training passes
    and print metric mean/std without asserting range bounds.
    """

    n_runs = max(calibrate_runs, 1)
    pf_metric_keys = ["PBE Mean"]

    data_dir = "data_out"

    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print("Data directory not found or empty, downloading pre-generated data...")

        gdrive_file_id = "1NtE_4Fn3-1_BNWidZVFeSTfXf3-B50Yr"
        zip_filename = "case14_ieee.10000_scenarios_2_variants.zip"
        gdrive_url = f"https://drive.google.com/uc?id={gdrive_file_id}"

        print(f"Downloading {zip_filename} from Google Drive...")
        gdown.download(gdrive_url, zip_filename, quiet=False)

        print(f"Extracting {zip_filename}...")
        with zipfile.ZipFile(zip_filename, "r") as zf:
            zf.extractall(".")

        os.remove(zip_filename)
        print(f"Data extracted to '{data_dir}'.")
    else:
        print(f"Data directory '{data_dir}' already exists, skipping download.")

    training_config_path = prepare_training_config()
    all_runs = []

    for run_i in range(n_runs):
        print(f"\n--- PF Training run {run_i + 1}/{n_runs} ---")
        execute_and_live_output(
            f"gridfm_graphkit train "
            f"--config {training_config_path} "
            f"--data_path data_out/ "
            f"--exp_name exp1 "
            f"--run_name run{run_i + 1} "
            f"--log_dir logs",
        )
        metrics = collect_metrics_from_log("logs", pf_metric_keys)
        all_runs.append(metrics)

    if calibrate_runs > 0:
        print_calibration_stats(all_runs, pf_metric_keys, confidence_interval=ci_level)
        return

    MAX_RETRIES = 5
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            print(f"\n--- PF Retry attempt {attempt}/{MAX_RETRIES} after metric interval failure ---")
            execute_and_live_output(
                f"gridfm_graphkit train "
                f"--config {training_config_path} "
                f"--data_path data_out/ "
                f"--exp_name exp1 "
                f"--run_name retry{attempt} "
                f"--log_dir logs",
            )
            metrics = collect_metrics_from_log("logs", pf_metric_keys)
        else:
            metrics = all_runs[0]

        pbe_mean_value = metrics["PBE Mean"]
        try:
            assert 0.2042 <= pbe_mean_value <= 0.6397, (
                f"PBE Mean value {pbe_mean_value} is outside 95% CI [0.2042, 0.6397]"
            )
            print(f"PBE Mean value {pbe_mean_value} is within 95% CI [0.2042, 0.6397] (attempt {attempt})")
            last_error = None
            break
        except AssertionError as e:
            print(f"Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            last_error = e

    if last_error is not None:
        raise last_error


@pytest.fixture
def cleanup_opf_test_artifacts():
    """
    Remove generated artifacts after the OPF test.
    """
    yield

    for d in ["data_out_opf", "logs_opf"]:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)


def test_train_opf(cleanup_opf_test_artifacts, calibrate_runs, ci_level):
    """
    Integration test for OPF data download and gridfm-graphkit OPF training.

    Steps:
    1. Download pre-generated OPF power grid data from Google Drive
    2. Train a model using gridfm-graphkit with the OPF config
    3. Validate OPF-specific metrics

    Pass --calibrate N to pytest (e.g. pytest --calibrate 5) to run N training passes
    and print metric mean/std without asserting range bounds.
    """

    n_runs = max(calibrate_runs, 1)
    opf_metric_keys = [
        "Avg. active res. (MW)",
        "Avg. reactive res. (MVar)",
        "RMSE PG generators (MW)",
        "Mean optimality gap (%)",
        "Mean branch thermal violation from (MVA)",
        "Mean branch thermal violation to (MVA)",
        "Mean branch angle difference violation (radians)",
        "Mean Qg violation PV buses",
        "Mean Qg violation REF buses",
        "Mean Qg violation",
    ]

    opf_data_dir = "data_out_opf"

    if not os.path.exists(opf_data_dir) or not os.listdir(opf_data_dir):
        print("OPF data directory not found or empty, downloading pre-generated data...")

        gdrive_file_id = "1p5f5mRvmBQh8lZpIyWWbTbU42aHAIsdT"  # pragma: allowlist secret
        zip_filename = "case14_ieee.10000_scenarios_2_variants_opf.zip"
        gdrive_url = f"https://drive.google.com/uc?id={gdrive_file_id}"

        print(f"Downloading {zip_filename} from Google Drive...")
        gdown.download(gdrive_url, zip_filename, quiet=False)

        print(f"Extracting {zip_filename}...")
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_filename, "r") as zf:
                zf.extractall(tmpdir)
            shutil.move(os.path.join(tmpdir, "data_out"), opf_data_dir)

        os.remove(zip_filename)
        print(f"OPF data extracted to '{opf_data_dir}'.")
    else:
        print(f"OPF data directory '{opf_data_dir}' already exists, skipping download.")

    training_config_path = prepare_opf_training_config()
    all_runs = []

    for run_i in range(n_runs):
        print(f"\n--- OPF Training run {run_i + 1}/{n_runs} ---")
        execute_and_live_output(
            f"gridfm_graphkit train "
            f"--config {training_config_path} "
            f"--data_path {opf_data_dir}/ "
            f"--exp_name exp_opf "
            f"--run_name run{run_i + 1} "
            f"--log_dir logs_opf",
        )
        metrics = collect_metrics_from_log("logs_opf", opf_metric_keys)
        all_runs.append(metrics)

    if calibrate_runs > 0:
        print_calibration_stats(all_runs, opf_metric_keys, confidence_interval=ci_level)
        return

    checks = {
        "Avg. active res. (MW)": (0.2067, 0.4619),
        "Avg. reactive res. (MVar)": (0.0825, 0.1492),
        "RMSE PG generators (MW)": (2.6480, 2.8693),
        "Mean optimality gap (%)": (1.1039, 1.4934),
        "Mean branch thermal violation from (MVA)": (0.0, 0.0),
        "Mean branch thermal violation to (MVA)": (0.0, 0.0),
        "Mean branch angle difference violation (radians)": (0.0, 0.0),
        "Mean Qg violation PV buses": (0.0167, 0.1546),
        "Mean Qg violation REF buses": (-0.0693, 0.4241),
        "Mean Qg violation": (0.0771, 0.1322),
    }

    MAX_RETRIES = 5
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            print(f"\n--- OPF Retry attempt {attempt}/{MAX_RETRIES} after metric interval failure ---")
            execute_and_live_output(
                f"gridfm_graphkit train "
                f"--config {training_config_path} "
                f"--data_path {opf_data_dir}/ "
                f"--exp_name exp_opf "
                f"--run_name retry{attempt} "
                f"--log_dir logs_opf",
            )
            metrics = collect_metrics_from_log("logs_opf", opf_metric_keys)
        else:
            metrics = all_runs[0]

        try:
            for metric_name, (lo, hi) in checks.items():
                assert metric_name in metrics, f"Metric '{metric_name}' not found in CSV"
                value = metrics[metric_name]
                assert lo <= value <= hi, (
                    f"Metric '{metric_name}' value {value} is outside 99.5% CI [{lo}, {hi}]"
                )
                print(f"{metric_name}: {value} is within 99.5% CI [{lo}, {hi}] (attempt {attempt})")
            last_error = None
            break
        except AssertionError as e:
            print(f"Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            last_error = e

    if last_error is not None:
        raise last_error
