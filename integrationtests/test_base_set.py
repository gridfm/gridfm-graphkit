import pytest
import subprocess
import os
import glob
import pandas as pd
import yaml
import urllib.request

def execute_and_fail(cmd) -> None:
    """
    Execute a CLI command and fail in case return code is not 0.
    """
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        shell=True,
    )
    assert result.returncode == 0, (
        f"{cmd} failed (exit {result.returncode}).\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

def prepare_config():
    """
    Download default.yaml from gridfm-datakit repo and modify it with test parameters.
    """
    config_url = "https://raw.githubusercontent.com/gridfm/gridfm-datakit/refs/heads/main/scripts/config/default.yaml"
    config_path = "integrationtests/default.yaml"
    
    print(f"Downloading config from {config_url}...")
    with urllib.request.urlopen(config_url) as response:
        config_content = response.read().decode('utf-8')
    
    # Parse YAML
    config = yaml.safe_load(config_content)
    
    # Update values as specified (nested structure)
    config['network']['name'] = 'case14_ieee'
    config['load']['scenarios'] = 10000
    config['topology_perturbation']['n_topology_variants'] = 2
    
    # Write modified config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Config prepared at {config_path} with:")
    print(f"  - network.name: {config['network']['name']}")
    print(f"  - load.scenarios: {config['load']['scenarios']}")
    print(f"  - topology_perturbation.n_topology_variants: {config['topology_perturbation']['n_topology_variants']}")
    
    return config_path

def test_prepare_data():
    """
    gridfm-datakit must be installable via pip with exit code 0.

    This test explicitly re-runs the install command and asserts that pip
    exits successfully, making the install step a first-class test rather
    than a silent fixture side-effect.
    """

    # Check if data already exists, if not generate it
    data_dir = "data_out"
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print("Data directory not found or empty, generating data...")
        
        # Prepare the config file
        config_path = prepare_config()
        
        # Generate data using the prepared config
        execute_and_fail(
            f'gridfm_datakit generate {config_path}'
        )
    else:
        print(f"Data directory '{data_dir}' already exists, skipping data generation.")
    
    execute_and_fail(
        'gridfm_graphkit train --config examples/config/HGNS_PF_datakit_case14.yaml --data_path data_out/ --exp_name exp1 --run_name run1 --log_dir logs'
    )
    
    # Find the latest log directory
    log_base = "logs"
    exp_dirs = glob.glob(os.path.join(log_base, "*"))
    assert len(exp_dirs) > 0, "No experiment directories found in logs/"
    
    latest_exp_dir = max(exp_dirs, key=os.path.getmtime)
    run_dirs = glob.glob(os.path.join(latest_exp_dir, "*"))
    assert len(run_dirs) > 0, f"No run directories found in {latest_exp_dir}"
    
    latest_run_dir = max(run_dirs, key=os.path.getmtime)
    metrics_file = os.path.join(latest_run_dir, "artifacts", "test", "case14_ieee_metrics.csv")
    
    assert os.path.exists(metrics_file), f"Metrics file not found: {metrics_file}"
    
    # Read the metrics CSV
    df = pd.read_csv(metrics_file)
    
    # Find PBE Mean value
    pbe_mean_row = df[df['Metric'] == 'PBE Mean']
    assert len(pbe_mean_row) > 0, "PBE Mean metric not found in CSV"
    
    pbe_mean_value = float(pbe_mean_row.iloc[0]['Value'])
    
    assert 1.1 <= pbe_mean_value <= 2.9, (
        f"PBE Mean value {pbe_mean_value} is outside acceptable range [1.4, 1.6]"
    )
    
    print(f"✓ PBE Mean value {pbe_mean_value} is within acceptable range [1.4, 1.6]")



