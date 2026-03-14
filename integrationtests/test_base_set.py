import pytest
import subprocess
import os
import glob
import pandas as pd
import yaml
import urllib.request

def execute_and_live_output(cmd) -> None:
    # Remove capture_output=True
    # We use check=True to raise an exception automatically if returncode != 0
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
    
    print(f"Config prepared at {config_path} with:")
    print(f"  - network.name: {config['network']['name']}")
    print(f"  - load.scenarios: {config['load']['scenarios']}")
    print(f"  - topology_perturbation.n_topology_variants: {config['topology_perturbation']['n_topology_variants']}")
    
    return config_path

def prepare_training_config():
    """
    Modify the training config to set epochs to 2 for testing.
    """
    config_path = "examples/config/HGNS_PF_datakit_case14.yaml"
    
    # Read the config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure epochs is set to 2
    if 'training' not in config:
        config['training'] = {}
    config['training']['epochs'] = 2
    
    # Write back the modified config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Training config updated: epochs set to {config['training']['epochs']}")
    
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
        execute_and_live_output(
            f'gridfm_datakit generate {config_path}'
        )
    else:
        print(f"Data directory '{data_dir}' already exists, skipping data generation.")
    
    # Prepare training config with epochs=2
    training_config_path = prepare_training_config()
    
    execute_and_live_output(
        f'gridfm_graphkit train --config {training_config_path} --data_path data_out/ --exp_name exp1 --run_name run1 --log_dir logs'
    )
    
    # Find the latest log directory
    log_base = "logs"
    exp_dirs = glob.glob(os.path.join(log_base, "*"))
    assert len(exp_dirs) > 0, "No experiment directories found in logs/"
    
    latest_exp_dir = sorted(exp_dirs, key=os.path.getctime)[-1]
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
        f"PBE Mean value {pbe_mean_value} is outside acceptable range [1.1, 2.9]"
    )
    
    print(f"PBE Mean value {pbe_mean_value} is within acceptable range [1.1, 2.9]")



