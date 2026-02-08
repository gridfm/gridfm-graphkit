#!/usr/bin/env python3
import yaml
from pathlib import Path
import re
import argparse

MLFLOW_BASE = Path("/dccstor/gridfm/mlflow_alban_pfdelta")
CONFIG_DIR = Path("/u/apu/gridfm_model_evaluation/examples/config")
DATA_BASE = Path("/dccstor/gridfm/pfdelta_converted_split_tasks")
VENV_ACTIVATE = "/u/apu/gridfm_model_evaluation/venv/bin/activate"
GRIDFM_DIR = "/u/apu/gridfm_model_evaluation"

def load_meta(exp_dir: Path):
    meta_path = exp_dir / "meta.yaml"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return yaml.safe_load(f)
    
def find_experiment(exp_name: str):
    """Find the MLflow experiment folder by exp_name in meta.yaml"""
    for exp_dir in MLFLOW_BASE.iterdir():
        if not exp_dir.is_dir():
            continue
        meta = load_meta(exp_dir)
        if meta is None:
            continue
        if meta.get("name") == exp_name:
            return exp_dir
    return None

def generate_commands(exp_name: str, model_base: str, task: str):
    exp_dir = find_experiment(exp_name)
    if exp_dir is None:
        print(f"Experiment {exp_name} not found in {MLFLOW_BASE}")
        return

    # Look for runs matching model_base_seedX
    for run_dir in exp_dir.iterdir():
        if not run_dir.is_dir():
            continue
        run_name = load_meta(run_dir).get("run_name") if (run_dir / "meta.yaml").exists() else None
        if run_name is None:
            continue

        m = re.match(rf"^{re.escape(model_base)}_seed(\d+)$", run_name)
        if not m:
            continue

        seed = m.group(1)

        # Model file
        model_file = run_dir / "artifacts" / "model" / "best_model_state_dict.epoch_99.pt"
        if not model_file.exists():
            print(f"Warning: model file not found for run {run_name}")
            continue

        # Config file
        config_file = CONFIG_DIR / f"HGNS_PF_pfdelta_bs64_seed{seed}.yaml"
        if not config_file.exists():
            print(f"Warning: config file {config_file} does not exist")
            continue

        # Data path
        data_path = DATA_BASE / f"task_{task}" / exp_name.split("_")[1]  # assumes grid name is second part

        # Eval run_name
        # Eval run_name
        eval_run_name = f"{model_base}_eval_epoch_99_seed{seed}"

        cmd = (
            f"cd {GRIDFM_DIR}; "
            f"source {VENV_ACTIVATE} && "
            f"gridfm_graphkit evaluate "
            f"--config {config_file} "
            f"--model_path {model_file} "
            f"--data_path {data_path} "
            f"--exp_name {exp_name} "
            f"--run_name {eval_run_name} "
            f"--log_dir {MLFLOW_BASE}"
        )
        print(cmd)
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str, help="MLflow experiment name (meta.yaml name)")
    parser.add_argument("model_base", type=str, help="Model base name (without _seedX)")
    parser.add_argument("task", type=str, help="Task version, e.g., 1.3")
    args = parser.parse_args()

    generate_commands(args.exp_name, args.model_base, args.task)