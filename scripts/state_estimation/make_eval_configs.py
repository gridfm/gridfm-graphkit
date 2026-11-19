"""Write the evaluation configs and commands for one trained model.

Each setting evaluates the model together with the WLS baseline:

- `swype_<p_mask>_<p_outlier>`: measurement masking and outlier rates, on the
  test set `perturbation_0.2_<k>` (training sigma_Y and k). p_mask 0.1, 0.2,
  0.4 give about 8, 7, 5 x |N| measurements. Paper Fig. 5 and 19.
- `unk_pert_perturbation_<sigma>_0`: the scenarios have admittance noise
  sigma_Y, and the estimators get the grid without it. Paper Fig. 6.

The script writes one YAML per setting and `commands.sh`, with one
`gridfm_graphkit evaluate` command per setting. Each command is one MLflow run
in the experiment `--exp_name`. The first command processes
`perturbation_0.0_0`, which holds the grid given to the estimators in the
`unk_pert_*` runs. Run it first. The other commands can run in any order or in
parallel.

Expected data layout (see generate_datakit_configs.py):

    <perturbation_data_dir>/<grid>/perturbation_<sigma>_<k>/<grid>/raw/*.parquet

Usage:
    python scripts/state_estimation/make_eval_configs.py \\
        --grid case14_ieee \\
        --config scripts/state_estimation/configs/case14_ieee_base_0.yaml \\
        --model_path <run>/artifacts/model/best_model_state_dict.pt \\
        --normalizer_stats <run>/artifacts/stats/normalizer_stats.pt \\
        --perturbation_data_dir data/se_perturbations \\
        --out_dir se_eval/case14_ieee \\
        --exp_name se_eval_case14_ieee
"""

import argparse
import os
import shlex
from copy import deepcopy

import yaml

# (p_mask, p_outlier) of the swype_* runs: Fig. 5 cells, plus p_outlier 0.3 at p_mask 0.4 for Fig. 19.
SWEEP = [(m, o) for m in [0.1, 0.2, 0.4] for o in [0.0, 0.1, 0.2]] + [(0.4, 0.3)]
# Admittance noise sigma_Y of the unk_pert_* runs (Fig. 6).
UNKNOWN_SIGMAS = [0.1, 0.3]
TRAIN_MASK_RATIO = 0.2
TRAIN_OUTLIER_RATIO = 0.1
TRAIN_SIGMA = 0.2
MEASUREMENTS = ["power_flow", "power_inj", "vm"]
SMALL_GRIDS = ["case14_ieee", "case30_ieee", "case57_ieee"]
# Test scenarios per setting. Each one is also solved with WLS, which is slow.
NUM_SCENARIOS = 500
EVAL_SEED = 10


def topology_k(grid):
    """The N-k topology perturbation used for training on this grid."""
    return 5 if grid in SMALL_GRIDS else 10


def set_ratios(config, mask_ratio, outlier_ratio):
    for name in MEASUREMENTS:
        config["task"]["measurements"][name]["mask_ratio"] = mask_ratio
        config["task"]["measurements"][name]["outlier_ratio"] = outlier_ratio


def eval_config(train_config, num_scenarios):
    """Turn a training config into a test-only config with the WLS baseline."""
    config = deepcopy(train_config)
    config["data"]["val_ratio"] = 0.0
    config["data"]["test_ratio"] = 0.99
    config["data"]["scenarios"] = [num_scenarios]
    config["data"]["workers"] = 4
    config["task"]["run_pp"] = True
    config["training"]["batch_size"] = 1
    config["training"]["accelerator"] = "cpu"
    config["training"]["devices"] = 1
    config["training"]["strategy"] = "auto"
    # The test loss is only logged. Voltage MAE keeps it cheap.
    config["training"]["losses"] = ["LossPerDim", "LossPerDim"]
    config["training"]["loss_weights"] = [0.5, 0.5]
    config["training"]["loss_args"] = [
        {"dim": "VM", "loss_str": "MAE"},
        {"dim": "VA", "loss_str": "MAE"},
    ]
    config["seed"] = EVAL_SEED
    set_ratios(config, TRAIN_MASK_RATIO, TRAIN_OUTLIER_RATIO)
    return config


def make_settings(train_config, grid, perturbation_data_dir, num_scenarios):
    """Return (run_name, config, data_path) for every setting.

    The first setting runs on perturbation_0.0_0. That run processes the
    dataset that holds the grid given to the estimators in the unk_pert_* runs.
    """
    grid_dir = os.path.join(perturbation_data_dir, grid)
    base_data = os.path.join(grid_dir, f"perturbation_{TRAIN_SIGMA}_{topology_k(grid)}")
    base = eval_config(train_config, num_scenarios)
    settings = []

    # The grid without perturbations is scenario 0 of perturbation_0.0_0.
    reference_grid = os.path.join(
        grid_dir,
        "perturbation_0.0_0",
        grid,
        "processed",
        "data_index_0.pt",
    )
    for sigma in [0.0] + UNKNOWN_SIGMAS:
        config = deepcopy(base)
        config["task"]["grid_path"] = reference_grid
        name = f"perturbation_{sigma}_0"
        settings.append((f"unk_pert_{name}", config, os.path.join(grid_dir, name)))

    for mask_ratio, outlier_ratio in SWEEP:
        config = deepcopy(base)
        set_ratios(config, mask_ratio, outlier_ratio)
        settings.append((f"swype_{mask_ratio}_{outlier_ratio}", config, base_data))

    return settings


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--grid",
        required=True,
        help="Grid name, for example case14_ieee.",
    )
    parser.add_argument("--config", required=True, help="Training config of the model.")
    parser.add_argument(
        "--model_path",
        required=True,
        help="best_model_state_dict.pt of the trained model.",
    )
    parser.add_argument(
        "--normalizer_stats",
        required=True,
        help="normalizer_stats.pt of the training run.",
    )
    parser.add_argument(
        "--perturbation_data_dir",
        required=True,
        help="Folder with one subfolder per grid.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Folder for the configs and commands.sh.",
    )
    parser.add_argument("--exp_name", required=True, help="MLflow experiment name.")
    parser.add_argument("--log_dir", default="mlruns", help="MLflow folder.")
    parser.add_argument("--num_scenarios", type=int, default=NUM_SCENARIOS)
    args = parser.parse_args()

    with open(args.config) as f:
        train_config = yaml.safe_load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    commands = []
    for run_name, config, data_path in make_settings(
        train_config,
        args.grid,
        args.perturbation_data_dir,
        args.num_scenarios,
    ):
        config_path = os.path.join(args.out_dir, f"{run_name}.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        command = [
            "gridfm_graphkit",
            "evaluate",
            "--config",
            config_path,
            "--data_path",
            data_path,
            "--model_path",
            args.model_path,
            "--normalizer_stats",
            args.normalizer_stats,
            "--exp_name",
            args.exp_name,
            "--run_name",
            f"{run_name}.yaml",
            "--log_dir",
            args.log_dir,
        ]
        commands.append(shlex.join(command))

    commands_path = os.path.join(args.out_dir, "commands.sh")
    with open(commands_path, "w") as f:
        f.write("\n".join(commands) + "\n")
    print(f"Wrote {len(commands)} configs and {commands_path}")


if __name__ == "__main__":
    main()
