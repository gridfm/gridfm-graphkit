"""Write the training configs for the state estimation experiments.

For each grid this writes two configs into `configs/`:

- `<grid>_base_0.yaml`: state estimation.
- `<grid>_pe_0.yaml`: state estimation plus branch r, x estimation.
  The input r and x are noised with U(0.8, 1.2).

All configs start from `base.yaml` and use seed 0.

Usage:
    python scripts/state_estimation/generate_configs.py
"""

import os
from copy import deepcopy

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
GRIDS = ["case14_ieee", "case30_ieee", "case57_ieee", "case118_ieee"]
SEED = 0
PARAMETER_NOISE = 0.2


def make_configs(base_config, grid):
    config = deepcopy(base_config)
    config["data"]["networks"] = [grid]
    config["seed"] = SEED
    base = deepcopy(config)

    config["task"]["parameter_estimation"] = True
    config["task"]["grid_noise_r"] = PARAMETER_NOISE
    config["task"]["grid_noise_x"] = PARAMETER_NOISE
    config["training"]["losses"].append("LossPerDim")
    config["training"]["loss_args"].append({"dim": "impedance", "loss_str": "MAE"})
    config["training"]["loss_weights"].append(0.1)
    return base, config


if __name__ == "__main__":
    with open(os.path.join(HERE, "base.yaml")) as f:
        base_config = yaml.safe_load(f)

    out_dir = os.path.join(HERE, "configs")
    os.makedirs(out_dir, exist_ok=True)
    for grid in GRIDS:
        for name, config in zip(["base", "pe"], make_configs(base_config, grid)):
            with open(os.path.join(out_dir, f"{grid}_{name}_{SEED}.yaml"), "w") as f:
                yaml.safe_dump(config, f)
