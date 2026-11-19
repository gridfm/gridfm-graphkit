"""Write gridfm-datakit configs for the state estimation test sets.

For each IEEE grid this writes four datakit configs:

    <out_dir>/<grid>/perturbation_<sigma>_<k>/config.yaml

- `perturbation_0.2_<k>`: the admittance noise (sigma_Y = 0.2) and N-k topology
  level of training, with k = 5 (10 on IEEE 118). Paper Fig. 5 and 19.
- `perturbation_0.1_0`, `perturbation_0.3_0`: admittance noise sigma_Y, no
  topology changes. Paper Fig. 6.
- `perturbation_0.0_0`: no perturbation. In the Fig. 6 runs the estimators get
  this grid instead of the perturbed one.

Each config starts from the datakit `args.log` of the grid's training dataset.
It keeps the load model and changes only:

- `admittance_perturbation.sigma` to sigma_Y,
- `topology_perturbation.k` to k (k = 0 disables topology changes),
- one topology variant per load scenario,
- loads scaled from the largest training scaling factor `u` down to 1.0.

Generate each dataset with gridfm-datakit:

    gridfm_datakit generate <out_dir>/<grid>/perturbation_<sigma>_<k>/config.yaml

Usage:
    python scripts/state_estimation/generate_datakit_configs.py \\
        --train_data_dir data --out_dir data/se_perturbations
"""

import argparse
import os

import yaml

GRIDS = ["case14_ieee", "case30_ieee", "case57_ieee", "case118_ieee"]
SMALL_GRIDS = ["case14_ieee", "case30_ieee", "case57_ieee"]


def perturbations(grid):
    """(sigma_Y, k) of every test set of the grid."""
    train_k = 5 if grid in SMALL_GRIDS else 10
    return [(0.2, train_k), (0.1, 0), (0.3, 0), (0.0, 0)]


def read_args_log(path):
    """Datakit args.log: a header line, a timestamp line, then the YAML config."""
    with open(path) as f:
        return yaml.safe_load("".join(f.readlines()[2:]))


def read_upper_scaling(path):
    """Upper load scaling factor `u` from scenarios_agg_load_profile.log."""
    with open(path) as f:
        lines = f.readlines()
    u = float(lines[2].strip()[2:])
    assert u > 1
    return u


def make_config(train_config, sigma, k, data_dir, upper_scaling):
    config = yaml.safe_load(yaml.safe_dump(train_config))
    if k != 0:
        config["topology_perturbation"]["k"] = k
    else:
        config["topology_perturbation"]["type"] = "none"
    config["topology_perturbation"]["n_topology_variants"] = 1
    config["admittance_perturbation"]["sigma"] = sigma
    config["settings"]["data_dir"] = data_dir
    config["load"]["start_scaling_factor"] = upper_scaling
    config["load"]["max_scaling_factor"] = 1.0
    return config


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--train_data_dir",
        required=True,
        help="Folder with <grid>/raw/args.log and scenarios_agg_load_profile.log of the training datasets.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Folder for the test datasets.",
    )
    parser.add_argument(
        "--grids",
        nargs="+",
        default=GRIDS,
        choices=GRIDS,
        help="Grids to write configs for. Default: all four IEEE grids.",
    )
    args = parser.parse_args()

    for grid in args.grids:
        raw_dir = os.path.join(args.train_data_dir, grid, "raw")
        train_config = read_args_log(os.path.join(raw_dir, "args.log"))
        upper_scaling = read_upper_scaling(
            os.path.join(raw_dir, "scenarios_agg_load_profile.log"),
        )
        for sigma, k in perturbations(grid):
            perturbation_dir = os.path.join(
                args.out_dir,
                grid,
                f"perturbation_{sigma}_{k}",
            )
            os.makedirs(perturbation_dir, exist_ok=True)
            config = make_config(
                train_config,
                sigma,
                k,
                perturbation_dir,
                upper_scaling,
            )
            with open(os.path.join(perturbation_dir, "config.yaml"), "w") as f:
                yaml.safe_dump(config, f)
        print(f"{grid}: wrote {len(perturbations(grid))} configs")


if __name__ == "__main__":
    main()
