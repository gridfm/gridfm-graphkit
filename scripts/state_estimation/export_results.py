"""Collect the evaluation runs used in the paper into one CSV.

For every grid and every run of make_eval_configs.py, the CSV has one row per
estimator set (c, pp, nc) and variable (Vm, Va), with the median and mean over
scenarios of the bus-averaged absolute error and the WLS convergence rate.
Parameter estimation runs add rows with set `pe` and variables `r`, `x`.

`make_figures.py --results <csv>` builds the figures from this file.

Usage:
    python scripts/state_estimation/export_results.py --mlflow_dir mlruns --out results.csv \\
        --experiments case14=se_eval_case14_ieee case30=se_eval_case30_ieee \\
                      case57=se_eval_case57_ieee case118=se_eval_case118_ieee \\
        --pe_experiment se_eval_pe
"""

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_eval_configs import SWEEP, UNKNOWN_SIGMAS  # noqa: E402
from make_figures import iter_runs, to_float  # noqa: E402

RUNS = [f"swype_{m}_{o}" for m, o in SWEEP] + [
    f"unk_pert_perturbation_{s}_0" for s in UNKNOWN_SIGMAS
]


def state_rows(mlflow_dir, grid, experiment):
    rows = []
    for name, folder in iter_runs(mlflow_dir, experiment):
        if name not in RUNS:
            continue
        for label in ["c", "pp", "nc"]:
            csvs = glob.glob(
                os.path.join(folder, "artifacts", "test", f"*_metrics_{label}.csv"),
            )
            if not csvs:
                continue
            df = pd.read_csv(csvs[0], index_col=0)
            for variable in ["Vm", "Va"]:
                rows.append(
                    {
                        "grid": grid,
                        "run": name,
                        "set": label,
                        "variable": variable,
                        "median": df.loc[variable, "median_grid"],
                        "mean": df.loc[variable, "mean_grid"],
                        "converged": df["converged"].iloc[0],
                    },
                )
    return rows


def parameter_rows(mlflow_dir, experiment):
    rows = []
    for name, folder in iter_runs(mlflow_dir, experiment):
        csvs = glob.glob(
            os.path.join(folder, "artifacts", "test", "*_parameters_cnc.csv"),
        )
        if not csvs:
            continue
        grid = os.path.basename(csvs[0]).split("_ieee")[0]
        df = pd.read_csv(csvs[0], index_col=0)
        for variable in ["r", "x"]:
            rows.append(
                {
                    "grid": grid,
                    "run": name,
                    "set": "pe",
                    "variable": variable,
                    "median": to_float(df.loc[variable, "relmedian_grid"]),
                    "mean": to_float(df.loc[variable, "relmean_grid"]),
                    "input_mean": to_float(df.loc[variable, "meas_relmean_grid"]),
                },
            )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mlflow_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--experiments",
        nargs=4,
        required=True,
        metavar="GRID=EXPERIMENT",
    )
    parser.add_argument("--pe_experiment", default=None)
    args = parser.parse_args()

    rows = []
    for item in args.experiments:
        grid, experiment = item.split("=", 1)
        rows += state_rows(args.mlflow_dir, grid, experiment)
    if args.pe_experiment:
        rows += parameter_rows(args.mlflow_dir, args.pe_experiment)
    df = pd.DataFrame(rows).sort_values(["grid", "set", "run", "variable"])
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
