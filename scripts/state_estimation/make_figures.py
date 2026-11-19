"""Build the state estimation figures of the paper from the evaluation runs.

Inputs are MLflow runs written by the commands of make_eval_configs.py, one
experiment per grid, or the CSV written from them by export_results.py. Every run has `artifacts/test/<grid>_metrics_<set>.csv`,
where <set> is c, pp, nc or cnc (see StateEstimationTask).

Outputs in --out_dir, as PDF and CSV:

- `fig5_base_case`: WLS / GENCO error ratio for 8, 7 and 5 x |N|
  measurements and p_outlier 0.0 to 0.2, plus the WLS convergence rate.
- `fig6_unknown_perturbation`: the same ratio when the scenarios have
  admittance noise sigma_Y and the estimators get the grid without it.
- `fig19_non_observable`: error of GENCO where WLS did not converge, divided
  by its error where WLS converged, at 5 x |N|.
- `parameter_estimation.csv`: relative branch r, x errors of the joint state and
  parameter estimator, and of its noisy input (needs --pe_experiment).

Every ratio uses the median over scenarios of the bus-averaged absolute error.
Figure numbers follow arXiv:2608.09921.

Usage, from the evaluation results of the paper:
    python scripts/state_estimation/make_figures.py --results scripts/state_estimation/results/se_evaluation_results.csv \\
        --out_dir se_figures

Usage, from new evaluation runs:
    python scripts/state_estimation/make_figures.py \\
        --mlflow_dir mlruns --out_dir se_figures \\
        --experiments case14=se_eval_case14_ieee case30=se_eval_case30_ieee \\
                      case57=se_eval_case57_ieee case118=se_eval_case118_ieee
"""

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

# Voltage magnitude and angle, as named in the metrics CSVs.
FEATURES = ["Vm", "Va"]
FEATURE_TITLES = {"Vm": "$V_m$", "Va": r"$\theta$"}
GRIDS = ["case14", "case30", "case57", "case118"]
GRID_LABELS = ["14", "30", "57", "118"]
# A grid has about 9 (1 - p_mask) |N| measurements.
MEASUREMENTS_PER_BUS = {0.1: 8, 0.2: 7, 0.4: 5}
FIG5_OUTLIERS = [0.0, 0.1, 0.2]
FIG6_SIGMAS = [0.1, 0.3]
FIG19_OUTLIERS = [0.0, 0.1, 0.2, 0.3]


# ---------------------------------------------------------------- reading runs


def find_experiment(mlflow_dir, name):
    """Folder of the MLflow experiment called `name`."""
    for meta_path in glob.glob(os.path.join(mlflow_dir, "*", "meta.yaml")):
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        if meta.get("lifecycle_stage") != "deleted" and meta.get("name") == name:
            return os.path.dirname(meta_path)
    raise ValueError(f"No MLflow experiment named {name} in {mlflow_dir}")


# MLflow run status of a finished run.
FINISHED = 3


def iter_runs(mlflow_dir, experiment_name):
    """(run name, run folder) of every finished run with test results.

    The `.yaml` suffix is removed from run names. If several runs have the same
    name, the newest one is used.
    """
    runs = []
    for meta_path in glob.glob(
        os.path.join(find_experiment(mlflow_dir, experiment_name), "*", "meta.yaml"),
    ):
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        folder = os.path.dirname(meta_path)
        if (
            meta.get("lifecycle_stage") == "deleted"
            or meta.get("status", FINISHED) != FINISHED
            or "run_name" not in meta
            or not glob.glob(os.path.join(folder, "artifacts", "test", "*.csv"))
        ):
            continue
        name = str(meta["run_name"]).removesuffix(".yaml")
        runs.append((meta.get("start_time") or 0, name, folder))

    seen = set()
    for _, name, folder in sorted(runs, reverse=True):
        if name not in seen:
            seen.add(name)
            yield name, folder


def read_runs(mlflow_dir, experiment_name, label):
    """One row per run: median, mean, std of the per-scenario error and the WLS convergence rate."""
    rows = {}
    for name, folder in iter_runs(mlflow_dir, experiment_name):
        csvs = glob.glob(
            os.path.join(folder, "artifacts", "test", f"*_metrics_{label}.csv"),
        )
        if not csvs:
            continue
        df = pd.read_csv(csvs[0], index_col=0)
        row = {
            "converged": float(df["converged"].iloc[0]) if "converged" in df else 1.0,
        }
        for feature in FEATURES:
            for stat in ["median", "mean", "std"]:
                row[f"{feature}_{stat}"] = lookup(df, feature, f"{stat}_grid")
        rows[name] = row
    columns = ["converged"] + [
        f"{f}_{s}" for f in FEATURES for s in ["median", "mean", "std"]
    ]
    return pd.DataFrame.from_dict(rows, orient="index", columns=columns)


def read_results_csv(path):
    """Tables of read_runs and read_parameter_runs from a CSV written by export_results.py."""
    df = pd.read_csv(path)
    results = {}
    for grid in GRIDS:
        results[grid] = {}
        for label in ["c", "pp", "nc"]:
            part = df[(df["grid"] == grid) & (df["set"] == label)]
            rows = {}
            for run, group in part.groupby("run"):
                row = {"converged": group["converged"].iloc[0]}
                for _, r in group.iterrows():
                    row[f"{r['variable']}_median"] = r["median"]
                    row[f"{r['variable']}_mean"] = r["mean"]
                rows[run] = row
            results[grid][label] = pd.DataFrame.from_dict(rows, orient="index")
    pe_rows = {}
    for _, r in df[df["set"] == "pe"].iterrows():
        row = pe_rows.setdefault(r["run"], {})
        row[f"{r['variable']}_rel_error"] = r["mean"]
        row[f"{r['variable']}_input_rel_error"] = r["input_mean"]
        row[f"{r['variable']}_reduction"] = 1 - r["mean"] / r["input_mean"]
    pe = (
        pd.DataFrame.from_dict(pe_rows, orient="index").sort_index()
        if pe_rows
        else None
    )
    return results, pe


def read_parameter_runs(mlflow_dir, experiment_name):
    """Branch r, x relative errors of the prediction and of the noisy input, one row per run.

    `reduction` is 1 - (prediction error / input error), the noise reduction in the paper text.
    """
    rows = {}
    for name, folder in iter_runs(mlflow_dir, experiment_name):
        params = glob.glob(
            os.path.join(folder, "artifacts", "test", "*_parameters_cnc.csv"),
        )
        if not params:
            continue
        pe = pd.read_csv(params[0], index_col=0)
        row = {}
        for feature in ["r", "x"]:
            predicted = to_float(pe.loc[feature, "relmean_grid"])
            noisy_input = to_float(pe.loc[feature, "meas_relmean_grid"])
            row[f"{feature}_rel_error"] = predicted
            row[f"{feature}_input_rel_error"] = noisy_input
            row[f"{feature}_reduction"] = 1 - predicted / noisy_input
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def to_float(value):
    """A float from a CSV cell. Cells written as "tensor(0.0571)" are also read."""
    if isinstance(value, str):
        return float(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)[0])
    return float(value)


def sweep_name(p_mask, p_outlier):
    return f"swype_{p_mask}_{p_outlier}"


def perturbation_name(sigma, k):
    return f"unk_pert_perturbation_{sigma}_{k}"


def lookup(df, row, column):
    """df.loc[row, column], or NaN if the run or set is missing.

    The `nc` set is missing when WLS converged on every scenario of a run.
    """
    if row in df.index and column in df.columns:
        return df.loc[row, column]
    return np.nan


def ratio_table(numerator, denominator, rows, row_labels, metric="median"):
    """numerator / denominator per feature and grid, plus the WLS convergence rate."""
    table = {}
    for feature in FEATURES:
        for grid in GRIDS:
            table[(feature, grid)] = [
                lookup(numerator[grid], r, f"{feature}_{metric}")
                / lookup(denominator[grid], r, f"{feature}_{metric}")
                for r in rows
            ]
    for grid in GRIDS:
        table[("converged", grid)] = [
            100 * lookup(denominator[grid], r, "converged") for r in rows
        ]
    df = pd.DataFrame(table, index=row_labels)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def fig5_table(results):
    rows, labels = [], []
    for p_mask, n_meas in MEASUREMENTS_PER_BUS.items():
        for p_outlier in FIG5_OUTLIERS:
            rows.append(sweep_name(p_mask, p_outlier))
            labels.append(f"{n_meas}, {p_outlier}")
    pp = {g: results[g]["pp"] for g in GRIDS}
    nn = {g: results[g]["c"] for g in GRIDS}
    return ratio_table(pp, nn, rows, labels)


def fig6_table(results):
    rows = [perturbation_name(sigma, 0) for sigma in FIG6_SIGMAS]
    pp = {g: results[g]["pp"] for g in GRIDS}
    nn = {g: results[g]["c"] for g in GRIDS}
    return ratio_table(pp, nn, rows, [str(s) for s in FIG6_SIGMAS])


def fig19_table(results):
    rows = [sweep_name(0.4, p) for p in FIG19_OUTLIERS]
    nc = {g: results[g]["nc"] for g in GRIDS}
    c = {g: results[g]["c"] for g in GRIDS}
    return ratio_table(nc, c, rows, [f"5, {p}" for p in FIG19_OUTLIERS])


# ------------------------------------------------------------------- plotting


def heatmap(
    table,
    path,
    row_title,
    reverse=False,
    title="Error reduction factor [-]",
    decimals=1,
):
    """Ratio panels on a log color scale centred at 1, then the WLS convergence panel."""
    vmin, vmax = 0.1, 20
    mid = (np.log10(1.0) - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))
    colors = [(0.0, "red"), (mid, "white"), (1.0, "green")]
    if reverse:
        colors = [(0.0, "green"), (mid, "white"), (1.0, "red")]
    cmap = mcolors.LinearSegmentedColormap.from_list("ratio", colors)

    panels = FEATURES + ["converged"]
    n_rows = table.shape[0]
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(2.2 * len(panels), 0.45 * n_rows + 1.5),
    )
    for ax, panel in zip(axes, panels):
        values = table[panel][GRIDS].values.astype(float)
        if panel == "converged":
            ax.imshow(values, vmin=0, vmax=100, aspect="auto")
            fmt = "{:.0f}"
            ax.set_title("WLS conv. (%)")
        else:
            ax.imshow(
                values,
                cmap=cmap,
                norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
                aspect="auto",
            )
            fmt = f"{{:.{decimals}f}}"
            ax.set_title(FEATURE_TITLES[panel])
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isfinite(values[i, j]):
                    ax.text(
                        j,
                        i,
                        fmt.format(values[i, j]),
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
        ax.set_xticks(range(len(GRIDS)))
        ax.set_xticklabels(GRID_LABELS)
        ax.set_yticks([])
    axes[0].set_yticks(range(n_rows))
    axes[0].set_yticklabels(table.index)
    axes[0].set_ylabel(row_title)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--results",
        default=None,
        help="CSV written by export_results.py. Used instead of --mlflow_dir and --experiments.",
    )
    parser.add_argument("--mlflow_dir", default=None)
    parser.add_argument(
        "--experiments",
        nargs=4,
        default=None,
        metavar="GRID=EXPERIMENT",
        help="Evaluation experiment of each IEEE grid, for example case14=se_eval_case14_ieee.",
    )
    parser.add_argument(
        "--pe_experiment",
        default=None,
        help="Experiment with the parameter estimation runs.",
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.results:
        results, pe = read_results_csv(args.results)
    else:
        if not (args.mlflow_dir and args.experiments):
            parser.error("pass --results, or --mlflow_dir and --experiments")
        experiments = dict(e.split("=", 1) for e in args.experiments)
        assert set(experiments) == set(GRIDS), f"--experiments needs exactly {GRIDS}"
        results = {
            grid: {
                label: read_runs(args.mlflow_dir, name, label)
                for label in ["c", "pp", "nc"]
            }
            for grid, name in experiments.items()
        }
        pe = (
            read_parameter_runs(args.mlflow_dir, args.pe_experiment)
            if args.pe_experiment
            else None
        )

    tables = {
        "fig5": fig5_table(results),
        "fig6": fig6_table(results),
        "fig19": fig19_table(results),
    }
    heatmap(
        tables["fig5"],
        os.path.join(args.out_dir, "fig5_base_case.pdf"),
        "# meas. (x|N|), $p_{outlier}$",
    )
    heatmap(
        tables["fig6"],
        os.path.join(args.out_dir, "fig6_unknown_perturbation.pdf"),
        r"$\sigma_Y$",
    )
    heatmap(
        tables["fig19"],
        os.path.join(args.out_dir, "fig19_non_observable.pdf"),
        "# meas. (x|N|), $p_{outlier}$",
        reverse=True,
        title="Degradation factor [-]",
        decimals=2,
    )
    for name, table in tables.items():
        table.to_csv(os.path.join(args.out_dir, f"{name}.csv"))

    if pe is not None:
        pe.to_csv(os.path.join(args.out_dir, "parameter_estimation.csv"))
        print(pe.to_string())


if __name__ == "__main__":
    main()
