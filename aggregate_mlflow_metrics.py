import argparse
import pandas as pd
import yaml
from pathlib import Path
import re
import numpy as np
import os

METRIC_COLUMNS = [
    "Avg. active res. (p.u.)",
    "Avg. reactive res. (p.u.)",
    "PBE (Mean, p.u.)",
    "PBE (Max, p.u.)",
]

METRIC_COLUMNS_agg = [
    "PBE (Mean, p.u.)",
    "PBE (Max, p.u.)",
]




import numpy as np

def to_latex_std_attached(mean, std, precision_mean=1, precision_std_default=1):
    # determine exponent from mean scale
    if mean == 0:
        exp = 0
    else:
        exp = int(np.floor(np.log10(np.abs(mean))))

    # rescale mean and std
    mean_rescaled = mean / (10 ** exp)
    std_rescaled  = std / (10 ** exp)

    # format mean
    mean_base = f"{mean_rescaled:.{precision_mean}f}"

    # get first digit after decimal of std_rescaled
    std_frac = std_rescaled - np.floor(std_rescaled)
    first_digit = int(np.floor(std_frac * 10))

    # decide precision of std
    precision_std = 2 if first_digit == 0 else precision_std_default

    # format std with comma for decimal
    std_base = f"{std_rescaled:.{precision_std}f}"

    return rf"{mean_base}\std{{{std_base}\e{{{exp}}}}}"

def strip_seed(model_name: str):
    return re.sub(r"_seed\d+$", "", model_name)


def parse_filename(fname: str):
    m = re.match(
        r"test_(case\d+)_(n(?:-\d+)?)_(feasible|nose)_metrics\.csv",
        fname,
    )
    if m is None:
        return None
    return {
        "grid": m.group(1),
        "perturbation": m.group(2),
        "feasibility": m.group(3),
    }


def load_run_name(run_dir: Path):
    meta_path = run_dir / "meta.yaml"
    if not meta_path.exists():
        return None
    with open(meta_path, "r") as f:
        meta = yaml.safe_load(f)
    return meta.get("run_name")


def load_metrics(csv_path: Path):
    df = pd.read_csv(csv_path)
    return dict(zip(df["Metric"], df["Value"]))


# ... [imports and helper functions unchanged] ...

def main(experiment_path: Path, output: str, run_names=None):
    rows = []

    # ------------------------------------------------------------
    # Load per-seed results
    # ------------------------------------------------------------
    for run_dir in experiment_path.iterdir():
        if not run_dir.is_dir():
            continue

        model_name = load_run_name(run_dir)
        if model_name is None:
            continue

        if run_names is not None and model_name not in run_names:
            continue

        test_dir = run_dir / "artifacts" / "test"
        if not test_dir.exists():
            continue

        for csv_path in test_dir.glob("*_metrics.csv"):
            parsed = parse_filename(csv_path.name)
            if parsed is None:
                continue

            metrics = load_metrics(csv_path)

            row = {
                "grid": parsed["grid"],
                "perturbation": parsed["perturbation"],
                "feasibility": parsed["feasibility"],
                "model": model_name,
            }

            for old, new in zip(
                [
                    "Avg. active res. (MW)",
                    "Avg. reactive res. (MVar)",
                    "PBE (Mean, MVA)",
                    "PBE (Max, MVA)",
                ],
                METRIC_COLUMNS,
            ):
                row[new] = metrics.get(old) / 100  # convert to p.u.

            rows.append(row)

    df_detailed = pd.DataFrame(rows)

    # ------------------------------------------------------------
    # Average NOSE rows across perturbations (per-seed)
    # ------------------------------------------------------------
    nose_df = df_detailed[df_detailed["feasibility"] == "nose"]
    non_nose_df = df_detailed[df_detailed["feasibility"] != "nose"]

    nose_avg = (
        nose_df
        .groupby(["grid", "model", "feasibility"], as_index=False)
        .mean(numeric_only=True)
    )
    nose_avg["perturbation"] = "avg"

    df_detailed = pd.concat([non_nose_df, nose_avg], ignore_index=True)

    # ------------------------------------------------------------
    # Sort and save detailed CSV
    # ------------------------------------------------------------
    df_detailed.sort_values(
        by=["grid", "perturbation", "feasibility", "PBE (Mean, p.u.)"],
        inplace=True,
    )
    os.makedirs(Path(output).parent, exist_ok=True)
    df_detailed.to_csv(output + "_detailed.csv", index=False)
    print(f"Saved detailed table to {output}_detailed.csv")

        # ------------------------------------------------------------
    # Aggregate across seeds (mean + std, n-1)
    # ------------------------------------------------------------
    df_agg = df_detailed.copy()
    df_agg["model_base"] = df_agg["model"].apply(strip_seed)
    group_cols = ["grid", "perturbation", "feasibility", "model_base"]

    mean_df = df_agg.groupby(group_cols, as_index=False)[METRIC_COLUMNS_agg].mean()
    std_df = df_agg.groupby(group_cols, as_index=False)[METRIC_COLUMNS_agg].std(ddof=1)
    std_df = std_df.rename(columns={m: f"{m} std" for m in METRIC_COLUMNS_agg})

    df_agg = mean_df.merge(std_df, on=group_cols)
    df_agg = df_agg.rename(columns={"model_base": "model"})

    # LaTeX columns: mean\std{std}
    for col in ["PBE (Mean, p.u.)", "PBE (Max, p.u.)"]:
        df_agg[col + " latex"] = df_agg.apply(
            lambda r: to_latex_std_attached(r[col], r[f"{col} std"]), axis=1
        )

    # Save aggregated CSV
    df_agg.to_csv(output + ".csv", index=False)
    print(f"Saved aggregated table to {output}.csv")

    # ------------------------------------------------------------
    # Plots: use df_detailed to include all seed models
    # ------------------------------------------------------------
    df_plot = df_detailed.copy()
    df_plot["dataset"] = (
        df_plot["grid"]
        + "_"
        + df_plot["perturbation"]
        + "_"
        + df_plot["feasibility"]
    )
    dataset_order = (
        df_plot[["grid", "perturbation", "feasibility", "dataset"]]
        .drop_duplicates()
        .sort_values(["grid", "perturbation", "feasibility"])
    )
    df_plot["dataset"] = pd.Categorical(
        df_plot["dataset"],
        categories=dataset_order["dataset"],
        ordered=True,
    )

    import plotly.express as px

    fig = px.line(
        df_plot.sort_values("dataset"),
        x="dataset",
        y="PBE (Mean, p.u.)",
        color="model",
        markers=True,
    )
    fig.update_yaxes(type="log")
    fig.write_html(output + "_mean.html")

    fig = px.line(
        df_plot.sort_values("dataset"),
        x="dataset",
        y="PBE (Max, p.u.)",
        color="model",
        markers=True,
    )
    fig.update_yaxes(type="log")
    fig.write_html(output + "_max.html")
    
    
    
    # ------------------------------------------------------------
    # Generate LaTeX table rows for mean and max (formatted with newlines)
    # ------------------------------------------------------------
    perturb_order = ["n", "n-1", "n-2", "avg"]  # 'avg' corresponds to nose avg

    for metric in ["PBE (Mean, p.u.) latex", "PBE (Max, p.u.) latex"]:
        row_vals = []
        for perturb in perturb_order:
            # select the row for this perturbation
            r = df_agg[df_agg["perturbation"] == perturb]
            
            # assert there is exactly one row for this perturbation
            assert len(r) == 1, f"Expected 1 row for perturbation {perturb}, got {len(r)}"

            row_vals.append(r.iloc[0][metric])

        # format each column on a separate line with indentation
        row_str = "        & \\graphkit \n"
        for val in row_vals:
            row_str += f"        & {val} \n"
        row_str += "        \\\\ \\hline\n"

        print(f"{metric.replace(' latex','')}:")
        print(row_str)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_path",
        type=Path,
        help="Path to MLflow experiment directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="aggregated_metrics",
    )
    parser.add_argument(
        "--run-names",
        nargs="+",
        default=None,
    )
    args = parser.parse_args()

    main(args.experiment_path, args.output, args.run_names)