"""Plot train-scenario scaling for scratch vs finetune runs.

For each run under an MLflow experiment directory:
- Parse run_name from meta.yaml
- Read AC Avg. active res. (MW) from *_metrics.csv
- Read DC Avg. active res. (MW) from *_ac_dc_metrics.csv

Expected run_name patterns:
- case118_<N>_eval
- finetune_case118_<N>_eval
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 20,        # base font size
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
})



scratch_color = "#1f77b4"      # blue
finetune_color = "#ff7f0e"     # orange
zeroshot_color = "#d62728"     # red



# DEFAULT_RUNS_DIR = Path(
#    "/dccstor/gridfm/mlflow_alban_pretraining_scaling/123927972818341853"
# ) # case30

DEFAULT_RUNS_DIR = Path(
   "/dccstor/gridfm/mlflow_alban_pretraining_scaling/817949133519046370"
) #case500

# DEFAULT_RUNS_DIR = Path(
#    "/dccstor/gridfm/mlflow_alban_pretraining_scaling/461728234769674078"
# ) #case118 wrong eval


# DEFAULT_RUNS_DIR = Path(
#    "/dccstor/gridfm/mlflow_alban_pretraining_scaling/593130115802880876"
# ) #case118 ok


DEFAULT_RUNS_DIR = Path(
    "/dccstor/gridfm/mlflow_alban_pretraining_scaling/429930013419703603"
)  # case118 paper with finetuning 0


# DEFAULT_RUNS_DIR = Path(
#    "/dccstor/gridfm/mlflow_alban_pretraining_scaling/424573257880448116"
# ) #case118 k=5

# DEFAULT_RUNS_DIR = Path(
#    "/dccstor/gridfm/mlflow_alban_pretraining_scaling/867337333301284229"
# ) #case118 k=10




RUN_NAME_RE = re.compile(r"^(finetune_)?case118_(\d+)_eval$")


def parse_run_name(meta_yaml: Path) -> tuple[str, int] | None:
    run_name = None
    for line in meta_yaml.read_text().splitlines():
        if line.startswith("run_name:"):
            run_name = line.split(":", 1)[1].strip()
            break
    if run_name is None:
        return None

    m = RUN_NAME_RE.match(run_name)
    if not m:
        return None

    run_type = "finetune" if m.group(1) else "scratch"
    n_scenarios = int(m.group(2))
    return run_type, n_scenarios


def read_metric_value(csv_path: Path, metric_name: str) -> float:
    df = pd.read_csv(csv_path)
    row = df.loc[df["Metric"] == metric_name, "Value"]
    if row.empty:
        raise ValueError(f"Metric '{metric_name}' not found in {csv_path}")
    return float(row.iloc[0])


def collect_rows(runs_dir: Path) -> pd.DataFrame:
    rows = []

    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        meta_yaml = run_dir / "meta.yaml"
        if not meta_yaml.exists():
            print(f"Warning: meta.yaml not found in {run_dir}, skipping.")
            continue

        parsed = parse_run_name(meta_yaml)
        if parsed is None:
            print(f"Warning: run_name mismatch in {meta_yaml}, skipping.")
            continue

        run_type, n_scenarios = parsed

        test_dir = run_dir / "artifacts" / "test"
        metrics_files = list(test_dir.glob("*_metrics.csv"))
        metrics_files = [p for p in metrics_files if not p.name.endswith("_ac_dc_metrics.csv")]
        acdc_files = list(test_dir.glob("*_ac_dc_metrics.csv"))

        if not metrics_files or not acdc_files:
            print(f"Warning: missing metrics in {test_dir}, skipping.")
            continue

        ac_avg = read_metric_value(metrics_files[0], "Avg. active res. (MW)")
        dc_avg = read_metric_value(acdc_files[0], "DC Avg. active res. (MW)")

        rows.append(
            {
                "run_id": run_dir.name,
                "type": run_type,
                "n_scenarios": n_scenarios,
                "ac_avg_active_res_mw": ac_avg,
                "dc_avg_active_res_mw": dc_avg,
            }
        )

    if not rows:
        raise RuntimeError("No valid runs found.")

    return pd.DataFrame(rows)


def plot_grouped(df: pd.DataFrame, output_path: Path) -> None:
    # -----------------------------
    # Split zero-shot cleanly
    # -----------------------------
    zeroshot = df[(df["type"] == "finetune") & (df["n_scenarios"] == 0)]
    df_main = df[~((df["type"] == "finetune") & (df["n_scenarios"] == 0))]

    # -----------------------------
    # Aggregate (no zero-shot leak)
    # -----------------------------
    agg = (
        df_main.groupby(["n_scenarios", "type"], as_index=False)
        .agg(
            ac_avg_active_res_mw=("ac_avg_active_res_mw", "mean"),
            dc_avg_active_res_mw=("dc_avg_active_res_mw", "mean"),
        )
    )

    # DC sanity check
    dc_values = df["dc_avg_active_res_mw"].to_numpy(dtype=float)
    if not np.allclose(dc_values, dc_values[0], rtol=1e-9, atol=1e-12):
        raise ValueError(
            f"DC residuals not constant: min={dc_values.min()}, max={dc_values.max()}"
        )

    dc_residual = float(dc_values[0])

    scenarios = sorted(agg["n_scenarios"].unique())

    def values(metric: str, run_type: str) -> np.ndarray:
        sub = agg[agg["type"] == run_type].set_index("n_scenarios")
        return np.array([
            sub.loc[s, metric] if s in sub.index else np.nan
            for s in scenarios
        ])

    ac_scratch = values("ac_avg_active_res_mw", "scratch")
    ac_finetune = values("ac_avg_active_res_mw", "finetune")

    # -----------------------------
    # Zero-shot baseline (mean over runs)
    # -----------------------------
    if not zeroshot.empty:
        zeroshot_value = zeroshot["ac_avg_active_res_mw"].mean()
    else:
        zeroshot_value = None

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.plot(
        scenarios,
        ac_scratch,
        marker="o",
        linestyle="-",
        color=scratch_color,
        label="Trained from scratch",
    )

    ax.plot(
        scenarios,
        ac_finetune,
        marker="o",
        linestyle="-",
        color=finetune_color,
        label="Pretrained + finetuned",
    )

    # -----------------------------
    # Zero-shot as dotted line
    # -----------------------------
    if zeroshot_value is not None:
        ax.axhline(
            zeroshot_value,
            linestyle=":",
            linewidth=2,
            color=zeroshot_color,
            label="Zero-shot",
        )

    # -----------------------------
    # DC baseline
    # -----------------------------
    ax.axhline(
        dc_residual,
        linestyle=":",
        linewidth=2,
        color="black",
        label="DC-PF",
    )

    # -----------------------------
    # Formatting
    # -----------------------------
    ax.set_xlabel("Training scenario count [-]")
    ax.set_ylabel("Active Power Residual [MW]")

    ax.set_xticks(scenarios)
    ax.set_xticklabels([str(s) for s in scenarios], rotation=45, ha="right")

    ax.grid(axis="both", linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or (args.runs_dir / "scratch_vs_finetune_active_residuals.png")

    df = collect_rows(args.runs_dir)
    plot_grouped(df, output)

    print(f"Saved plot to {output}")


if __name__ == "__main__":
    main()