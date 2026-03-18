"""Plot train-scenario scaling for scratch vs finetune runs.

For each run under an MLflow experiment directory:
- Parse run_name from meta.yaml
- Read AC Avg. active res. (MW) from *_metrics.csv
- Read DC Avg. active res. (MW) from *_ac_dc_metrics.csv

Expected run_name patterns:
- case30_<N>_eval
- finetune_case30_<N>_eval
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


# DEFAULT_RUNS_DIR = Path(
#    "/dccstor/gridfm/mlflow_alban_pretraining_scaling/424573257880448116"
# ) #case118 k=5

# DEFAULT_RUNS_DIR = Path(
#    "/dccstor/gridfm/mlflow_alban_pretraining_scaling/867337333301284229"
# ) #case118 k=10












RUN_NAME_RE = re.compile(r"^(finetune_)?case500_(\d+)_eval$")


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
            continue

        parsed = parse_run_name(meta_yaml)
        if parsed is None:
            continue
        run_type, n_scenarios = parsed

        test_dir = run_dir / "artifacts" / "test"
        metrics_files = list(test_dir.glob("*_metrics.csv"))
        # Exclude *_ac_dc_metrics.csv
        metrics_files = [p for p in metrics_files if not p.name.endswith("_ac_dc_metrics.csv")]
        acdc_files = list(test_dir.glob("*_ac_dc_metrics.csv"))
        if not metrics_files or not acdc_files:
            continue

        ac_avg_active = read_metric_value(metrics_files[0], "Avg. active res. (MW)")
        dc_avg_active = read_metric_value(acdc_files[0], "DC Avg. active res. (MW)")

        rows.append(
            {
                "run_id": run_dir.name,
                "type": run_type,
                "n_scenarios": n_scenarios,
                "ac_avg_active_res_mw": ac_avg_active,
                "dc_avg_active_res_mw": dc_avg_active,
            }
        )

    if not rows:
        raise RuntimeError(f"No matching runs found in {runs_dir}")
    return pd.DataFrame(rows)


def plot_grouped(df: pd.DataFrame, output_path: Path) -> None:
    # Aggregate in case multiple runs share same type + n_scenarios
    agg = (
        df.groupby(["n_scenarios", "type"], as_index=False)
        .agg(
            ac_avg_active_res_mw=("ac_avg_active_res_mw", "mean"),
            dc_avg_active_res_mw=("dc_avg_active_res_mw", "mean"),
        )
    )
    dc_values = df["dc_avg_active_res_mw"].to_numpy(dtype=float)
    if not np.allclose(dc_values, dc_values[0], rtol=1e-9, atol=1e-12):
        print(df)
        raise ValueError(
            "DC residuals are not constant across runs: "
            f"min={dc_values.min():.6g}, max={dc_values.max():.6g}"
        )
    dc_residual = float(dc_values[0])

    scenario_all = sorted(agg["n_scenarios"].unique())
    scenario_small = [s for s in scenario_all if s in {100, 1000}]
    scenario_rest = [s for s in scenario_all if s not in {100, 1000}]
    width = 0.35

    def values(metric_col: str, run_type: str, scenario_vals: list[int]) -> np.ndarray:
        sub = agg[agg["type"] == run_type].set_index("n_scenarios")
        return np.array([sub.loc[s, metric_col] if s in sub.index else np.nan for s in scenario_vals])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    groups = [
        ("100 & 1000", scenario_small, axes[0]),
        ("Other scenario counts", scenario_rest, axes[1]),
    ]

    for title, scenario_vals, ax in groups:
        x = np.arange(len(scenario_vals))
        ac_scratch = values("ac_avg_active_res_mw", "scratch", scenario_vals)
        ac_finetune = values("ac_avg_active_res_mw", "finetune", scenario_vals)

        ax.bar(x - width / 2, ac_scratch, width, label="Trained from scratch")
        ax.bar(x + width / 2, ac_finetune, width, label="Pretrained + Finetuned")

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in scenario_vals], rotation=0)
        ax.set_xlabel("Training scenarios")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend()

    axes[0].set_ylabel("Residual (MW)")
    fig.suptitle("Scratch vs Finetune", y=1.02)
    fig.text(
        0.5,
        0.98,
        f"DC Avg. active residual (constant across runs): {dc_residual:.6f} MW",
        ha="center",
        va="top",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    runs_dir = args.runs_dir
    output = args.output or (runs_dir / "scratch_vs_finetune_active_residuals.png")
    print("here")
    df = collect_rows(runs_dir)
    plot_grouped(df, output)
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    main()
