"""Load the aligned s1 runtime data used by the paper figures."""

from __future__ import annotations

import csv
import os
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parent
GENCO_MATRIX_ROOT = RUNTIME_ROOT / "genco_pf_in_memory"
# OPF figures use the PF GENCO sweep. The two tasks share one forward pass.
GENCO_OPF_MATRIX_ROOT = GENCO_MATRIX_ROOT
_POWER_MODELS_DEFAULT = (
    RUNTIME_ROOT.parents[1].parent
    / "gridfm-datakit"
    / "scripts"
    / "runtime"
    / "outputs_julia"
    / "full_matrix"
)
POWER_MODELS_MATRIX_ROOT = Path(
    os.environ.get("POWERMODELS_MATRIX_ROOT", _POWER_MODELS_DEFAULT)
)
PAPER_FIGURE_ROOT = RUNTIME_ROOT / "figures"

# Shared typography for paper runtime / residual figures.
PAPER_PLOT_FIGSIZE = (10.5, 6.0)
PAPER_PLOT_RC = {
    "font.size": 18,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "legend.title_fontsize": 18,
}
PAPER_TOP_TICK_FONTSIZE = 16
PAPER_ANNOTATION_FONTSIZE = 15
PAPER_YLABEL_PAD = 10
PAPER_SAVE_PAD_INCHES = 0.15
# Two-line ylabel avoids clipping with large axes.labelsize.
PAPER_RUNTIME_YLABEL = "Amortized per-instance\nruntime [ms]"


def save_paper_figure(figure, path) -> None:
    """Save with padding so large axis labels are not clipped."""
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", pad_inches=PAPER_SAVE_PAD_INCHES)

NETWORKS = (
    ("case14_ieee", 14, "small"),
    ("case30_ieee", 30, "small"),
    ("case57_ieee", 57, "small"),
    ("case118_ieee", 118, "small"),
    ("case500_goc", 500, "small"),
    ("case2000_goc", 2_000, "large"),
    ("case10000_goc", 10_000, "large"),
)
MODELS = ("base", "small", "tiny")


def _load_genco_benchmark_csv(csv_path: Path) -> list[tuple[int, float]]:
    """Return successful ``(batch size, milliseconds/sample)`` rows from a CSV."""
    curve: list[tuple[int, float]] = []
    with csv_path.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["status"] != "ok":
                continue
            elapsed_ms = float(row["outer_elapsed_ms"])
            num_samples = int(row["num_samples"])
            curve.append((int(row["batch_size"]), elapsed_ms / num_samples))

    if not curve:
        raise ValueError(f"No successful GENCO rows in {csv_path}")
    return sorted(curve)


def load_genco_curve(
    network: str,
    model: str,
    matrix_root: Path = GENCO_MATRIX_ROOT,
) -> list[tuple[int, float]]:
    """Return successful ``(batch size, milliseconds/sample)`` rows for PF."""
    group = "ieee" if network.endswith("_ieee") else "goc"
    csv_path = matrix_root / group / f"benchmark_{network}_{model}.csv"
    return _load_genco_benchmark_csv(csv_path)


def load_genco_opf_curve(
    network: str,
    model: str,
    matrix_root: Path = GENCO_OPF_MATRIX_ROOT,
) -> list[tuple[int, float]]:
    """OPF uses the PF GENCO sweep. The forward pass does not depend on the task."""
    return load_genco_curve(network, model, matrix_root)


def load_powermodels_curve(
    network: str,
    scope: str,
    mode: str,
    matrix_root: Path = POWER_MODELS_MATRIX_ROOT,
) -> list[tuple[int, float]]:
    """Return ``(worker count, milliseconds/attempt)`` using the paper metric.

    The PowerModels metric is always ``pf_elapsed_s / n_pfs``. Failed attempts
    therefore remain in the denominator, matching the benchmark protocol.
    """
    csv_path = (
        matrix_root / scope / "setup1" / f"benchmark_{network}_{mode}.csv"
    )

    curve: list[tuple[int, float]] = []
    with csv_path.open(newline="") as file:
        for row in csv.DictReader(file):
            elapsed_s = float(row["pf_elapsed_s"])
            n_pfs = int(row["n_pfs"])
            curve.append((int(row["p"]), 1_000.0 * elapsed_s / n_pfs))

    if not curve:
        raise ValueError(f"No PowerModels rows in {csv_path}")
    return sorted(curve)


def best_time(curve: list[tuple[int, float]]) -> tuple[int, float]:
    """Return the configuration and minimum milliseconds per instance."""
    return min(curve, key=lambda point: point[1])
