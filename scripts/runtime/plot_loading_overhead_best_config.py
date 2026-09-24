#!/usr/bin/env python3
"""Plot loading-inclusive/in-memory runtime at each protocol's best config."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _s1_runtime_plot_data import (
    GENCO_MATRIX_ROOT,
    NETWORKS,
    PAPER_FIGURE_ROOT,
    PAPER_PLOT_FIGSIZE,
    PAPER_PLOT_RC,
    PAPER_SAVE_PAD_INCHES,
    PAPER_YLABEL_PAD,
    POWER_MODELS_MATRIX_ROOT,
)


GENCO_LOADING_ROOT = GENCO_MATRIX_ROOT.parent / "genco_pf_from_disk"
SERIES = (
    ("PM AC-PF", "pf", "#377eb8"),
    ("PM DC-PF", "dcpf", "#8c8caa"),
    ("PM AC-OPF", "opf", "#f0a442"),
    ("PM DC-OPF", "dcopf", "#4daf7c"),
)
GENCO_COLOR = "#c44e9b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genco-in-memory-root", type=Path, default=GENCO_MATRIX_ROOT)
    parser.add_argument("--genco-loading-root", type=Path, default=GENCO_LOADING_ROOT)
    parser.add_argument("--powermodels-root", type=Path, default=POWER_MODELS_MATRIX_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=PAPER_FIGURE_ROOT / "loading_overhead_best_config.pdf",
    )
    return parser.parse_args()


def best_genco_time(root: Path, network: str) -> float:
    group = "ieee" if network.endswith("_ieee") else "goc"
    csv_path = Path(group) / f"benchmark_{network}_tiny.csv"
    values: list[float] = []
    with (root / csv_path).open(newline="") as file:
        for row in csv.DictReader(file):
            if row["status"] == "ok":
                values.append(
                    float(row["outer_elapsed_ms"]) / int(row["num_samples"])
                )
    if not values:
        raise ValueError(f"No successful GENCO rows in {root / csv_path}")
    return min(values)


def best_pm_time(root: Path, scope: str, setup: str, network: str, mode: str) -> float:
    csv_path = root / scope / setup / f"benchmark_{network}_{mode}.csv"
    values: list[float] = []
    with csv_path.open(newline="") as file:
        for row in csv.DictReader(file):
            values.append(1_000.0 * float(row["pf_elapsed_s"]) / int(row["n_pfs"]))
    if not values:
        raise ValueError(f"No PowerModels rows in {csv_path}")
    return min(values)


def main() -> None:
    args = parse_args()
    if not args.output.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {args.output.parent}")

    labels = [str(bus_count) for _network, bus_count, _scope in NETWORKS]
    ratios: dict[str, list[float]] = {label: [] for label, _mode, _color in SERIES}
    ratios["GENCO Tiny"] = []

    for network, _bus_count, scope in NETWORKS:
        for label, mode, _color in SERIES:
            in_memory = best_pm_time(
                args.powermodels_root, scope, "setup1", network, mode
            )
            loading = best_pm_time(
                args.powermodels_root, scope, "setup2", network, mode
            )
            ratios[label].append(loading / in_memory)

        genco_in_memory = best_genco_time(args.genco_in_memory_root, network)
        genco_loading = best_genco_time(args.genco_loading_root, network)
        ratios["GENCO Tiny"].append(genco_loading / genco_in_memory)

    plt.rcParams.update(PAPER_PLOT_RC)
    figure, axis = plt.subplots(figsize=PAPER_PLOT_FIGSIZE)
    x = np.arange(len(labels))
    width = 0.15
    plotted_series = [
        (label, ratios[label], color) for label, _mode, color in SERIES
    ] + [("GENCO Tiny", ratios["GENCO Tiny"], GENCO_COLOR)]

    offsets = (np.arange(len(plotted_series)) - (len(plotted_series) - 1) / 2) * width
    for offset, (label, values, color) in zip(offsets, plotted_series, strict=True):
        axis.bar(x + offset, values, width=width, color=color, label=label)

    axis.axhline(1.0, color="#555555", linestyle="--", linewidth=1.1)
    axis.set_xticks(x, labels)
    axis.set_xlabel("Bus count [-]")
    axis.set_ylabel("Wall-clock time\nratio [-]", labelpad=PAPER_YLABEL_PAD)
    axis.set_ylim(0.0, 4.7)
    axis.grid(axis="y", alpha=0.20)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=5, frameon=False)
    figure.tight_layout(pad=0.45, rect=(0.0, 0.0, 1.0, 0.90))
    figure.savefig(args.output, bbox_inches="tight", pad_inches=PAPER_SAVE_PAD_INCHES)
    plt.close(figure)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
