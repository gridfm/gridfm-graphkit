#!/usr/bin/env python3
"""Regenerate the appendix GENCO batch-size figure (all networks/models)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _s1_runtime_plot_data import (
    GENCO_MATRIX_ROOT,
    MODELS,
    NETWORKS,
    PAPER_FIGURE_ROOT,
    PAPER_PLOT_FIGSIZE,
    PAPER_PLOT_RC,
    PAPER_RUNTIME_YLABEL,
    PAPER_SAVE_PAD_INCHES,
    PAPER_YLABEL_PAD,
    load_genco_curve,
)


# Sequential palette ordered by increasing bus count.
NETWORK_COLORS = {
    14: "#74a9cf",
    30: "#2b8cbe",
    57: "#045a8d",
    118: "#41ab5d",
    500: "#fec44f",
    2_000: "#ef6548",
    10_000: "#7a0177",
}
MODEL_LINESTYLES = {"base": "-", "small": "--", "tiny": ":"}
MODEL_ORDER = {"base": 0, "small": 1, "tiny": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pf-root", type=Path, default=GENCO_MATRIX_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=PAPER_FIGURE_ROOT / "pf_opf_benchmark_matrix_appendix.pdf",
    )
    return parser.parse_args()


def legend_handles(series_keys: list[tuple[int, str]]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=NETWORK_COLORS[bus_count],
            marker="o",
            markersize=4,
            linewidth=1.7,
            linestyle=MODEL_LINESTYLES[model],
            label=f"{bus_count} - {model}",
        )
        for bus_count, model in series_keys
    ]


def main() -> None:
    args = parse_args()
    if not args.output.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {args.output.parent}")

    plt.rcParams.update(PAPER_PLOT_RC)

    figure, axis = plt.subplots(figsize=PAPER_PLOT_FIGSIZE)
    series_keys: list[tuple[int, str]] = []

    for network, bus_count, _scope in NETWORKS:
        for model in MODELS:
            curve = load_genco_curve(network, model, args.pf_root)
            axis.plot(
                *zip(*curve),
                color=NETWORK_COLORS[bus_count],
                linestyle=MODEL_LINESTYLES[model],
                marker="o",
                markersize=4,
                linewidth=1.7,
            )
            series_keys.append((bus_count, model))

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Batch size [-]")
    axis.set_ylabel(PAPER_RUNTIME_YLABEL, labelpad=PAPER_YLABEL_PAD)
    axis.grid(True, which="both", alpha=0.20)

    figure.legend(
        handles=legend_handles(series_keys),
        title="Bus count - GENCO",
        loc="center left",
        bbox_to_anchor=(0.86, 0.5),
        frameon=False,
    )
    figure.tight_layout(pad=0.35, rect=(0.0, 0.0, 0.82, 1.0))
    figure.savefig(args.output, bbox_inches="tight", pad_inches=PAPER_SAVE_PAD_INCHES)
    plt.close(figure)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
