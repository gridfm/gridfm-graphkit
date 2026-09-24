#!/usr/bin/env python3
"""Regenerate the appendix PowerModels worker-count scaling figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

from _s1_runtime_plot_data import (
    NETWORKS,
    PAPER_FIGURE_ROOT,
    PAPER_PLOT_RC,
    PAPER_RUNTIME_YLABEL,
    PAPER_SAVE_PAD_INCHES,
    PAPER_YLABEL_PAD,
    POWER_MODELS_MATRIX_ROOT,
    load_powermodels_curve,
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
MODES = ("pf", "dcpf", "opf", "dcopf")
MODE_TITLES = {
    "pf": "PF",
    "dcpf": "DC-PF",
    "opf": "OPF",
    "dcopf": "DC-OPF",
}
WORKER_TICKS = [24, 56, 88, 120, 152, 184, 216]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--powermodels-root",
        type=Path,
        default=POWER_MODELS_MATRIX_ROOT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PAPER_FIGURE_ROOT / "juliacall_scaling_comparison.pdf",
    )
    return parser.parse_args()


def plot_mode_panel(
    axis: plt.Axes,
    *,
    mode: str,
    matrix_root: Path,
    show_ylabel: bool,
    add_legend_labels: bool,
) -> None:
    for network, bus_count, scope in NETWORKS:
        curve = load_powermodels_curve(network, scope, mode, matrix_root)
        workers, times_ms = zip(*curve)
        axis.plot(
            workers,
            times_ms,
            color=NETWORK_COLORS[bus_count],
            marker="o",
            markersize=4,
            linewidth=1.7,
            label=str(bus_count) if add_legend_labels else "_nolegend_",
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(WORKER_TICKS[0], WORKER_TICKS[-1])
    axis.xaxis.set_major_locator(FixedLocator(WORKER_TICKS))
    axis.xaxis.set_major_formatter(FixedFormatter([str(tick) for tick in WORKER_TICKS]))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_xlabel("Number of workers [-]")
    if show_ylabel:
        axis.set_ylabel(PAPER_RUNTIME_YLABEL, labelpad=PAPER_YLABEL_PAD)
    axis.set_title(MODE_TITLES[mode])
    axis.tick_params(axis="x", rotation=90)
    axis.grid(True, which="both", alpha=0.20)


def main() -> None:
    args = parse_args()
    if not args.output.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {args.output.parent}")

    plt.rcParams.update(
        {
            **PAPER_PLOT_RC,
            "xtick.labelsize": 14,
            "axes.titlesize": PAPER_PLOT_RC["axes.labelsize"],
        }
    )

    figure, axes = plt.subplots(1, len(MODES), figsize=(17.0, 6.0), sharex=True, sharey=True)
    for index, (axis, mode) in enumerate(zip(axes, MODES, strict=True)):
        plot_mode_panel(
            axis,
            mode=mode,
            matrix_root=args.powermodels_root,
            show_ylabel=index == 0,
            add_legend_labels=index == 0,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        title="Bus count",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=len(labels),
        frameon=False,
    )
    figure.tight_layout(pad=0.45, rect=(0.02, 0.0, 1.0, 0.90))
    figure.savefig(args.output, bbox_inches="tight", pad_inches=PAPER_SAVE_PAD_INCHES)
    plt.close(figure)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
