#!/usr/bin/env python3
"""Regenerate the paper's s1 OPF runtime comparison from raw CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from _s1_runtime_plot_data import (
    GENCO_OPF_MATRIX_ROOT,
    MODELS,
    NETWORKS,
    PAPER_ANNOTATION_FONTSIZE,
    PAPER_FIGURE_ROOT,
    PAPER_PLOT_FIGSIZE,
    PAPER_PLOT_RC,
    PAPER_RUNTIME_YLABEL,
    PAPER_TOP_TICK_FONTSIZE,
    PAPER_YLABEL_PAD,
    POWER_MODELS_MATRIX_ROOT,
    best_time,
    load_genco_curve,
    load_powermodels_curve,
    save_paper_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genco-root", type=Path, default=GENCO_OPF_MATRIX_ROOT)
    parser.add_argument(
        "--powermodels-root",
        type=Path,
        default=POWER_MODELS_MATRIX_ROOT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PAPER_FIGURE_ROOT / "runtime_comparison_from_raw_opf.pdf",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.output.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {args.output.parent}")

    bus_counts = [bus_count for _network, bus_count, _scope in NETWORKS]
    network_labels = [
        f"{'IEEE' if scope == 'small' and bus_count < 500 else 'GOC'} {bus_count}"
        for _network, bus_count, scope in NETWORKS
    ]

    genco_best: dict[str, list[float]] = {model: [] for model in MODELS}
    powermodels_best: dict[str, list[float]] = {"opf": [], "dcopf": []}
    summary_rows: list[tuple[str, str, int, float]] = []

    for network, bus_count, scope in NETWORKS:
        for model in MODELS:
            batch_size, time_ms = best_time(
                load_genco_curve(network, model, args.genco_root)
            )
            genco_best[model].append(time_ms)
            summary_rows.append((str(bus_count), f"GENCO {model}", batch_size, time_ms))

        for mode in ("opf", "dcopf"):
            worker_count, time_ms = best_time(
                load_powermodels_curve(
                    network,
                    scope,
                    mode,
                    args.powermodels_root,
                )
            )
            powermodels_best[mode].append(time_ms)
            summary_rows.append(
                (str(bus_count), f"PowerModels {mode.upper()}", worker_count, time_ms)
            )

    plt.rcParams.update(PAPER_PLOT_RC)
    figure, axis = plt.subplots(figsize=PAPER_PLOT_FIGSIZE)

    pm_opf_line = axis.plot(
        bus_counts,
        powermodels_best["opf"],
        color="#2ca02c",
        linestyle="-",
        marker="D",
        markersize=6,
        linewidth=1.8,
        label="AC-OPF",
    )[0]
    pm_dcopf_line = axis.plot(
        bus_counts,
        powermodels_best["dcopf"],
        color="#2ca02c",
        linestyle=":",
        marker="D",
        markersize=6,
        linewidth=1.8,
        label="DC-OPF",
    )[0]
    genco_lines = []
    for model, linestyle in (("base", "-"), ("small", "--"), ("tiny", ":")):
        line = axis.plot(
            bus_counts,
            genco_best[model],
            color="#1f6fff",
            linestyle=linestyle,
            marker="o",
            markersize=6,
            linewidth=1.8,
            label=model.capitalize(),
        )[0]
        genco_lines.append(line)

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(11.5, 12_000)
    axis.set_ylim(0.0045, 1_000)
    axis.set_xlabel("Bus count [-]")
    axis.set_ylabel(PAPER_RUNTIME_YLABEL, labelpad=PAPER_YLABEL_PAD)
    axis.grid(True, which="both", alpha=0.20)
    axis.plot(
        [500, 10_000],
        [0.007, 0.14],
        color="#555555",
        linestyle="--",
        linewidth=1.3,
    )
    axis.text(
        1_200,
        0.055,
        "slope = 1",
        color="#555555",
        ha="left",
        va="center",
        fontsize=PAPER_ANNOTATION_FONTSIZE,
    )

    pm_legend = axis.legend(
        handles=[pm_opf_line, pm_dcopf_line],
        title="PowerModels",
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=False,
    )
    axis.add_artist(pm_legend)
    axis.legend(
        handles=genco_lines,
        title="GENCO",
        loc="upper left",
        bbox_to_anchor=(0.30, 0.99),
        frameon=False,
    )

    top_axis = axis.twiny()
    top_axis.set_xscale("log")
    top_axis.set_xlim(axis.get_xlim())
    top_axis.set_xticks(bus_counts)
    top_axis.set_xticklabels(
        network_labels, rotation=30, ha="left", fontsize=PAPER_TOP_TICK_FONTSIZE
    )
    top_axis.tick_params(axis="x", which="major", pad=4)

    save_paper_figure(figure, args.output)
    plt.close(figure)
    print(f"Wrote {args.output}")
    print("bus,series,best_configuration,time_ms")
    for row in summary_rows:
        print(f"{row[0]},{row[1]},{row[2]},{row[3]:.9g}")


if __name__ == "__main__":
    main()
