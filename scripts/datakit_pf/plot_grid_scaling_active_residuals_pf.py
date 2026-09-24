#!/usr/bin/env python3
"""Regenerate the paper's PF active-residual vs grid-size figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

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
PAPER_YLABEL_PAD = 10
PAPER_SAVE_PAD_INCHES = 0.15

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "results" / "pf_eval_aggregated.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "figures" / "grid_scaling_active_residuals_pf.pdf"

# (csv grid name, bus count, top-axis label)
GRIDS = (
    ("case14", 14, "case\n14"),
    ("case30", 30, "case\n30"),
    ("case57", 57, "case\n57"),
    ("case118", 118, "case\n118"),
    ("case500", 500, "case\n500"),
    ("case2000", 2_000, "case\n2000"),
    ("case10000", 10_000, "case\n10000"),
)

# Large GOC grids: plot tiny/small only (no base).
NO_BASE_GRIDS = {"case2000", "case10000"}

GENCO_COLOR = "#1060f0"
DC_COLOR = "#009870"

GENCO_STYLES = {
    "base": {"linestyle": "-", "label": "GENCO base"},
    "small": {"linestyle": "--", "label": "GENCO small"},
    "tiny": {"linestyle": ":", "label": "GENCO tiny"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def series_from_agg(
    agg: pd.DataFrame,
    metric: str,
    model: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bus_counts, means, stds) for grids that have the requested series."""
    xs: list[int] = []
    ys: list[float] = []
    es: list[float] = []
    for grid, bus_count, _label in GRIDS:
        rows = agg[agg["grid"] == grid]
        if model is not None:
            if grid in NO_BASE_GRIDS and model == "base":
                continue
            rows = rows[rows["model"] == model]
        else:
            # AC/DC are identical across models for a grid; take the first row.
            rows = rows.drop_duplicates("grid")
        if rows.empty:
            continue
        row = rows.iloc[0]
        xs.append(bus_count)
        ys.append(float(row[metric]))
        es.append(float(row[f"{metric} std"]))
    return np.asarray(xs, dtype=float), np.asarray(ys), np.asarray(es)


def main() -> None:
    args = parse_args()
    if not args.csv.is_file():
        raise FileNotFoundError(args.csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    agg = pd.read_csv(args.csv)
    bus_counts = [bus for _g, bus, _l in GRIDS]
    top_labels = [label for _g, _b, label in GRIDS]

    plt.rcParams.update(PAPER_PLOT_RC)

    figure, ax_top = plt.subplots(figsize=PAPER_PLOT_FIGSIZE)

    # --- GENCO (top) ---
    for model, style in GENCO_STYLES.items():
        x, y, e = series_from_agg(agg, "GENCO Avg. active res. (MW)", model=model)
        ax_top.errorbar(
            x,
            y,
            yerr=e,
            color=GENCO_COLOR,
            linestyle=style["linestyle"],
            marker="o",
            markersize=6,
            linewidth=1.8,
            capsize=3,
            label=style["label"],
        )

    # --- DC PF (top) ---
    x_dc, y_dc, e_dc = series_from_agg(agg, "DC Avg. active res. (MW)")
    ax_top.errorbar(
        x_dc,
        y_dc,
        yerr=e_dc,
        color=DC_COLOR,
        linestyle="-",
        marker="s",
        markersize=6,
        linewidth=1.8,
        capsize=3,
        label="DC PF",
    )

    ax_top.set_xscale("log")
    ax_top.set_yscale("log")
    ax_top.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_top.set_xlim(10, 14_000)
    ax_top.set_ylim(1e-3, 1e2)
    ax_top.set_xlabel("Bus count [-]")
    ax_top.set_ylabel("Active power\nresidual [MW]", labelpad=PAPER_YLABEL_PAD)
    ax_top.xaxis.set_major_locator(LogLocator(base=10))
    ax_top.xaxis.set_minor_formatter(NullFormatter())

    # Top twin axis with case labels
    top_axis = ax_top.twiny()
    top_axis.set_xscale("log")
    top_axis.set_xlim(ax_top.get_xlim())
    top_axis.set_xticks(bus_counts)
    top_axis.set_xticklabels(top_labels, fontsize=PAPER_TOP_TICK_FONTSIZE)
    top_axis.tick_params(axis="x", which="major", pad=4)

    # Legend for GENCO variants and DC-PF.
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=GENCO_COLOR,
            linestyle=GENCO_STYLES[m]["linestyle"],
            marker="o",
            markersize=6,
            linewidth=1.8,
            label=GENCO_STYLES[m]["label"],
        )
        for m in ("base", "small", "tiny")
    ] + [
        Line2D(
            [0],
            [0],
            color=DC_COLOR,
            linestyle="-",
            marker="s",
            markersize=6,
            linewidth=1.8,
            label="DC PF",
        ),
    ]
    ax_top.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
        borderaxespad=0.2,
    )

    figure.tight_layout()
    figure.savefig(args.output, bbox_inches="tight", pad_inches=PAPER_SAVE_PAD_INCHES)
    plt.close(figure)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
