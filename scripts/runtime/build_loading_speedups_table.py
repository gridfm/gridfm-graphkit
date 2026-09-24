#!/usr/bin/env python3
"""Build the appendix GENCO Tiny loading-speedup table.

Speedup is the classical best per-instance time divided by GENCO Tiny's best
per-instance time. In-memory and from-disk each pick their own best batch or
worker count.
"""

from __future__ import annotations

import csv
from pathlib import Path

from _s1_runtime_plot_data import (
    GENCO_MATRIX_ROOT,
    NETWORKS,
    POWER_MODELS_MATRIX_ROOT,
    best_time,
    load_genco_curve,
)


def best_pm(scope: str, setup: str, network: str, mode: str) -> float:
    csv_path = (
        POWER_MODELS_MATRIX_ROOT / scope / setup / f"benchmark_{network}_{mode}.csv"
    )
    best = None
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            ms = 1_000.0 * float(row["pf_elapsed_s"]) / int(row["n_pfs"])
            best = ms if best is None else min(best, ms)
    if best is None:
        raise ValueError(f"No PowerModels rows in {csv_path}")
    return best


def main() -> None:
    loading_root = GENCO_MATRIX_ROOT.parent / "genco_pf_from_disk"
    print(r"\begin{tabular}{@{}l cc cc@{}}")
    print(r"\toprule")
    print(r"& \multicolumn{2}{c}{vs.\ AC-PF} & \multicolumn{2}{c}{vs.\ AC-OPF} \\")
    print(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    print(r"Grid & In-memory & From-disk & In-memory & From-disk \\")
    print(r"\midrule")
    for network, bus_count, scope in NETWORKS:
        label = f"IEEE {bus_count}" if network.endswith("_ieee") else f"GOC {bus_count:,}".replace(",", "{,}")
        genco_in = best_time(load_genco_curve(network, "tiny", GENCO_MATRIX_ROOT))[1]
        genco_disk = best_time(load_genco_curve(network, "tiny", loading_root))[1]
        cells = [
            best_pm(scope, "setup1", network, "pf") / genco_in,
            best_pm(scope, "setup2", network, "pf") / genco_disk,
            best_pm(scope, "setup1", network, "opf") / genco_in,
            best_pm(scope, "setup2", network, "opf") / genco_disk,
        ]
        a, b, c, d = cells
        print(
            f"{label} & ${a:.1f}\\times$ & ${b:.1f}\\times$ & ${c:.1f}\\times$ & ${d:.1f}\\times$ \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
