#!/usr/bin/env python3
"""Build Table 5 (OPFData / HH-MPNN vs GENCO Base) from eval metrics CSVs."""

from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

GRIDS = [
    ("case14_ieee", "IEEE 14"),
    ("case30_ieee", "IEEE 30"),
    ("case57_ieee", "IEEE 57"),
    ("case118_ieee", "IEEE 118"),
    ("case500_goc", "GOC 500"),
    ("case2000_goc", "GOC 2000"),
]
SEEDS = (42, 3, 17)

METRIC_KEYS = {
    "gap": "Mean optimality gap (%)",
    "Sf": "Mean branch thermal violation from (MVA)",
    "St": "Mean branch thermal violation to (MVA)",
    "P": "Avg. active res. (MW)",
    "Q": "Avg. reactive res. (MVar)",
    "Qg": "Mean Qg violation",
}

# From Arowolo et al. / paper Table 5. Not retrained.
HH_MPNN = {
    "case14_ieee": dict(gap=0.01, Sf=0.00, St=0.00, P=2.00e-4, Q=2.70e-2, Qg=0.00),
    "case30_ieee": dict(gap=0.18, Sf=5.20e-3, St=3.00e-4, P=1.31e-2, Q=7.60e-3, Qg=0.00),
    "case57_ieee": dict(gap=0.00, Sf=0.00, St=0.00, P=2.10e-2, Q=1.21e-1, Qg=0.00),
    "case118_ieee": dict(gap=0.24, Sf=1.37e-1, St=1.38e-1, P=5.11e-2, Q=1.99e-1, Qg=0.00),
    "case500_goc": dict(gap=0.02, Sf=5.80e-2, St=5.60e-2, P=1.72e-2, Q=1.09e-1, Qg=0.00),
    "case2000_goc": dict(gap=0.00, Sf=2.80e-3, St=2.80e-3, P=1.03e-2, Q=2.65e-2, Qg=0.00),
}


def load_metrics(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["Metric"]] = float(row["Value"])
    return out


def mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


def fmt_gap(mean: float, std: float | None = None) -> str:
    if std is None:
        return f"{mean:.2f}"
    return f"{mean:.2f}\\std{{{std:.2f}}}"


def fmt_sci(value: float, std: float | None = None) -> str:
    if abs(value) < 1e-20 and (std is None or abs(std) < 1e-20):
        return "0.00" if std is None else "0.00"
    if abs(value) < 1e-20:
        body = "0.00"
    else:
        exp = int(math.floor(math.log10(abs(value))))
        mant = value / (10**exp)
        if abs(mant) >= 9.995:
            mant /= 10
            exp += 1
        body = f"{mant:.2f}e{exp:+d}".replace("e+", "e").replace("e-0", "e-").replace("e+0", "e")
        # 8.27e-4 style (no plus, no zero-pad)
        body = f"{mant:.2f}e{exp}"
    if std is None:
        return body
    if abs(std) < 1e-20:
        return f"{body}\\std{{0.00}}"
    sexp = int(math.floor(math.log10(abs(std))))
    smant = std / (10**sexp)
    if abs(smant) >= 9.995:
        smant /= 10
        sexp += 1
    sbody = f"{smant:.2f}e{sexp}"
    return f"{body}\\std{{{sbody}}}"


def better(a: float, b: float) -> bool:
    """True if a is strictly better (lower) than b."""
    return a < b


def cell(text: str, win: bool) -> str:
    return f"\\G{{{text}}}" if win else text


def main() -> None:
    genco: dict[str, dict[str, tuple[float, float]]] = {}
    for grid, _label in GRIDS:
        series: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
        for seed in SEEDS:
            path = RESULTS / grid / f"seed{seed}" / "metrics.csv"
            m = load_metrics(path)
            for k, col in METRIC_KEYS.items():
                series[k].append(m[col])
        genco[grid] = {k: mean_std(vals) for k, vals in series.items()}

    lines = [
        r"\begin{table*}",
        r"\caption{%",
        r"  Constraint violations and optimality gap comparison between GENCO Base and HH-MPNN.",
        r"  \textbf{Bold} indicates the better result per metric per system.",
        r"  $Q_g$ violations are structurally zero for HH-MPNN (reactive limits enforced via sigmoid activation).%",
        r"}",
        r"\label{tab:opf_results}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llllllll@{}}",
        r"\toprule",
        r"& & Optimality & \multicolumn{2}{c}{Thermal limits} & \multicolumn{2}{c}{Power balance} & React.\ gen. bounds \\",
        r"\cmidrule(lr){3-3}\cmidrule(lr){4-6}\cmidrule(lr){6-7}\cmidrule(lr){8-8}",
        r"System & Model",
        r"  & Gap (\%)",
        r"  & $S_{ij}(+)$ [MVA]",
        r"  & $S_{ij}(-)$ [MVA]",
        r"  & $\mathrm{PBRes}_{P}$ [MW]",
        r"  & $\mathrm{PBRes}_{Q}$ [MVar]",
        r"  & $Q_g$ [MVar] \\",
        r"\midrule",
    ]

    keys = ("gap", "Sf", "St", "P", "Q", "Qg")
    for i, (grid, label) in enumerate(GRIDS):
        hh = HH_MPNN[grid]
        ge = genco[grid]
        hh_cells = []
        ge_cells = []
        for k in keys:
            hv, gv = hh[k], ge[k][0]
            hh_win = better(hv, gv)
            ge_win = better(gv, hv)
            if k == "gap":
                hh_txt = fmt_gap(hv)
                ge_mean = f"{gv:.2f}"
                ge_std = f"{ge[k][1]:.2f}"
                hh_cells.append(cell(hh_txt, hh_win))
                ge_cells.append(f"{cell(ge_mean, ge_win)}\\std{{{ge_std}}}")
            else:
                ge_txt = fmt_sci(gv, ge[k][1])
                hh_txt = fmt_sci(hv)
                hh_cells.append(cell(hh_txt, hh_win))
                if "\\std{" in ge_txt:
                    mean_part, rest = ge_txt.split("\\std{", 1)
                    std_part = rest[:-1]
                    ge_cells.append(f"{cell(mean_part, ge_win)}\\std{{{std_part}}}")
                else:
                    ge_cells.append(cell(ge_txt, ge_win))
        lines.append(rf"\multirow{{2}}{{*}}{{{label}}}")
        lines.append("  & HH-MPNN & " + " & ".join(hh_cells) + r" \\")
        lines.append("  & GENCO Base   & " + " & ".join(ge_cells) + r" \\")
        if i < len(GRIDS) - 1:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\\",
            r"\raggedright\footnotesize",
            r"$\pm$ denotes standard deviation over three seeds.",
            r"\end{table*}",
            "",
        ]
    )
    out = ROOT / "table5_genco.tex"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
