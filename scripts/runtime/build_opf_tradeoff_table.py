#!/usr/bin/env python3
"""Build OPF GENCO tradeoff summary tables (LaTeX) for Small and Tiny.

Mirrors the PF tradeoff builder: no intermediate summary CSV. Runtimes come
from the PF GENCO in-memory sweep (the OPF figure uses the same timings)
and the PowerModels setup-1 matrix. Quality metrics come from
``scripts/datakit_opf/results/opf_scaling_aggregated.csv``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR

DEFAULT_QUALITY_CSV = (
    SCRIPT_DIR.parent / "datakit_opf" / "results" / "opf_scaling_aggregated.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR

GENCO_MODELS = ("small", "tiny")
NETWORK_BY_GRID = {
    14: ("case14_ieee", "small", "14-ieee"),
    30: ("case30_ieee", "small", "30-ieee"),
    57: ("case57_ieee", "small", "57-ieee"),
    118: ("case118_ieee", "small", "118-ieee"),
    500: ("case500_goc", "small", "500-goc"),
    2000: ("case2000_goc", "large", "2000-goc"),
}
FEASIBILITY_COLUMNS = [
    "S_ij(+)",
    "S_ij(-)",
    "Pb",
    "Qb",
    "Qg_violation",
]
QUALITY_MODEL_TEMPLATE = "HGNS_OPF_datakit_case{grid}_{model}_mean_std"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality-csv",
        type=Path,
        default=DEFAULT_QUALITY_CSV,
        help="OPF quality metrics CSV (mean ± std sheet).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output LaTeX files.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=GENCO_MODELS,
        default=list(GENCO_MODELS),
        help="GENCO variants to build tables for (default: small tiny).",
    )
    return parser.parse_args()


def output_path_for_model(output_dir: Path, model: str) -> Path:
    if model == "small":
        return output_dir / "opf_tradeoff_table.tex"
    return output_dir / f"opf_tradeoff_table_{model}.tex"


def label_for(table_id: str, model: str) -> str:
    if model == "small":
        return table_id
    return f"{table_id}_{model}"


def parse_mean(value: object) -> float | None:
    text = str(value).strip()
    if not text or text == "nan":
        return None
    if "±" in text:
        text = text.split("±", 1)[0].strip()
    return float(text)


def load_runtime_from_s1(models: list[str]) -> dict[str, dict[int, float]]:
    """Load best-config latencies from raw s1 GENCO-OPF / PowerModels matrices."""
    if str(BENCHMARK_DIR) not in sys.path:
        sys.path.insert(0, str(BENCHMARK_DIR))
    from _s1_runtime_plot_data import (  # type: ignore
        best_time,
        load_genco_opf_curve,
        load_powermodels_curve,
    )

    runtime: dict[str, dict[int, float]] = {"AC-OPF": {}, "DC-OPF": {}}
    for model in models:
        runtime[model.capitalize()] = {}

    for grid, (network, scope, _system) in NETWORK_BY_GRID.items():
        for model in models:
            _batch, genco_ms = best_time(load_genco_opf_curve(network, model))
            runtime[model.capitalize()][grid] = genco_ms
        for mode, label in (("opf", "AC-OPF"), ("dcopf", "DC-OPF")):
            if grid in runtime[label]:
                continue
            _workers, latency_ms = best_time(
                load_powermodels_curve(network, scope, mode)
            )
            runtime[label][grid] = latency_ms
    return runtime


def load_quality(path: Path) -> dict[tuple[str, str], dict[str, float | None]]:
    columns = ["opt gap", *FEASIBILITY_COLUMNS]
    quality: dict[tuple[str, str], dict[str, float | None]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            quality[(row["System"], row["Model"])] = {
                column: parse_mean(row.get(column, "")) for column in columns
            }
    return quality


def ratio(baseline: float | None, model_value: float | None) -> float | None:
    if baseline is None or model_value is None or model_value <= 0:
        return None
    return baseline / model_value


def mean_feasibility_ratio(
    dc_row: dict[str, float | None],
    genco_row: dict[str, float | None],
) -> float | None:
    ratios = []
    for column in FEASIBILITY_COLUMNS:
        value_ratio = ratio(dc_row.get(column), genco_row.get(column))
        if value_ratio is not None:
            ratios.append(value_ratio)
    if not ratios:
        return None
    return sum(ratios) / len(ratios)


def fmt_speedup(value: float) -> str:
    if value >= 10:
        return f"${value:.1f}\\times$"
    return f"${value:.2f}\\times$"


def fmt_improvement(value: float | None) -> str:
    if value is None:
        return "--"
    if value >= 10:
        return f"${value:.1f}\\times$"
    return f"${value:.2f}\\times$"


def build_rows(
    model: str,
    runtime: dict[str, dict[int, float]],
    quality: dict[tuple[str, str], dict[str, float | None]],
) -> list[dict[str, object]]:
    label = model.capitalize()
    rows: list[dict[str, object]] = []
    for grid, (_network, _scope, system) in NETWORK_BY_GRID.items():
        genco_time = runtime[label][grid]
        ac_time = runtime["AC-OPF"][grid]
        dc_time = runtime["DC-OPF"][grid]
        dc_quality = quality[(system, "DC-OPF")]
        quality_key = (system, QUALITY_MODEL_TEMPLATE.format(grid=grid, model=model))
        genco_quality = quality.get(quality_key)
        if genco_quality is None:
            opt_improvement = None
            feas_improvement = None
        else:
            opt_improvement = ratio(dc_quality["opt gap"], genco_quality["opt gap"])
            feas_improvement = mean_feasibility_ratio(dc_quality, genco_quality)
        rows.append(
            {
                "grid": grid,
                "model": model,
                "speedup_ac": ac_time / genco_time,
                "opt_improvement": opt_improvement,
                "feas_improvement": feas_improvement,
                "speedup_dc": dc_time / genco_time,
            }
        )
    return rows


def build_body(rows: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for row in rows:
        model = str(row["model"]).capitalize()
        opt = row["opt_improvement"]
        feas = row["feas_improvement"]
        lines.append(
            f"{int(row['grid']):4d} & {model:5s} & "
            f"{fmt_speedup(float(row['speedup_ac']))} & "
            f"{fmt_improvement(float(opt) if opt is not None else None)} & "
            f"{fmt_improvement(float(feas) if feas is not None else None)} & "
            f"{fmt_speedup(float(row['speedup_dc']))} \\\\"
        )
    return "\n".join(lines)


def build_latex(model: str, rows: list[dict[str, object]]) -> str:
    body = build_body(rows)
    model_label = model.capitalize()
    return f"""\\begin{{table}}[t]
\\centering
\\scriptsize
\\setlength{{\\tabcolsep}}{{4pt}}
\\renewcommand{{\\arraystretch}}{{1.2}}
% Requires \\usepackage{{booktabs}}.
\\begin{{tabular}}{{@{{}}l l c c c c@{{}}}}
\\toprule
Grid & Model
& \\shortstack{{Speedup\\\\vs AC-OPF}}
& \\shortstack{{Opt.\\ gap $\\downarrow$\\\\vs DC-OPF}}
& \\shortstack{{Feasibility $\\downarrow$\\\\vs DC-OPF}}
& \\shortstack{{Speedup\\\\vs DC-OPF}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{Scaling performance summary for GENCO {model_label} on OPF. Speedups use best-config s1 runtimes; optimality and feasibility improvements are ratios of DC-OPF to GENCO mean metrics when available.}}
\\label{{{label_for("tab:opf_selected_genco_tradeoff", model)}}}
\\end{{table}}
"""


def main() -> int:
    args = parse_args()
    runtime = load_runtime_from_s1(args.models)
    quality = load_quality(args.quality_csv)

    for model in args.models:
        rows = build_rows(model, runtime, quality)
        latex = build_latex(model, rows)
        output = output_path_for_model(args.output_dir, model)
        output.write_text(latex, encoding="utf-8")
        print(latex)
        print(f"\nWrote {output}")
        missing = [
            int(row["grid"])
            for row in rows
            if row["opt_improvement"] is None and row["feas_improvement"] is None
        ]
        if missing:
            print(
                f"Note: no quality metrics for GENCO {model.capitalize()} "
                f"on grids {missing}; opt/feas columns are '--'."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
