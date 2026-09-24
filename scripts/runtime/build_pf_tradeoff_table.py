#!/usr/bin/env python3
"""Build the PF GENCO tradeoff summary table (LaTeX).

Runtimes are always read from the same raw s1 matrix CSVs used by
``scripts/benchmark_inference/plot_runtime_comparison_from_raw_pf.py``
(via ``_s1_runtime_plot_data``). Residuals come from
``pf_eval_aggregated.csv``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR

DEFAULT_RESIDUALS_CSV = (
    SCRIPT_DIR.parent / "datakit_pf" / "results" / "pf_eval_aggregated.csv"
)
DEFAULT_OUTPUT_TEX = SCRIPT_DIR / "pf_tradeoff_table.tex"

GRIDS = [14, 30, 57, 118, 500, 2000, 10000]
GENCO_MODELS = ["base", "small", "tiny"]
GENCO_RUNTIME_LABELS = {"base": "Base", "small": "Small", "tiny": "Tiny"}
MODEL_PREFERENCE = {"tiny": 0, "small": 1, "base": 2}

# Merged utility messages keyed by the first grid in each group.
UTILITY_GROUPS: list[tuple[list[int], str]] = [
    (
        [14, 30, 57, 118, 500],
        "None. Slower than AC-PF so AC-PF should be used instead as it provides more accurate solutions.",
    ),
    (
        [2000, 10000],
        "Much faster than AC-PF; slower than DC-PF with similar residuals, but complete solutions (reactive power and voltage magnitude).",
    ),
]

# Fixed-width Utility column so long recommendations wrap instead of overflowing.
UTILITY_COL_WIDTH = "0.36\\linewidth"
# Rough wrap capacity of the Utility column at \\scriptsize in a single-column table.
_UTILITY_CHARS_PER_LINE = 40
_UTILITY_EM_PER_LINE = 1.25


def _multirow_last_row_suffix(n_rows: int, message: str) -> str:
    """Pad the last row so a wrapped ``\\multirow`` Utility cell fits in-bounds.

    ``\\multirow`` centers text without growing row heights; without this pad,
    long messages spill past ``\\midrule`` / ``\\bottomrule``.
    """
    if n_rows <= 1:
        return " \\\\"
    approx_lines = max(
        1, (len(message) + _UTILITY_CHARS_PER_LINE - 1) // _UTILITY_CHARS_PER_LINE
    )
    # Extra lines beyond what the spanned rows can hold, plus a small slack when
    # the text is about as tall as the span (scriptsize row box vs parbox).
    extra_lines = max(0, approx_lines - n_rows)
    if approx_lines >= n_rows:
        extra_lines += 0.6
    if extra_lines <= 0:
        return " \\\\"
    return f" \\\\[{extra_lines * _UTILITY_EM_PER_LINE:.2f}em]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--residuals-csv",
        type=Path,
        default=DEFAULT_RESIDUALS_CSV,
        help="Aggregated PF residual CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_TEX,
        help="Output LaTeX file path.",
    )
    return parser.parse_args()


def load_runtime_from_s1() -> dict[str, dict[int, float]]:
    """Load best-config latencies from the raw s1 GENCO/PowerModels matrices."""
    if str(BENCHMARK_DIR) not in sys.path:
        sys.path.insert(0, str(BENCHMARK_DIR))
    from _s1_runtime_plot_data import (  # type: ignore
        MODELS,
        NETWORKS,
        best_time,
        load_genco_curve,
        load_powermodels_curve,
    )

    runtime: dict[str, dict[int, float]] = {}
    for network, bus_count, scope in NETWORKS:
        for model in MODELS:
            _batch_size, latency_ms = best_time(load_genco_curve(network, model))
            runtime.setdefault(model.capitalize(), {})[bus_count] = latency_ms
        for mode, label in (("pf", "AC-PF"), ("dcpf", "DC-PF")):
            _workers, latency_ms = best_time(
                load_powermodels_curve(network, scope, mode)
            )
            runtime.setdefault(label, {})[bus_count] = latency_ms
    return runtime


def load_residuals(path: Path) -> dict[tuple[int, str], dict[str, float]]:
    residuals: dict[tuple[int, str], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grid = int(row["grid"].replace("case", ""))
            model = row["model"]
            residuals[(grid, model)] = {
                "genco": float(row["GENCO Avg. active res. (MW)"]),
                "dc": float(row["DC Avg. active res. (MW)"]),
            }
    return residuals


def fmt_speedup(value: float) -> str:
    return f"${value:.1f}\\times$"


def fmt_residual_decrease(value: float) -> str:
    if value >= 10:
        return f"${value:.1f}\\times$"
    return f"${value:.2f}\\times$"


def dc_residual_for_grid(
    grid: int,
    residuals: dict[tuple[int, str], dict[str, float]],
) -> float:
    for model in GENCO_MODELS:
        key = (grid, model)
        if key in residuals:
            return residuals[key]["dc"]
    raise KeyError(f"No DC residual for grid {grid}")


def select_variant(
    grid: int,
    residuals: dict[tuple[int, str], dict[str, float]],
    runtime: dict[str, dict[int, float]],
) -> dict[str, object]:
    ac_time = runtime["AC-PF"][grid]
    dc_time = runtime["DC-PF"][grid]
    dc_residual = dc_residual_for_grid(grid, residuals)

    best: dict[str, object] | None = None
    for model in GENCO_MODELS:
        if (grid, model) not in residuals:
            continue
        label = GENCO_RUNTIME_LABELS[model]
        if grid not in runtime[label]:
            continue
        genco_time = runtime[label][grid]
        genco_residual = residuals[(grid, model)]["genco"]
        speedup_ac = ac_time / genco_time
        speedup_dc = dc_time / genco_time
        residual_decrease = dc_residual / genco_residual
        # Prefer smaller models, then higher AC-PF speedup.
        rank = (-MODEL_PREFERENCE[model], speedup_ac)
        if best is None or rank > best["rank"]:
            best = {
                "grid": grid,
                "model": model,
                "speedup_ac": speedup_ac,
                "speedup_dc": speedup_dc,
                "residual_decrease": residual_decrease,
                "rank": rank,
            }

    if best is None:
        raise RuntimeError(f"No selectable GENCO variant for grid {grid}")
    return best


def format_metric_cells(row: dict[str, object]) -> str:
    model = str(row["model"]).capitalize()
    return (
        f"{int(row['grid']):4d} & {model:5s} & "
        f"{fmt_speedup(float(row['speedup_ac']))} & "
        f"{fmt_residual_decrease(float(row['residual_decrease']))} & "
        f"{fmt_speedup(float(row['speedup_dc']))}"
    )


def build_body(rows: list[dict[str, object]]) -> str:
    """Metric rows with a wrapping Utility column merged across each regime."""
    by_grid = {int(row["grid"]): row for row in rows}
    lines: list[str] = []
    for group_index, (grids, message) in enumerate(UTILITY_GROUPS):
        n_rows = len(grids)
        for index, grid in enumerate(grids):
            if index == 0 and n_rows > 1:
                # "=" uses the p-column width so the merged cell wraps.
                utility = f"\\multirow{{{n_rows}}}{{=}}{{{message}}}"
            elif index == 0:
                utility = message
            else:
                utility = ""
            if index == n_rows - 1:
                suffix = _multirow_last_row_suffix(n_rows, message)
            else:
                suffix = " \\\\"
            lines.append(f"{format_metric_cells(by_grid[grid])} & {utility}{suffix}")
        if group_index < len(UTILITY_GROUPS) - 1:
            lines.append("\\midrule")
    return "\n".join(lines)


def build_latex(rows: list[dict[str, object]]) -> str:
    body = build_body(rows)
    return f"""\\begin{{table}}[t]
\\centering
\\scriptsize
\\setlength{{\\tabcolsep}}{{4pt}}
\\renewcommand{{\\arraystretch}}{{1.2}}
% Requires \\usepackage{{booktabs,multirow,array}}.
\\begin{{tabular}}{{@{{}}l l c c c >{{\\raggedright\\arraybackslash}}p{{{UTILITY_COL_WIDTH}}}@{{}}}}
\\toprule
Grid & Model
& \\shortstack{{Speedup\\\\vs AC-PF}}
& \\shortstack{{Residual $\\downarrow$\\\\vs DC-PF}}
& \\shortstack{{Speedup\\\\vs DC-PF}}
& Utility \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{Scaling performance summary for the selected GENCO variant. The Utility column summarizes the recommended use of GENCO relative to AC-PF and DC-PF for each grid-size regime.}}
\\label{{tab:pf_selected_genco_tradeoff}}
\\end{{table}}
"""


def main() -> int:
    args = parse_args()
    residuals = load_residuals(args.residuals_csv)
    runtime = load_runtime_from_s1()

    rows = [select_variant(grid, residuals, runtime) for grid in GRIDS]

    latex = build_latex(rows)
    args.output.write_text(latex, encoding="utf-8")
    print(latex)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
