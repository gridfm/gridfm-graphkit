from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
_repo_results = ROOT.parents[1] / "results"
RESULTS_DIR = (
    _repo_results
    if (_repo_results / "1.1" / "metrics_summary.csv").exists()
    else ROOT / "results"
)

COLUMNS = ["N", "N-1", "N-2", "Close-to- infeasible"]
COLUMN_TO_ROW = {
    "N": ("n", "feasible"),
    "N-1": ("n-1", "feasible"),
    "N-2": ("n-2", "feasible"),
    "Close-to- infeasible": ("avg", "nose"),
}

# The appendix follows the PF-Delta task grouping. No 2.1 summary is present
# in the GENCO results, so the combined "1.3, 2.1" row uses the 1.3 summary.
TASKS = [
    ("1.1", "1.1"),
    ("1.2", "1.2"),
    ("1.3, 2.1", "1.3"),
    ("2.3", "2.3"),
    ("4.1", "4.1"),
    ("4.2", "4.2"),
    ("4.3", "4.3"),
]

BASELINES = {
    "mean": {
        "1.1": {
            "PFNet": [
                r"3.2\std{0.2\e{-1}}",
                r"5.2\std{0.2\e{-1}}",
                r"5.3\std{0.3\e{-1}}",
                r"1.4\std{0.1\e{0}}",
            ],
            "CANOS-PF": [
                r"2.7\std{0.03\e{-2}}",
                r"7.0\std{0.8\e{-1}}",
                r"5.6\std{0.7\e{-1}}",
                r"1.1\std{0.1\e{0}}",
            ],
            "GNS": [
                r"5.3\std{2.7\e{-1}}",
                r"6.8\std{2.6\e{-1}}",
                r"7.1\std{2.5\e{-1}}",
                r"1.0\std{0.2\e{0}}",
            ],
        },
        "1.2": {
            "PFNet": [
                r"3.2\std{0.2\e{-1}}",
                r"4.0\std{0.2\e{-1}}",
                r"4.4\std{0.2\e{-1}}",
                r"1.2\std{0.05\e{0}}",
            ],
            "CANOS-PF": [
                r"3.8\std{0.2\e{-2}}",
                r"5.9\std{0.2\e{-2}}",
                r"7.7\std{0.3\e{-2}}",
                r"7.5\std{0.5\e{-1}}",
            ],
            "GNS": [
                r"3.6\std{0.3\e{-1}}",
                r"4.0\std{0.3\e{-1}}",
                r"4.1\std{0.3\e{-1}}",
                r"8.2\std{1.1\e{-1}}",
            ],
        },
        "1.3, 2.1": {
            "PFNet": [
                r"3.4\std{0.08\e{-1}}",
                r"4.0\std{0.05\e{-1}}",
                r"4.2\std{0.09\e{-1}}",
                r"1.1\std{0.03\e{0}}",
            ],
            "CANOS-PF": [
                r"4.0\std{0.4\e{-2}}",
                r"5.6\std{0.5\e{-2}}",
                r"6.3\std{0.5\e{-2}}",
                r"7.2\std{0.9\e{-1}}",
            ],
            "GNS": [
                r"3.5\std{0.4\e{-1}}",
                r"3.8\std{0.4\e{-1}}",
                r"3.9\std{0.4\e{-1}}",
                r"7.7\std{0.5\e{-1}}",
            ],
        },
        "2.3": {
            "PFNet": [
                r"4.1\std{0.6\e{-1}}",
                r"4.9\std{0.7\e{-1}}",
                r"5.1\std{0.7\e{-1}}",
                r"1.1\std{0.08\e{0}}",
            ],
            "CANOS-PF": [
                r"9.0\std{1.7\e{-2}}",
                r"1.2\std{0.2\e{-1}}",
                r"1.3\std{0.2\e{-1}}",
                r"8.3\std{0.4\e{-1}}",
            ],
            "GNS": [
                r"8.3\std{2.6\e{-1}}",
                r"8.8\std{2.7\e{-1}}",
                r"8.9\std{2.7\e{-1}}",
                r"1.3\std{0.3\e{0}}",
            ],
        },
        "4.1": {
            "PFNet": [
                r"4.5\std{0.4\e{-1}}",
                r"5.2\std{0.09\e{-1}}",
                r"5.3\std{0.07\e{-1}}",
                r"1.3\std{0.04\e{0}}",
            ],
            "CANOS-PF": [
                r"1.6\std{0.1\e{-1}}",
                r"2.0\std{0.2\e{-1}}",
                r"2.1\std{0.2\e{-1}}",
                r"8.0\std{0.4\e{-1}}",
            ],
            "GNS": [
                r"3.5\std{0.1\e{-1}}",
                r"3.7\std{0.2\e{-1}}",
                r"3.8\std{0.2\e{-1}}",
                r"6.6\std{0.3\e{-1}}",
            ],
        },
        "4.2": {
            "PFNet": [
                r"5.6\std{0.7\e{-1}}",
                r"6.6\std{0.7\e{-1}}",
                r"6.7\std{0.7\e{-1}}",
                r"1.2\std{0.1\e{0}}",
            ],
            "CANOS-PF": [
                r"1.2\std{0.02\e{-1}}",
                r"1.5\std{0.02\e{-1}}",
                r"1.6\std{0.03\e{-1}}",
                r"4.3\std{0.1\e{-1}}",
            ],
            "GNS": [
                r"4.0\std{0.9\e{-1}}",
                r"4.6\std{1.0\e{-1}}",
                r"4.6\std{1.0\e{-1}}",
                r"6.4\std{1.4\e{-1}}",
            ],
        },
        "4.3": {
            "PFNet": [
                r"1.2\std{0.04\e{0}}",
                r"1.4\std{0.04\e{0}}",
                r"1.5\std{0.02\e{0}}",
                r"1.1\std{0.07\e{0}}",
            ],
            "CANOS-PF": [
                r"4.5\std{0.2\e{-1}}",
                r"5.7\std{0.2\e{-1}}",
                r"6.0\std{0.3\e{-1}}",
                r"3.7\std{0.2\e{-1}}",
            ],
            "GNS": [
                r"6.3\std{2.2\e{-1}}",
                r"7.3\std{2.4\e{-1}}",
                r"7.3\std{2.4\e{-1}}",
                r"7.0\std{2.8\e{-1}}",
            ],
        },
    },
    "max": {
        "1.1": {
            "PFNet": [
                r"1.5\std{0.2\e{1}}",
                r"5.7\std{0.3\e{1}}",
                r"5.1\std{0.8\e{1}}",
                r"7.0\std{2.1\e{1}}",
            ],
            "CANOS-PF": [
                r"2.1\std{0.2\e{0}}",
                r"2.4\std{0.2\e{2}}",
                r"2.3\std{0.2\e{2}}",
                r"2.0\std{0.3\e{2}}",
            ],
            "GNS": [
                r"2.3\std{1.0\e{1}}",
                r"8.9\std{4.2\e{1}}",
                r"1.4\std{0.7\e{2}}",
                r"8.4\std{5.6\e{1}}",
            ],
        },
        "1.2": {
            "PFNet": [
                r"1.1\std{0.1\e{1}}",
                r"4.2\std{0.7\e{1}}",
                r"5.0\std{1.3\e{1}}",
                r"3.7\std{0.6\e{1}}",
            ],
            "CANOS-PF": [
                r"1.2\std{0.5\e{0}}",
                r"9.1\std{1.4\e{0}}",
                r"3.6\std{2.4\e{1}}",
                r"4.9\std{2.0\e{1}}",
            ],
            "GNS": [
                r"1.6\std{0.2\e{1}}",
                r"2.1\std{0.08\e{1}}",
                r"1.7\std{0.2\e{1}}",
                r"2.4\std{0.8\e{1}}",
            ],
        },
        "1.3, 2.1": {
            "PFNet": [
                r"1.0\std{0.2\e{1}}",
                r"3.5\std{0.3\e{1}}",
                r"3.5\std{0.4\e{1}}",
                r"3.5\std{0.6\e{1}}",
            ],
            "CANOS-PF": [
                r"1.0\std{0.1\e{0}}",
                r"6.9\std{2.6\e{0}}",
                r"4.5\std{0.7\e{0}}",
                r"4.0\std{0.09\e{1}}",
            ],
            "GNS": [
                r"1.1\std{0.3\e{1}}",
                r"2.1\std{0.3\e{1}}",
                r"1.8\std{0.2\e{1}}",
                r"1.9\std{0.5\e{1}}",
            ],
        },
        "2.3": {
            "PFNet": [
                r"1.1\std{0.2\e{1}}",
                r"4.8\std{1.8\e{1}}",
                r"3.9\std{0.2\e{1}}",
                r"4.3\std{0.7\e{1}}",
            ],
            "CANOS-PF": [
                r"2.2\std{0.3\e{0}}",
                r"9.2\std{1.9\e{0}}",
                r"1.1\std{0.3\e{1}}",
                r"3.4\std{0.6\e{1}}",
            ],
            "GNS": [
                r"3.7\std{3.5\e{1}}",
                r"5.4\std{5.1\e{1}}",
                r"5.3\std{3.8\e{1}}",
                r"4.0\std{2.3\e{1}}",
            ],
        },
        "4.1": {
            "PFNet": [
                r"1.5\std{0.2\e{1}}",
                r"4.5\std{0.2\e{1}}",
                r"4.3\std{0.3\e{1}}",
                r"8.5\std{0.8\e{1}}",
            ],
            "CANOS-PF": [
                r"7.6\std{2.5\e{0}}",
                r"1.5\std{0.1\e{1}}",
                r"2.2\std{0.1\e{1}}",
                r"7.7\std{0.4\e{1}}",
            ],
            "GNS": [
                r"1.2\std{0.2\e{1}}",
                r"2.1\std{0.3\e{1}}",
                r"1.5\std{0.07\e{1}}",
                r"2.0\std{0.3\e{1}}",
            ],
        },
        "4.2": {
            "PFNet": [
                r"1.6\std{0.3\e{1}}",
                r"4.9\std{0.8\e{1}}",
                r"4.1\std{1.2\e{1}}",
                r"8.5\std{0.4\e{1}}",
            ],
            "CANOS-PF": [
                r"4.4\std{1.2\e{0}}",
                r"1.4\std{0.4\e{1}}",
                r"1.6\std{0.5\e{1}}",
                r"7.5\std{2.7\e{1}}",
            ],
            "GNS": [
                r"1.8\std{0.6\e{1}}",
                r"2.4\std{0.3\e{1}}",
                r"2.5\std{1.4\e{1}}",
                r"2.2\std{0.2\e{1}}",
            ],
        },
        "4.3": {
            "PFNet": [
                r"3.3\std{0.8\e{1}}",
                r"6.0\std{2.2\e{1}}",
                r"5.0\std{0.9\e{1}}",
                r"8.1\std{0.3\e{1}}",
            ],
            "CANOS-PF": [
                r"1.2\std{0.3\e{1}}",
                r"5.1\std{3.4\e{1}}",
                r"4.3\std{0.5\e{1}}",
                r"2.5\std{0.4\e{1}}",
            ],
            "GNS": [
                r"2.4\std{1.9\e{1}}",
                r"4.5\std{3.6\e{1}}",
                r"4.1\std{2.3\e{1}}",
                r"2.5\std{0.5\e{1}}",
            ],
        },
    },
}

TABLE_CONFIG = {
    "mean": {
        "source_path": ROOT / "table_a_8.tex",
        "path": ROOT / "table_a_8_genco.tex",
        "metric": "PBE (Mean, p.u.) latex",
        "title": "Power Balance Loss (Mean)",
        "caption": "Power Balance Loss (Mean) across different grid conditions.",
        "label": "table:results-full1",
        "table_type": "tasks",
    },
    "max": {
        "source_path": ROOT / "table_a_9.tex",
        "path": ROOT / "table_a_9_genco.tex",
        "metric": "PBE (Max, p.u.) latex",
        "title": "Power Balance Loss (Max)",
        "caption": "Power Balance Loss (Max) across different grid conditions.",
        "label": "table:results-full2",
        "table_type": "tasks",
    },
    "mean_31": {
        "source_path": ROOT / "table_a_10.tex",
        "path": ROOT / "table_a_10_genco.tex",
        "metric": "PBE (Mean, p.u.) latex",
        "title": "Power Balance Loss (Mean)",
        "caption": "Power Balance Loss (Mean) across different bus sizes and grid conditions.",
        "label": "table:all-cases-mean",
        "table_type": "experiment_31",
        "row_header": "Case",
        "column_header": "Close-to-infeasible",
        "models": ("PFNet", "CANOS-PF", "GNS", "NR"),
    },
    "max_31": {
        "source_path": ROOT / "table_a_11.tex",
        "path": ROOT / "table_a_11_genco.tex",
        "metric": "PBE (Max, p.u.) latex",
        "title": "Power Balance Loss (Max)",
        "caption": "Power Balance Loss (Max) across different bus sizes and grid conditions.",
        "label": "table:all-cases-max",
        "table_type": "experiment_31",
        "row_header": "Case",
        "column_header": "Close-to-infeasible",
        "models": ("PFNet", "CANOS-PF", "GNS"),
    },
}

EXPERIMENT_31_CASES = ("57", "118", "500")
EXPERIMENT_31_BASELINES = {
    "mean_31": {
        "57": {
            "PFNet": [
                r"2.3\std{0.4\e{0}}",
                r"2.3\std{0.4\e{0}}",
                r"2.3\std{0.4\e{0}}",
                r"2.4\std{0.4\e{0}}",
            ],
            "CANOS-PF": [
                r"1.7\std{0.2\e{0}}",
                r"1.8\std{0.2\e{0}}",
                r"1.8\std{0.2\e{0}}",
                r"1.8\std{0.2\e{0}}",
            ],
            "GNS": [
                r"3.3\std{1.4\e{1}}",
                r"8.8\std{3.9\e{1}}",
                r"1.5\std{0.7\e{2}}",
                r"8.0\std{2.0\e{-1}}",
            ],
            "NR": [
                r"1.1\std{0.0\e{-6}}",
                r"1.2\std{0.0\e{-6}}",
                r"1.1\std{0.0\e{-6}}",
                r"1.3\std{0.0\e{-6}}",
            ],
        },
        "118": {
            "PFNet": [
                r"3.4\std{0.08\e{-1}}",
                r"4.0\std{0.05\e{-1}}",
                r"4.2\std{0.09\e{-1}}",
                r"1.1\std{0.03\e{0}}",
            ],
            "CANOS-PF": [
                r"4.0\std{0.4\e{-2}}",
                r"5.6\std{0.5\e{-2}}",
                r"6.3\std{0.5\e{-2}}",
                r"7.2\std{0.9\e{-1}}",
            ],
            "GNS": [
                r"3.5\std{0.4\e{-1}}",
                r"3.8\std{0.4\e{-1}}",
                r"3.9\std{0.4\e{-1}}",
                r"7.7\std{0.5\e{-1}}",
            ],
            "NR": [
                r"3.7\std{0.0\e{-6}}",
                r"3.2\std{0.0\e{-6}}",
                r"3.3\std{0.0\e{-6}}",
                r"4.6\std{0.0\e{-6}}",
            ],
        },
        "500": {
            "PFNet": [
                r"8.3\std{3.9\e{1}}",
                r"8.1\std{3.8\e{1}}",
                r"8.4\std{4.0\e{1}}",
                r"8.9\std{3.8\e{1}}",
            ],
            "CANOS-PF": [
                r"2.3\std{0.6\e{1}}",
                r"2.3\std{0.6\e{1}}",
                r"2.3\std{0.6\e{1}}",
                r"2.6\std{0.6\e{1}}",
            ],
            "GNS": [
                r"2.4\std{0.7\e{1}}",
                r"2.4\std{0.7\e{1}}",
                r"2.4\std{0.7\e{1}}",
                r"2.4\std{0.7\e{1}}",
            ],
            "NR": [
                r"1.4\std{0.0\e{-5}}",
                r"1.4\std{0.0\e{-5}}",
                r"1.3\std{0.0\e{-5}}",
                r"1.6\std{0.0\e{-5}}",
            ],
        },
    },
    "max_31": {
        "57": {
            "PFNet": [
                r"1.2\std{0.2\e{1}}",
                r"1.6\std{0.2\e{1}}",
                r"1.5\std{0.2\e{1}}",
                r"2.1\std{0.2\e{1}}",
            ],
            "CANOS-PF": [
                r"4.5\std{0.5\e{1}}",
                r"4.6\std{0.4\e{1}}",
                r"4.7\std{0.4\e{1}}",
                r"4.6\std{0.3\e{1}}",
            ],
            "GNS": [
                r"3.0\std{1.3\e{5}}",
                r"4.4\std{2.2\e{5}}",
                r"5.2\std{2.4\e{5}}",
                r"4.3\std{1.8\e{1}}",
            ],
        },
        "118": {
            "PFNet": [
                r"1.0\std{0.2\e{1}}",
                r"3.5\std{0.3\e{1}}",
                r"3.5\std{0.4\e{1}}",
                r"3.5\std{0.6\e{1}}",
            ],
            "CANOS-PF": [
                r"1.0\std{0.1\e{0}}",
                r"6.9\std{2.6\e{0}}",
                r"4.5\std{0.7\e{0}}",
                r"4.0\std{0.09\e{1}}",
            ],
            "GNS": [
                r"1.1\std{0.3\e{1}}",
                r"2.1\std{0.3\e{1}}",
                r"1.8\std{0.2\e{1}}",
                r"1.9\std{0.5\e{1}}",
            ],
        },
        "500": {
            "PFNet": [
                r"1.9\std{0.6\e{3}}",
                r"1.9\std{0.6\e{3}}",
                r"1.9\std{0.6\e{3}}",
                r"1.9\std{0.6\e{3}}",
            ],
            "CANOS-PF": [
                r"7.0\std{1.7\e{2}}",
                r"7.5\std{1.8\e{2}}",
                r"7.2\std{1.5\e{2}}",
                r"7.8\std{0.3\e{2}}",
            ],
            "GNS": [
                r"5.9\std{2.3\e{2}}",
                r"5.9\std{2.3\e{2}}",
                r"6.1\std{2.2\e{2}}",
                r"5.8\std{2.1\e{2}}",
            ],
        },
    },
}


def read_genco_values(task: str, metric: str, *, grid: str = "case118") -> list[str]:
    csv_path = RESULTS_DIR / task / "metrics_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metrics summary: {csv_path}")

    values_by_condition: dict[tuple[str, str], str] = {}
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["grid"] != grid:
                continue
            key = (row["perturbation"].strip(), row["feasibility"].strip())
            values_by_condition[key] = row[metric].strip()

    values = []
    for column in COLUMNS:
        key = COLUMN_TO_ROW[column]
        if key not in values_by_condition:
            raise KeyError(f"Missing {column} ({key}) in {csv_path} for {grid}")
        values.append(values_by_condition[key])
    return values


def render_model_row(model: str, values: list[str], *, final: bool = False) -> list[str]:
    end = r" \\ \hline" if final else r" \\"
    return [
        f"        & {model} ",
        f"        & {values[0]} ",
        f"        & {values[1]} ",
        f"        & {values[2]} ",
        f"        & {values[3]}{end}",
    ]


def render_task_table(kind: str) -> str:
    config = TABLE_CONFIG[kind]
    lines = [
        r"\begin{table}[h!]",
        r"    \centering",
        r"    \small",
        r"    \renewcommand{\arraystretch}{1.2}",
        r"    \begin{tabular}{c|c|c|c|c|c} \hline ",
        r"        ",
        rf"        \multicolumn{{2}}{{c|}}{{\textbf{{Experiment}}}} & \multicolumn{{4}}{{|c}}{{\textbf{{{config['title']}}}}}\\ \hline ",
        r"        ",
        r"        \textbf{Task} & ",
        r"        \textbf{Model} & \textbf{N} & \textbf{N-1} ",
        r"        & \textbf{N-2} & \textbf{Close-to- infeasible} \\ \hline",
    ]

    for task_label, result_task in TASKS:
        lines.append("")
        lines.append(rf"        \multirow{{4}}{{*}}{{{task_label}}} ")

        genco_values = read_genco_values(result_task, config["metric"])
        lines.extend(render_model_row("GENCO Base", genco_values))

        for model in ("PFNet", "CANOS-PF", "GNS"):
            final = model == "GNS"
            lines.extend(render_model_row(model, BASELINES[kind][task_label][model], final=final))

    lines.extend(
        [
            r"    \end{tabular}",
            rf"    \caption{{{config['caption']}}}",
            rf"    \label{{{config['label']}}}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def render_experiment_31_table(kind: str) -> str:
    config = TABLE_CONFIG[kind]
    baselines = EXPERIMENT_31_BASELINES[kind]
    model_count = len(config["models"]) + 1
    lines = [
        r"\begin{table}[h!]",
        r"    \centering",
        r"    \small",
        r"    \renewcommand{\arraystretch}{1.2}",
        r"    \begin{tabular}{c|c|c|c|c|c} \hline ",
        r"        ",
        rf"        \multicolumn{{2}}{{c|}}{{\textbf{{Experiment}}}} & \multicolumn{{4}}{{|c}}{{\textbf{{{config['title']}}}}}\\ \hline ",
        r"        ",
        rf"        \textbf{{{config['row_header']}}} & ",
        r"        \textbf{Model} & \textbf{N} & \textbf{N-1} ",
        rf"        & \textbf{{N-2}} & \textbf{{{config['column_header']}}} \\ \hline",
    ]

    for case in EXPERIMENT_31_CASES:
        lines.append("")
        lines.append(rf"        \multirow{{{model_count}}}{{*}}{{{case}}} ")

        genco_values = read_genco_values("3.1", config["metric"], grid=f"case{case}")
        lines.extend(render_model_row("GENCO Base", genco_values))

        baseline_models = config["models"]
        for model in baseline_models:
            final = model == baseline_models[-1]
            lines.extend(render_model_row(model, baselines[case][model], final=final))

    lines.extend(
        [
            r"    \end{tabular}",
            rf"    \caption{{{config['caption']}}}",
            rf"    \label{{{config['label']}}}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def render_table(kind: str) -> str:
    if TABLE_CONFIG[kind]["table_type"] == "experiment_31":
        return render_experiment_31_table(kind)
    return render_task_table(kind)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build PF-Delta LaTeX tables with GENCO Base rows from "
            "results/*/metrics_summary.csv."
        )
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="overwrite source .tex files instead of writing *_genco.tex files",
    )
    args = parser.parse_args()

    for kind, config in TABLE_CONFIG.items():
        output_path = config["source_path"] if args.in_place else config["path"]
        output_path.write_text(render_table(kind))
        print(f"updated {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
