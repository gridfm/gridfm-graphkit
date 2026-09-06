# %%


# %%

result_type = "mean" # "mean" or "max"


latex_table_max = r"""\begin{table}[h!]
    \centering
    \small
    \renewcommand{\arraystretch}{1.2}
    \begin{tabular}{c|c|c|c|c|c} \hline 
        
        \multicolumn{2}{c|}{\textbf{Experiment}} & \multicolumn{4}{|c}{\textbf{Power Balance Loss (Max)}}\\ \hline 
        
        \textbf{Task} & 
        \textbf{Model} & \textbf{N} & \textbf{N-1} 
        & \textbf{N-2} & \textbf{Close-to- infeasible} \\ \hline

        \multirow{3}{*}{1.1} 
        & PFNet 
        & 1.5\std{0.2\e{1}}
        & 5.7\std{0.3\e{1}}
        & 5.1\std{0.8\e{1}}
        & 7.0\std{2.1\e{1}}  \\
        & CANOS-PF 
        & 9.4\std{2.6\e{0}} 
        & 1.7\std{0.2\e{2}} 
        & 1.6\std{0.3\e{2}} 
        & 1.9\std{0.4\e{2}} \\
        & GNS 
        & 2.3\std{1.0\e{1}} 
        & 8.9\std{4.2\e{1}} 
        & 1.4\std{0.7\e{2}} 
        & 8.4\std{5.6\e{1}} \\ 
        & \textbf{\graphkit }
        & \textbf{7.7\std{4.8\e{-2}}}
        & \textbf{2.5\std{0.7\e{1}}}
        & \textbf{3.0\std{0.9\e{1}}}
        & \textbf{4.3\std{1.9\e{1}}} \\ \hline

        \multirow{3}{*}{1.2} 
        & PFNet 
        & 1.1\std{0.1\e{1}} 
        & 4.2\std{0.7\e{1}}  
        & 5.0\std{1.3\e{1}} 
        & 3.7\std{0.6\e{1}} \\
        & CANOS-PF 
        & 3.9\std{1.1\e{0}} 
        & 1.2\std{0.06\e{1}} 
        & 1.5\std{0.1\e{1}} 
        & 2.7\std{0.4\e{1}} \\
        & GNS 
        & 1.6\std{0.2\e{1}} 
        & 2.1\std{0.08\e{1}} 
        & 1.7\std{0.2\e{1}} 
        & \textbf{2.4\std{0.8\e{1}}} \\ 
        & \textbf{\graphkit }
        & \textbf{6.8\std{6.3\e{-2}}}
        & \textbf{1.8\std{1.9\e{0}}}
        & \textbf{3.2\std{3.9\e{0}}}
        & 3.0\std{1.10\e{1}} \\ \hline

        \multirow{3}{*}{1.3, 2.1} 
        & PFNet 
        & 1.0\std{0.2\e{1}} 
        & 3.5\std{0.3\e{1}} 
        & 3.5\std{0.4\e{1}} 
        & 3.5\std{0.6\e{1}} \\
        & CANOS-PF 
        & 3.7\std{0.1\e{0}} 
        & 1.1\std{0.1\e{1}} 
        & 8.9\std{2.1\e{0}} 
        & 3.4\std{0.4\e{1}} \\
        & GNS 
        & 1.1\std{0.3\e{1}} 
        & 2.1\std{0.3\e{1}} 
        & 1.8\std{0.2\e{1}} 
        & \textbf{1.9\std{0.5\e{1}}} \\ 
        & \graphkit 
        & \textbf{6.6\std{4.01\e{-2}}}
        & \textbf{1.2\std{0.6\e{0}}}
        & \textbf{2.0\std{1.8\e{0}}}
        & 3.9\std{2.6\e{1}} \\ \hline

        \multirow{3}{*}{2.3} 
        & PFNet 
        & 1.1\std{0.2\e{1}} 
        & 4.8\std{1.8\e{1}} 
        & 3.9\std{0.2\e{1}} 
        & 4.3\std{0.7\e{1}} \\
        & CANOS-PF 
        & 0.8\std{0.4\e{1}} 
        & 3.4\std{0.8\e{1}} 
        & 2.9\std{0.2\e{1}} 
        & 4.3\std{0.8\e{1}} \\
        & GNS 
        & 3.7\std{3.5\e{1}} 
        & 5.4\std{5.1\e{1}} 
        & 5.3\std{3.8\e{1}} 
        & 4.0\std{2.3\e{1}} \\
        & \graphkit 
        & \textbf{1.1\std{0.3\e{-1}} }
        & \textbf{1.7\std{0.5\e{0}}} 
        & \textbf{2.2\std{0.3\e{0}} }
        & \textbf{3.2\std{0.4\e{1}} }
        \\ \hline
        
        \multirow{3}{*}{4.1} 
        & PFNet 
        & 1.5\std{0.2\e{1}} 
        & 4.5\std{0.2\e{1}} 
        & 4.3\std{0.3\e{1}} 
        & 8.5\std{0.8\e{1}} \\
        & CANOS-PF 
        & 1.1\std{0.08\e{1}} 
        & 3.2\std{0.2\e{1}} 
        & 3.1\std{0.2\e{1}} 
        & 9.1\std{0.8\e{1}} \\
        & GNS 
        & 1.2\std{0.2\e{1}} 
        & 2.1\std{0.3\e{1}} 
        & 1.5\std{0.07\e{1}} 
        & 2.0\std{0.3\e{1}} \\ 
        & \graphkit 
        & \textbf{1.6\std{2.2\e{0}} }
        & \textbf{5.6\std{3.2\e{0}} }
        & \textbf{3.1\std{1.7\e{0}} }
        & 3.1\std{1.8\e{1}} 
        \\ \hline

        \multirow{3}{*}{4.2} 
        & PFNet 
        & 1.6\std{0.3\e{1}} 
        & 4.9\std{0.8\e{1}} 
        & 4.1\std{1.2\e{1}} 
        & 8.5\std{0.4\e{1}} \\
        & CANOS-PF 
        & 8.5\std{1.5\e{0}} 
        & 2.8\std{0.06\e{1}} 
        & 2.7\std{0.08\e{1}} 
        & 4.6\std{0.6\e{1}} \\
        & GNS 
        & 1.8\std{0.6\e{1}} 
        & 2.4\std{0.3\e{1}} 
        & 2.5\std{1.4\e{1}} 
        & 2.2\std{0.2\e{1}} \\ 
        & \graphkit 
        & \textbf{1.1\std{1.2\e{0}} }
        & \textbf{5.9\std{5.02\e{0}}} 
        & \textbf{4.2\std{2.1\e{0}} }
        & 2.8\std{0.8\e{1}} 
        \\ \hline

        \multirow{3}{*}{4.3} 
        & PFNet 
        & 3.3\std{0.8\e{1}} 
        & 6.0\std{2.2\e{1}} 
        & 5.0\std{0.9\e{1}} 
        & 8.1\std{0.3\e{1}} \\
        & CANOS-PF 
        & 2.0\std{0.5\e{1}} 
        & 4.0\std{0.3\e{1}} 
        & 5.2\std{0.3\e{1}} 
        & 2.4\std{0.3\e{1}} \\
        & GNS 
        & 2.4\std{1.9\e{1}} 
        & 4.5\std{3.6\e{1}} 
        & 4.1\std{2.3\e{1}} 
        & 2.5\std{0.5\e{1}} \\
        & \graphkit 
        & \textbf{4.1\std{0.8\e{0}} }
        & \textbf{2.7\std{2.9\e{1}} }
        & \textbf{1.5\std{1.09\e{1}} }
        & \textbf{1.5\std{0.2\e{1}} }
        \\ \hline
    \end{tabular}
    \caption{Power Balance Loss (Max) across different grid conditions.}
    \label{table:results-full2}
\end{table}"""


latex_table_mean = r"""\begin{table}[h!]
    \centering
    \small
    \renewcommand{\arraystretch}{1.2}
    \begin{tabular}{c|c|c|c|c|c} \hline 
        
        \multicolumn{2}{c|}{\textbf{Experiment}} & \multicolumn{4}{|c}{\textbf{Power Balance Loss (Mean)}}\\ \hline 
        
        \textbf{Task} & 
        \textbf{Model} & \textbf{N} & \textbf{N-1} 
        & \textbf{N-2} & \textbf{Close-to- infeasible} \\ \hline

        \multirow{3}{*}{1.1} 
        & PFNet 
        & 3.2\std{0.2\e{-1}} 
        & 5.2\std{0.2\e{-1}}
        & 5.3\std{0.3\e{-1}}
        & 1.4\std{0.1\e{0}} \\
        & CANOS-PF 
        & 1.9\std{0.3\e{-1}} 
        & 7.6\std{0.8\e{-1}} 
        & 6.6\std{0.5\e{-1}} 
        & 1.2\std{0.1\e{0}} \\
        & GNS 
        & 5.3\std{2.7\e{-1}} 
        & 6.8\std{2.6\e{-1}} 
        & 7.1\std{2.5\e{-1}} 
        & 1.0\std{0.2\e{0}} \\
        & \textbf{\graphkit}
        & \textbf{1.4\std{0.4\e{-3}}}
        & \textbf{4.2\std{2.2\e{-2}}}
        & \textbf{5.0\std{2.6\e{-2}}}
        & \textbf{3.0\std{0.8\e{-1}}} \\ \hline %without top perturbations in data, results worsen on top perturbations

        \multirow{3}{*}{1.2} 
        & PFNet 
        & 3.2\std{0.2\e{-1}} 
        & 4.0\std{0.2\e{-1}} 
        & 4.4\std{0.2\e{-1}} 
        & 1.2\std{0.05\e{0}} \\
        & CANOS-PF 
        & 1.8\std{0.06\e{-1}} 
        & 2.2\std{0.08\e{-1}} 
        & 2.4\std{0.09\e{-1}} 
        & 8.7\std{0.7\e{-1}} \\
        & GNS 
        & 3.6\std{0.3\e{-1}} 
        & 4.0\std{0.3\e{-1}} 
        & 4.1\std{0.3\e{-1}} 
        & 8.2\std{1.1\e{-1}} \\
        & \textbf{\graphkit}
        & \textbf{1.7\std{1.3\e{-3}}}
        & \textbf{3.7\std{2.5\e{-3}}}
        & \textbf{5.5\std{3.7\e{-3}}}
        & \textbf{1.8\std{0.4\e{-1}}}\\ \hline %N-1 is enough to have good perf even on n-2

        \multirow{3}{*}{1.3, 2.1}
        & PFNet 
        & 3.4\std{0.08\e{-1}} 
        & 4.0\std{0.05\e{-1}} 
        & 4.2\std{0.09\e{-1}} 
        & 1.1\std{0.03\e{0}} \\
        & CANOS-PF 
        & 1.9\std{0.2\e{-1}} 
        & 2.1\std{0.1\e{-1}} 
        & 2.2\std{0.1\e{-1}} 
        & 9.7\std{0.7\e{-1}} \\
        & GNS 
        & 3.5\std{0.4\e{-1}} 
        & 3.8\std{0.4\e{-1}} 
        & 3.9\std{0.4\e{-1}} 
        & 7.7\std{0.5\e{-1}} \\ 
        & \textbf{\graphkit }
        & \textbf{2.7\std{2.2\e{-3}} }
        & \textbf{4.6\std{3.4\e{-3}} }
        & \textbf{5.9\std{4.3\e{-3}} }
        & \textbf{2.2\std{0.8\e{-1}} }
        \\ \hline %training on everything impacts loss. training on n-1 gives better results on n-2 then training on n-1 and n-2 !!

        \multirow{3}{*}{2.3} 
        & PFNet 
        & 4.1\std{0.6\e{-1}} 
        & 4.9\std{0.7\e{-1}} 
        & 5.1\std{0.7\e{-1}} 
        & 1.1\std{0.08\e{0}} \\
        & CANOS-PF 
        & 4.1\std{0.8\e{-1}} 
        & 4.6\std{0.9\e{-1}} 
        & 4.7\std{0.8\e{-1}} 
        & 1.0\std{0.1\e{0}} \\
        & GNS 
        & 8.3\std{2.6\e{-1}} 
        & 8.8\std{2.7\e{-1}} 
        & 8.9\std{2.7\e{-1}} 
        & 1.3\std{0.3\e{0}} \\
        & \graphkit 
        & \textbf{5.0\std{2.5\e{-3}} }
        & \textbf{8.1\std{3.7\e{-3}} }
        & \textbf{9.9\std{4.2\e{-3}} }
        & \textbf{2.5\std{0.4\e{-1}} }
        \\ \hline

        \multirow{3}{*}{4.1} 
        & PFNet 
        & 4.5\std{0.4\e{-1}} 
        & 5.2\std{0.09\e{-1}} 
        & 5.3\std{0.07\e{-1}} 
        & 1.3\std{0.04\e{0}} \\
        & CANOS-PF 
        & 4.7\std{0.1\e{-1}} 
        & 5.3\std{0.05\e{-1}} 
        & 5.4\std{0.06\e{-1}} 
        & 1.7\std{0.01\e{0}} \\
        & GNS 
        & 3.5\std{0.1\e{-1}} 
        & 3.7\std{0.2\e{-1}} 
        & 3.8\std{0.2\e{-1}} 
        & 6.6\std{0.3\e{-1}} \\ 
        & \graphkit 
        & \textbf{2.7\std{1.8\e{-2}} }
        & \textbf{3.6\std{2.4\e{-2}}} 
        & \textbf{4.2\std{2.7\e{-2}} }
        & \textbf{2.1\std{1.2\e{-1}} }
        \\ \hline

        \multirow{3}{*}{4.2}
        & PFNet 
        & 5.6\std{0.7\e{-1}} 
        & 6.6\std{0.7\e{-1}} 
        & 6.7\std{0.7\e{-1}} 
        & 1.2\std{0.1\e{0}} \\
        & CANOS-PF 
        & 3.9\std{0.8\e{-1}} 
        & 4.3\std{0.8\e{-1}} 
        & 4.4\std{0.8\e{-1}} 
        & 8.8\std{0.9\e{-1}} \\
        & GNS 
        & 4.0\std{0.9\e{-1}} 
        & 4.6\std{1.0\e{-1}} 
        & 4.6\std{1.0\e{-1}} 
        & 6.4\std{1.4\e{-1}} \\ 
        & \graphkit 
        & \textbf{4.4\std{3.8\e{-2}} }
        & \textbf{6.0\std{5.0\e{-2}} }
        & \textbf{7.0\std{5.7\e{-2}} }
        & \textbf{1.8\std{1.2\e{-1}}} 
        \\ \hline

        \multirow{3}{*}{4.3} 
        & PFNet 
        & 1.2\std{0.04\e{0}} 
        & 1.4\std{0.04\e{0}} 
        & 1.5\std{0.02\e{0}} 
        & 1.1\std{0.07\e{0}} \\
        & CANOS-PF 
        & 1.1\std{0.04\e{0}} 
        & 1.2\std{0.03\e{0}} 
        & 1.2\std{0.02\e{0}} 
        & 0.8\std{0.05\e{0}} \\
        & GNS 
        & 6.3\std{2.2\e{-1}} 
        & 7.3\std{2.4\e{-1}} 
        & 7.3\std{2.4\e{-1}} 
        & 7.0\std{2.8\e{-1}} \\ 
        & \graphkit 
        & \textbf{1.5\std{0.5\e{-1}} }
        & \textbf{2.0\std{0.6\e{-1}} }
        & \textbf{2.2\std{0.6\e{-1}} }
        & \textbf{1.3\std{0.5\e{-1}}} 
        \\ \hline
    \end{tabular}
    \caption{Power Balance Loss (Mean) across different grid conditions.}
    \label{table:results-full1}
\end{table}"""


latex_table = latex_table_mean if result_type == "mean" else latex_table_max
# %%
import re
import pandas as pd

# Strip \textbf wrapper
def strip_textbf(s):
    return re.sub(r"\\textbf{([^}]*)}", r"\1", s).strip()

# Patterns
task_pattern = re.compile(r"\\multirow{[^}]+}{\*}{(?P<Task>[^}]+)}")
# Each cell looks like: coeff \std{ std_val \e{ exp } }
# Build a named-group fragment for one cell: captures coeff, std, and exp
def _cell(name: str) -> str:
    return (
        rf"&\s*(?P<{name}_coeff>[\d.]+)"
        rf"\\std\{{(?P<{name}_std>[\d.]+)"
        rf"\\e\{{(?P<{name}_exp>-?\d+)\s*\}}\s*\}}\s*"
    )

row_pattern = re.compile(
    r"&\s*(?P<Model>[^&]+?)\s*"
    + _cell("N") + _cell("N1") + _cell("N2") + _cell("Close")
)


def iter_latex_rows(table: str) -> list[str]:
    """
    Coalesce a LaTeX table body into logical rows.

    The source table is often wrapped across multiple lines; we treat `\\` as the
    end-of-row marker and concatenate lines until then.
    """
    rows: list[str] = []
    buf = ""

    for raw in table.splitlines():
        line = strip_textbf(raw.strip())
        if not line:
            continue

        # Drop LaTeX comments (safe for this specific input table)
        if "%" in line:
            line = line.split("%", 1)[0].rstrip()
            if not line:
                continue

        buf = f"{buf} {line}".strip()

        # End-of-row in LaTeX tabular
        if r"\\" in line:
            rows.append(buf)
            buf = ""

    if buf:
        rows.append(buf)

    return rows

COL_NAMES = ["N", "N1", "N2", "Close"]

rows_data: list[dict] = []
current_task = None

for row in iter_latex_rows(latex_table):
    task_match = task_pattern.search(row)
    if task_match:
        current_task = task_match.group("Task").strip()

    row_match = row_pattern.search(row)
    if row_match and current_task is not None:
        model = row_match.group("Model").strip().lstrip("\\").strip()
        entry = {"Task": current_task, "Model": model}

        for col in COL_NAMES:
            coeff = float(row_match.group(f"{col}_coeff"))
            std   = float(row_match.group(f"{col}_std"))
            exp   = int(row_match.group(f"{col}_exp"))
            entry[col]          = coeff * 10 ** exp
            entry[f"{col}_std"] = std * 10 ** exp

        rows_data.append(entry)

# Build DataFrame
df = pd.DataFrame(rows_data)

# Reorder columns: Task, Model, then value/std pairs
ordered = ["Task", "Model"]
for col in COL_NAMES:
    ordered += [col, f"{col}_std"]
df = df[ordered]

print(df.to_string())

# %%
# Barplot (matches screenshot) using parsed means/stds (no JSON needed).
# Produces: figures/experimental_results_from_parsed.(png|svg)

from pathlib import Path
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np


def plot_barplot_from_parsed(
    df_in: pd.DataFrame,
    out_base: str = "figures/experimental_results_from_parsed",
    include_models: Optional[List[str]] = None,
) -> None:
    # Task layout like the screenshot
    tasks_top = ["1.1", "1.2", "1.3, 2.1", "2.3"]
    tasks_bot = ["4.1", "4.2", "4.3", None]  # pad to 4 columns
    task_rows = [tasks_top, tasks_bot]

    # Map perturbations (columns) -> x labels
    attrs = [
        ("N", "N_std", "N"),
        ("N1", "N1_std", "N-1"),
        ("N2", "N2_std", "N-2"),
        ("Close", "Close_std", "C2I"),
    ]

    model_display = {
        "CANOS-PF": "CANOS-PF",
        "GNS": "GNS-S",
        "PFNet": "PFNet",
        "graphkit": "graphkit",
    }

    palette = {
        "CANOS-PF": "#D9A6D9",
        "GNS": "#F27AA4",
        "PFNet": "#69D8FF",
        "graphkit": "#d62728",
    }

    if include_models is None:
        include_models = ["CANOS-PF", "GNS", "PFNet", "graphkit"]

    fontsize = 18
    nrows, ncols = 2, 4
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 5 * nrows),
        sharey=(result_type == "mean"),
    )
    axes = np.array(axes).reshape(nrows, ncols)

    x = np.arange(len(attrs))
    width = 0.18 if len(include_models) >= 4 else 0.22
    offsets = (np.arange(len(include_models)) - (len(include_models) - 1) / 2) * width

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            task = task_rows[r][c]
            if task is None:
                ax.set_visible(False)
                continue

            g = df_in[df_in["Task"] == task]
            if g.empty:
                ax.set_visible(False)
                continue

            for mi, model in enumerate(include_models):
                row = g[g["Model"] == model]
                if row.empty:
                    continue

                means = [float(row[col].values[0]) for (col, _std, _lbl) in attrs]
                errs = [float(row[std].values[0]) for (_col, std, _lbl) in attrs]

                ax.bar(
                    x + offsets[mi],
                    means,
                    width=width,
                    yerr=errs,
                    capsize=3,
                    color=palette.get(model, "#999999"),
                    edgecolor="white",
                    linewidth=0.8,
                    error_kw={"elinewidth": 1.6, "capthick": 1.6},
                    label=model_display.get(model, model),
                )

            ax.set_xticks(x)
            ax.set_xticklabels([lbl for *_rest, lbl in attrs], fontsize=fontsize)
            ax.tick_params(axis="y", labelsize=fontsize)
            ax.set_xlabel("")
            ax.yaxis.grid(True, alpha=0.35)
            if result_type == "mean":
                ax.set_yticks([0, 0.45, 0.9, 1.35, 1.8])
                ax.set_ylim(0, 1.8)
            else:
                # Requested y-ranges for Max plots:
                # - Task 1.1: (0, 250)
                # - All other tasks: (0, 120)
                if task == "1.1":
                    ax.set_ylim(0, 250)
                    ax.set_yticks([0, 50, 100, 150, 200, 250])
                else:
                    ax.set_ylim(0, 120)
                    ax.set_yticks([0, 30, 60, 90, 120])

            if c == 0:
                ax.set_ylabel(
                    f"Power Balance Loss ({result_type.capitalize()}, p.u.)",
                    fontsize=fontsize - 2,
                    fontweight="bold",
                )
                ax.legend(
                    title="Model",
                    loc="upper left",
                    fontsize=fontsize - 6,
                    title_fontsize=fontsize - 6,
                    frameon=True,
                )
            else:
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()

            title_text = "Task 1.3 / 2.1" if task == "1.3, 2.1" else f"Task {task}"
            ax.set_title(title_text, fontsize=fontsize, fontweight="bold")

    # For max plots the y tick labels (esp. up to 250) need more gutter space.
    if result_type == "max":
        plt.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.08, wspace=0.22, hspace=0.38)
    else:
        plt.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.08, wspace=0.05, hspace=0.35)

    out_png = Path(f"{out_base}_{result_type}.png")
    out_svg = Path(f"{out_base}_{result_type}.svg")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved to {out_png}")
    print(f"Saved to {out_svg}")


plot_barplot_from_parsed(df)

# %%
# Spider (radar) plot: one subplot per task, one line per model,
# axes = perturbation levels (N, N-1, N-2, Close-to-infeasible).

import matplotlib.pyplot as plt
import numpy as np

PERTURBATIONS = ["N", "N1", "N2", "Close"]
AXIS_LABELS = ["N", "N-1", "N-2", "Close-to-\ninfeasible"]
MODELS = ["PFNet", "CANOS-PF", "GNS", "graphkit"]
MODEL_COLORS = {
    "PFNet": "#1f77b4",
    "CANOS-PF": "#ff7f0e",
    "GNS": "#2ca02c",
    "graphkit": "#d62728",
}
MODEL_STYLES = {
    "PFNet": "--",
    "CANOS-PF": "--",
    "GNS": "--",
    "graphkit": "-",
}

tasks = df["Task"].unique()
n_tasks = len(tasks)
n_axes = len(PERTURBATIONS)
angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
angles += angles[:1]  # close the polygon

ncols = 4
nrows = (n_tasks + ncols - 1) // ncols
fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(5 * ncols, 5 * nrows),
    subplot_kw=dict(polar=True),
)
axes_flat = np.array(axes).flatten()

for idx, task in enumerate(tasks):
    ax = axes_flat[idx]
    group = df[df["Task"] == task]

    for model in MODELS:
        model_row = group[group["Model"] == model]
        if model_row.empty:
            continue
        vals = [model_row[col].values[0] for col in PERTURBATIONS]
        vals += vals[:1]  # close polygon

        ax.plot(
            angles, vals,
            linestyle=MODEL_STYLES[model],
            linewidth=2.5 if model == "graphkit" else 1.5,
            label=model,
            color=MODEL_COLORS[model],
        )
        ax.fill(angles, vals, alpha=0.05, color=MODEL_COLORS[model])

        # Annotate graphkit values as text near each axis
        if model == "graphkit":
            for j, col in enumerate(PERTURBATIONS):
                v = vals[j]
                # Format: scientific-ish for tiny values, normal otherwise
                if v < 0.01:
                    txt = f"{v:.1e}"
                elif v < 0.1:
                    txt = f"{v:.3f}"
                else:
                    txt = f"{v:.2f}"
                # Place label at ~15% of radial range so it's readable
                r_max = ax.get_ylim()[1]
                r_pos = r_max * 0.18
                ax.annotate(
                    txt,
                    xy=(angles[j], r_pos),
                    fontsize=7,
                    fontweight="bold",
                    color=MODEL_COLORS["graphkit"],
                    ha="center", va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        fc="white", ec=MODEL_COLORS["graphkit"],
                        alpha=0.85, lw=0.5,
                    ),
                )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(AXIS_LABELS, size=9)
    ax.set_title(f"Task {task}", size=13, fontweight="bold", pad=18)

    # Show radial tick labels only every other tick
    ticks = ax.get_yticks()
    ax.set_yticks(ticks)
    labels = [f"{t:.1f}" if (i+1) % 2 == 0 else "" for i, t in enumerate(ticks)]
    ax.set_yticklabels(labels, size=7)

# Hide unused subplots
for idx in range(n_tasks, len(axes_flat)):
    axes_flat[idx].set_visible(False)

# Single shared legend
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    ncol=len(MODELS),
    fontsize=11,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
)

fig.suptitle(
    f"Power Balance ({result_type.capitalize()})",
    fontsize=16, fontweight="bold", y=1.02,
)
fig.tight_layout()
fig.savefig(f"scripts/spider_plots_{result_type}.svg", bbox_inches="tight")
print(f"\nSaved to scripts/spider_plots_{result_type}.svg")

