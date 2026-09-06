import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import seaborn as sns


FONT_RC = {
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
plt.rcParams.update(FONT_RC)
sns.set_theme(style="ticks", rc=FONT_RC)


def times_font(size: float) -> font_manager.FontProperties:
    return font_manager.FontProperties(family="Times New Roman", size=size)

ROOT = Path(__file__).resolve().parent

METRIC_CONFIGS = {
    "mean": {
        "table": ROOT / "table_a_8_genco_exponent_fixed.tex",
        "output": ROOT / "figures" / "pbl_all_rows_mean.pdf",
        "ylabel": "Power Balance Loss (Mean) [p.u.]",
        "ylim": (0, 1.8),
        "yticks": [0, 0.45, 0.9, 1.35, 1.8],
    },
    "max": {
        "table": ROOT / "table_a_9_genco_exponent_fixed.tex",
        "output": ROOT / "figures" / "pbl_all_rows_max.pdf",
        "ylabel": "Power Balance Loss (Max) [p.u.]",
        "ylim": None,
        "yticks": None,
    },
}

TASKS = [1.1, 1.2, 1.3, 2.3, 4.1, 4.2, 4.3]
GRID_ATTRIBUTES = ["N", "N-1", "N-2", "C2I"]
MODELS = ["GENCO Base", "PFNet", "CANOS-PF", "GNS"]
PLOT_NAMES = {"GNS": "GNS-S"}
TASK_LABELS = {"1.1": 1.1, "1.2": 1.2, "1.3, 2.1": 1.3, "2.3": 2.3, "4.1": 4.1, "4.2": 4.2, "4.3": 4.3}
PALETTE = {
    "CANOS-PF": "#D9A6D9",
    "GNS-S": "#F27AA4",
    "PFNet": "#8FD694",
    "GENCO Base": "#69D8FF",
}


def _parse_sci(token: str) -> float:
    match = re.fullmatch(r"([\d.]+)(?:e([+-]?\d+))?", token)
    if not match:
        raise ValueError(f"Cannot parse scientific token: {token!r}")
    value = float(match.group(1))
    exp = int(match.group(2)) if match.group(2) is not None else 0
    return value * (10 ** exp)


def parse_metric(cell: str) -> tuple[float, float]:
    cell = re.sub(r"\\G\{([^{}]+)\}", r"\1", cell)
    cell = re.sub(r"\\textbf\{([^{}]+)\}", r"\1", cell)
    cell = re.sub(r"\\textcolor\{gray\}\{([^{}]+)\}", r"\1", cell)

    # OPF-style: 1.4e-3\std{0.4e-3}
    match = re.search(
        r"([\d.]+(?:e[+-]?\d+)?)\\std\{([\d.]+(?:e[+-]?\d+)?)\}",
        cell,
    )
    if match:
        return _parse_sci(match.group(1)), _parse_sci(match.group(2))

    # Legacy fixed forms
    match = re.search(
        r"\(([\d.]+)\s*\$\\pm\$\s*([\d.]+)\)(?:e\{(-?\d+)\}|\s*\$\\times\s*10\^\{(-?\d+)\}\$)"
        r"|([\d.]+)\s*\$\\pm\$\s*([\d.]+)",
        cell,
    )
    if match is None:
        raise ValueError(f"Cannot parse metric cell: {cell!r}")
    if match.group(1):
        exponent = match.group(3) if match.group(3) is not None else match.group(4)
        value, std, exponent = float(match.group(1)), float(match.group(2)), int(exponent)
    else:
        value, std, exponent = float(match.group(5)), float(match.group(6)), 0
    scale = 10 ** exponent
    return value * scale, std * scale


def parse_table(path: Path) -> dict[float, dict[str, dict[str, tuple[float, float]]]]:
    parsed: dict[float, dict[str, dict[str, tuple[float, float]]]] = {}
    task = None
    model = None
    metric_idx = 0

    for line in path.read_text().splitlines():
        if "\\multirow{4}{*}{" in line:
            label = line.rsplit("{", 1)[1].rstrip("} ").strip()
            task = TASK_LABELS[label]
            model = None
            metric_idx = 0
            parsed[task] = {}
            continue

        stripped = line.strip()
        if stripped in {f"& {name}" for name in MODELS}:
            model = stripped[2:].strip()
            metric_idx = 0
            parsed[task][model] = {}
            continue

        if model and ("\\std{" in line or "\\pm" in line):
            mean, std = parse_metric(line.split("&", 1)[1].strip())
            parsed[task][model][GRID_ATTRIBUTES[metric_idx]] = (mean, std)
            metric_idx += 1
            if metric_idx == len(GRID_ATTRIBUTES):
                model = None

    return parsed


def task_df(task_stats: dict[str, dict[str, tuple[float, float]]], metric: str) -> pd.DataFrame:
    rows = []
    for model, attributes in task_stats.items():
        plot_name = PLOT_NAMES.get(model, model)
        for attribute, (mean, std) in attributes.items():
            for sample in (max(mean - std, 0.0), mean, mean + std):
                rows.append(
                    {
                        "Grid Attribute": attribute,
                        metric: sample,
                        "Model": plot_name,
                    }
                )
    return pd.DataFrame(rows)


def y_axis_scale(
    results: dict[float, dict[str, dict[str, tuple[float, float]]]],
    tasks: list[float] | None = None,
) -> tuple[tuple[float, float], list[float]]:
    task_results = results if tasks is None else {task: results[task] for task in tasks}
    ymax = max(
        mean + std
        for task in task_results.values()
        for model in task.values()
        for mean, std in model.values()
    )
    ymax *= 1.1
    magnitude = 10 ** math.floor(math.log10(ymax)) if ymax > 0 else 1
    ymax = math.ceil(ymax / magnitude * 4) / 4 * magnitude
    yticks = [i * ymax / 4 for i in range(5)]
    return (0, ymax), yticks


def axis_limits(
    metric: str,
    results: dict[float, dict[str, dict[str, tuple[float, float]]]],
    config: dict,
) -> dict[float, tuple[tuple[float, float], list[float]]]:
    if config["ylim"] is not None:
        limits = (config["ylim"], config["yticks"])
        return {task: limits for task in TASKS}

    if metric == "max":
        return {
            TASKS[0]: y_axis_scale(results, [TASKS[0]]),
            "rest": y_axis_scale(results, TASKS[1:]),
        }

    limits = y_axis_scale(results)
    return {task: limits for task in TASKS}


MAX_FIRST_PANEL_GAP = 0.05


def widen_gap_after_first_panel(axes: list, gap: float) -> None:
    axes[0].figure.canvas.draw()
    for ax in axes[1:]:
        pos = ax.get_position()
        ax.set_position([pos.x0 + gap, pos.y0, pos.width, pos.height])


def main(metric: str = "mean") -> None:
    config = METRIC_CONFIGS[metric]
    table = config["table"]
    output = config["output"]
    ylabel = config["ylabel"]
    results = parse_table(table)
    limits_by_task = axis_limits(metric, results, config)
    fontsize = 38
    legend_fontsize = 38
    panel_w, panel_h = 5, 8
    fig, axes = plt.subplots(
        1,
        len(TASKS),
        figsize=(panel_w * len(TASKS), panel_h),
        sharex=True,
        sharey=metric != "max",
    )
    axes = list(axes.flat)
    if metric == "max" and len(axes) > 2:
        for ax in axes[2:]:
            ax.sharey(axes[1])

    handles = None
    labels = None
    for ax, task in zip(axes, TASKS):
        sns.barplot(
            data=task_df(results[task], ylabel),
            x="Grid Attribute",
            order=GRID_ATTRIBUTES,
            y=ylabel,
            hue="Model",
            hue_order=["GENCO Base", "CANOS-PF", "GNS-S", "PFNet"],
            ax=ax,
            palette=PALETTE,
            errorbar="sd",
            capsize=0.1,
            edgecolor="white",
            err_kws={"linewidth": 1.6},
        )
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        ax.set_xlabel("")
        ax.set_ylabel("")
        label = "Task 1.3 / 2.1" if task == 1.3 else f"Task {task}"
        ax.text(
            0.90,
            0.95,
            label,
            transform=ax.transAxes,
            fontproperties=times_font(fontsize),
            va="top",
            ha="right",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
        ax.yaxis.grid(True)
        if metric == "max":
            ylim, yticks = limits_by_task[TASKS[0] if task == TASKS[0] else "rest"]
        else:
            ylim, yticks = limits_by_task[task]
        ax.set_yticks(yticks)
        ax.set_ylim(*ylim)
        ax.tick_params(axis="x", labelsize=fontsize, rotation=0)
        ax.tick_params(axis="y", labelsize=fontsize)
        tick_font = times_font(fontsize)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontproperties(tick_font)

    if metric == "max":
        for ax in axes[2:]:
            ax.tick_params(labelleft=False)
    else:
        for ax in axes[1:]:
            ax.tick_params(labelleft=False)

    plt.subplots_adjust(wspace=0.05, left=0.06, right=0.99, top=0.82)
    if metric == "max":
        widen_gap_after_first_panel(axes, MAX_FIRST_PANEL_GAP)

    fig.supylabel(ylabel, fontproperties=times_font(fontsize), x=0.0)

    pos = axes[0].get_position()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(pos.x0 + pos.width / 2 + 0.25, pos.y1 + 0.03),
        ncol=4,
        prop=times_font(legend_fontsize),
        frameon=False,
    )
    fig.canvas.draw()
    legend_bbox = legend.get_window_extent().transformed(fig.transFigure.inverted())
    fig.text(
        legend_bbox.x0 - 0.008,
        legend_bbox.y0 + legend_bbox.height / 2,
        "Model: ",
        ha="right",
        va="center",
        fontproperties=times_font(legend_fontsize),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot power balance loss bar charts from LaTeX tables.")
    parser.add_argument(
        "--metric",
        choices=("mean", "max"),
        default="mean",
        help="mean: table_a_8 (default); max: table_a_9",
    )
    args = parser.parse_args()
    main(args.metric)
