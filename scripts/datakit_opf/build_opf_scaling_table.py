from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "results" / "sheet2_from_experiments_mean_std_only.csv"
OUTPUT_TEX = ROOT / "opf_scaling_table.tex"

SYSTEM_ORDER = ["14-ieee", "30-ieee", "57-ieee", "118-ieee", "500-goc", "2000-goc"]
MODEL_ROWS = [
    ("HGNS_OPF_datakit_case{case}_base_mean_std", "GENCO base"),
    ("HGNS_OPF_datakit_case{case}_small_mean_std", "GENCO small"),
    (("DC-OPF_mean_std", "DC-OPF"), "DC-OPF"),
]
METRIC_COLUMNS = [
    "opt gap",
    "theta_ij",
    "S_ij(+)",
    "S_ij(-)",
    "Pb",
    "Qb",
    "Qg_violation",
]


def case_number(system):
    return system.split("-", 1)[0]


def format_system(system):
    prefix, suffix = system.split("-", 1)
    return f"{prefix}-{suffix.upper()}"


def parse_pm(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    text = str(value).strip()
    if "±" not in text:
        return float(text), 0.0
    mean_text, std_text = text.split("±", 1)
    return float(mean_text.strip()), float(std_text.strip())


def fmt_scientific(value):
    if abs(value) < 1e-15:
        return "0"
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa}e{int(exponent):+d}"


def fmt_value(parsed, decimal=False, bold=False):
    if parsed is None:
        return "--"
    mean, std = parsed
    if decimal:
        text = f"{mean:.2f}" if std == 0 else f"{mean:.2f} $\\pm$ {std:.2f}"
    else:
        text = (
            fmt_scientific(mean)
            if std == 0
            else f"{fmt_scientific(mean)} $\\pm$ {fmt_scientific(std)}"
        )
    return f"\\textbf{{{text}}}" if bold else text


def best_mean_flags(parsed_values):
    flags = [False] * len(parsed_values)

    available = [
        (idx, value) for idx, value in enumerate(parsed_values)
        if value is not None
    ]

    if len(available) < 2:
        return flags

    means = [v[0] for _, v in available]
    best_mean = min(means)

    for idx, (mean, _) in available:
        if mean == best_mean:
            flags[idx] = True

    return flags


def row_for_model(df, system, model_template):
    templates = model_template if isinstance(model_template, tuple) else (model_template,)
    for template in templates:
        model = template.format(case=case_number(system))
        matches = df[(df["System"] == system) & (df["Model"] == model)]
        if not matches.empty:
            return matches.iloc[0]
    return None


def build_table_rows(df):
    rows = []
    for system in SYSTEM_ORDER:
        model_data = []
        for model_template, label in MODEL_ROWS:
            row = row_for_model(df, system, model_template)
            model_data.append(
                {
                    "label": label,
                    "values": (
                        [None for _ in METRIC_COLUMNS]
                        if row is None
                        else [parse_pm(row[column]) for column in METRIC_COLUMNS]
                    ),
                }
            )

        bold_by_metric = [
            best_mean_flags([model["values"][idx] for model in model_data])
            for idx in range(len(METRIC_COLUMNS))
        ]

        for model_idx, model in enumerate(model_data):
            system_cell = format_system(system) if model_idx == 0 else ""
            prefix = f"{system_cell} \n& {model['label']} "
            cells = []
            for metric_idx, parsed in enumerate(model["values"]):
                cells.append(
                    fmt_value(
                        parsed,
                        decimal=METRIC_COLUMNS[metric_idx] in {"opt gap", "theta_ij"},
                        bold=bold_by_metric[metric_idx][model_idx],
                    )
                )
            rows.append(prefix + "\n& " + " \n& ".join(cells) + r" \\")
        rows.append(r"\hline")
        rows.append("")

    return "\n".join(rows)


def build_latex(df):
    return (
        r"""\begin{table*}[h]
\centering
\caption{Constraint violations and optimality gaps for GENCO and DC-OPF on datakit. Bold entries mark the lowest mean in each metric. The numbers come from results/sheet2_from_experiments_mean_std_only.csv.}
\label{tab:opf_scaling}
\resizebox{\linewidth}{!}{
\begin{tabular}{|l|l|l|l|l|l|l|l|l|}
\hline
System & Model & Opt. Gap (\%) & $\theta_{ij}$ [rad] & $S_{ij}(+)$ [MVA] & $S_{ij}(-)$ [MVA] & $P_b$ [MW] & $Q_b$ [MVar] & $Q_g$ [MVar] \\
\hline

"""
        + build_table_rows(df)
        + r"""
\end{tabular}
}

\end{table*}
"""
    )


def main():
    df = pd.read_csv(INPUT_CSV)
    latex = build_latex(df)
    OUTPUT_TEX.write_text(latex, encoding="utf-8")
    print(latex)


if __name__ == "__main__":
    main()