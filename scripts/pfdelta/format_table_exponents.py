"""Convert PFΔ genco tables to OPF-style number formatting.

Source cells look like:  value\\std{std\\e{exp}}
OPF-style cells look like:  valuee-exp\\std{stde-exp}
with \\G{value} for the best (lowest) neural model per column, and
\\textcolor{gray}{...} for NR rows.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent

INPUT_TABLES = [
    ROOT / "table_a_8_genco.tex",
    ROOT / "table_a_9_genco.tex",
    ROOT / "table_a_10_genco.tex",
    ROOT / "table_a_11_genco.tex",
]

METRIC_PATTERN = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\\std\{(?P<std>\d+(?:\.\d+)?)\\e\{(?P<exp>-?\d+)\}\}"
)
MODEL_RE = re.compile(r"^\s*&\s*(GENCO Base|PFNet|CANOS-PF|GNS|NR)\s*$")


def sci_token(mantissa: str, exp: int) -> str:
    if exp == 0:
        return mantissa
    return f"{mantissa}e{exp}"


def format_metric(value: str, std: str, exp: int, *, best: bool = False, gray: bool = False) -> str:
    mean = sci_token(value, exp)
    err = sci_token(std, exp)
    if best:
        cell = rf"\G{{{mean}}}\std{{{err}}}"
    else:
        cell = rf"{mean}\std{{{err}}}"
    if gray:
        cell = rf"\textcolor{{gray}}{{{cell}}}"
    return cell


def parse_metric(raw: str) -> tuple[str, str, int, float]:
    m = METRIC_PATTERN.search(raw)
    if not m:
        raise ValueError(f"Cannot parse metric: {raw!r}")
    value, std, exp = m.group("value"), m.group("std"), int(m.group("exp"))
    return value, std, exp, float(value) * (10 ** exp)


def convert_table_text(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0

    while i < len(lines):
        if r"\multirow" not in lines[i]:
            out.append(lines[i])
            i += 1
            continue

        out.append(lines[i])
        i += 1
        models: list[dict] = []
        current = None

        while i < len(lines) and r"\multirow" not in lines[i]:
            line = lines[i]
            name_m = MODEL_RE.match(line.rstrip("\n"))
            if name_m:
                if current is not None:
                    models.append(current)
                current = {
                    "name": name_m.group(1),
                    "name_line": line,
                    "cells": [],  # list of (raw_line, value, std, exp, numeric)
                }
                i += 1
                while i < len(lines) and len(current["cells"]) < 4:
                    if MODEL_RE.match(lines[i].rstrip("\n")) or r"\multirow" in lines[i]:
                        break
                    cell_line = lines[i]
                    if METRIC_PATTERN.search(cell_line):
                        value, std, exp, numeric = parse_metric(cell_line)
                        current["cells"].append((cell_line, value, std, exp, numeric))
                        i += 1
                    else:
                        break
                continue

            if current is not None:
                models.append(current)
                current = None

            if models:
                _emit_group(out, models)
                models = []
                out.append(line)
                i += 1
                break

            out.append(line)
            i += 1
        else:
            if current is not None:
                models.append(current)
            if models:
                _emit_group(out, models)

    return "".join(out)


def _emit_group(out: list[str], models: list[dict]) -> None:
    n_cols = max(len(m["cells"]) for m in models)
    best = []
    for c in range(n_cols):
        cands = [
            (m["cells"][c][4], mi)
            for mi, m in enumerate(models)
            if m["name"] != "NR" and c < len(m["cells"])
        ]
        best.append(min(cands, key=lambda x: x[0])[1] if cands else None)

    for mi, m in enumerate(models):
        is_nr = m["name"] == "NR"
        name = m["name"]
        indent = re.match(r"^(\s*)&", m["name_line"]).group(1)
        if is_nr:
            out.append(f"{indent}& \\textcolor{{gray}}{{{name}}}\n")
        else:
            out.append(f"{indent}& {name}\n")

        for c, (cell_line, value, std, exp, _num) in enumerate(m["cells"]):
            indent = re.match(r"^(\s*)&", cell_line).group(1)
            suffix_m = re.search(r"(\s*\\\\\s*(?:\\hline)?\s*)$", cell_line.rstrip("\n"))
            suffix = suffix_m.group(1) if suffix_m else ""
            if not suffix.endswith("\n"):
                suffix = (suffix + "\n") if cell_line.endswith("\n") else suffix
            elif not cell_line.endswith("\n") and suffix.endswith("\n"):
                pass
            if cell_line.endswith("\n") and not suffix.endswith("\n"):
                suffix += "\n"

            cell = format_metric(
                value,
                std,
                exp,
                best=(not is_nr and best[c] == mi),
                gray=is_nr,
            )
            # Ensure a space before \\
            if suffix.lstrip().startswith("\\") and not suffix.startswith(" "):
                suffix = " " + suffix.lstrip()
            if not (suffix.startswith(" ") or suffix.startswith("\n")):
                suffix = " " + suffix
            out.append(f"{indent}& {cell}{suffix}")


def output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_exponent_fixed{input_path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert table metrics from value\\std{std\\e{exp}} to OPF-style "
            "valuee-exp\\std{stde-exp}, with \\G{...} for best and gray NR."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="input .tex files (defaults to table_a_8/9/10/11_genco.tex)",
    )
    args = parser.parse_args()

    input_paths = args.inputs or INPUT_TABLES
    for input_path in input_paths:
        resolved = input_path if input_path.is_absolute() else ROOT / input_path
        if not resolved.exists():
            raise FileNotFoundError(f"Missing table file: {resolved}")

        converted = convert_table_text(resolved.read_text())
        destination = output_path(resolved)
        destination.write_text(converted)
        print(f"updated {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
