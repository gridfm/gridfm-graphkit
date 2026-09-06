# §5.1 reproducibility result

Date: 2026-09-06  
Code: `gridfm-graphkit` branch `pfdelta_alban`, commit `0ea0594`  
Paper artifacts: `GENCO/paper/figures/pf_delta/`

## Did the new scripts get committed?

Yes, locally on `pfdelta_alban` as **`0ea0594`** (“Replace the stale parse.py barplot with the PFΔ table and figure scripts.”). That commit is **ahead of `origin/pfdelta_alban` by 1** (not pushed).

It adds `examples/pfdelta/{make_latex_tables,format_table_exponents,make_pbl_barplots}.py` plus the generated `table_a_{8,9,10,11}_genco*.tex` files, and deletes `scripts/parse.py`.

Generated PDFs under `examples/pfdelta/figures/` were left untracked.

## What was reproduced

This check covers the **table and figure pipeline** from committed GENCO CSVs, not a from-scratch train/eval.

Not re-run: Hugging Face download, conversion, `gridfm_graphkit train` / `evaluate`, or `aggregate_mlflow_metrics.py`. GENCO means and stds are those already in `results/{1.1,1.2,1.3,2.3,3.1,4.1,4.2,4.3}/metrics_summary.csv` on `pfdelta_alban`.

## Pipeline that was run

Working directory: `/Users/apu/repro/gridfm-graphkit` (same files as `/Users/apu/to_del_just_to_evaluate_contingency/gridfm-graphkit/examples/pfdelta`, with two compatibility edits: parse OPF-style `\std` cells; read `results/` from the repo root).

```bash
python examples/pfdelta/make_latex_tables.py
python examples/pfdelta/format_table_exponents.py
python examples/pfdelta/make_pbl_barplots.py --metric mean
python examples/pfdelta/make_pbl_barplots.py --metric max
```

- `make_latex_tables.py` writes GENCO rows from `results/*/metrics_summary.csv` (`PBE (Mean|Max, p.u.) latex`) and inserts the hardcoded PFNet / CANOS-PF / GNS / NR cells.
- `format_table_exponents.py` writes `table_a_*_genco_exponent_fixed.tex` (OPF-style `valuee-exp\std{...}`, `\G{...}` on the best neural model).
- `make_pbl_barplots.py` reads `table_a_8_genco_exponent_fixed.tex` (mean) and `table_a_9_genco_exponent_fixed.tex` (max) and writes `examples/pfdelta/figures/pbl_all_rows_{mean,max}.pdf`.

## Match against the paper

**Table 4** (`GENCO/paper/figures/pf_delta/pf_delta_a_10.tex` vs `examples/pfdelta/table_a_10_genco_exponent_fixed.tex`): **every numeric cell matches exactly** (60 mean/std tokens, 0 mismatches), including CANOS-PF on IEEE 118 N / N-1 / N-2: `4.0e-2`, `5.6e-2`, `6.3e-2`. Cosmetic differences only: `GENCO Base` vs `\genco Base`, `GNS` vs `GNS-S`, table environment / `\resizebox`, `[p.u.]` in the header, and the published caption.

**GENCO rows vs CSVs:** parsed values in `table_a_8` / `table_a_9` / `table_a_10` match `results/*/metrics_summary.csv` with **no mismatches**. Task 1.3 vs Task 3.1 IEEE 118 GENCO latex differs slightly in the CSVs (e.g. N-1 mean `4.6e-3` vs `4.5e-3`); the scripts keep that split: barplots use 1.3, Table 4 uses 3.1. That is also how the paper files are split.

**Figures 4 and 18:** rebuilt from those tables. Output PDFs are the same size as the paper PDFs (38 380 and 40 938 bytes) but **not byte-identical** (different hashes). No pixel diff was run.

**Not in this check:** training, evaluation, MLflow aggregation, or PFΔ baselines (those cells are frozen in `make_latex_tables.py`).
