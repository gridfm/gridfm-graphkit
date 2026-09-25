# Reproducing the GENCO paper (GPU runtime)

PF inference timings for the GENCO paper. The OPF GENCO curves in the paper use these same PF timings. There is no separate OPF GENCO benchmark.

PowerModels (Julia) timings, environment pins, and `submit_matrix.sh` are in [gridfm-datakit `genco-paper-repro`](https://github.com/gridfm/gridfm-datakit/tree/genco-paper-repro), `scripts/runtime/README.md`.

## Data

Both protocols read processed graphs, not raw parquet and not a trained checkpoint. The model is built at the config hidden size (tiny 12 / small 24 / base 48) with a random `baseMVA`. Wall time does not depend on the paper weights.

```text
$GENCO_DATA_PATH/<network>/processed/data_index_*.pt
```

The paper jobs used the PF finetuning tree:

```text
/dccstor/gridfm/powermodels_data/v4/finetuning/pf
```

That is the full PF tree. It is not a separate "pf small" release. Each job uses the first 10,000 `data_index_*.pt` files per network (preloaded for in-memory, copied to node-local `/tmp` for from-disk). Sample counts timed against that pool match the PowerModels matrix (IEEE 4M/3M/2M/2M, GOC 500k/50k/10k).

Override the path with `GENCO_DATA_PATH`. `GENCO_PYTHON` defaults to `python` on `PATH` (paper jobs used the evaluation-repo venv: Python 3.12.9, PyTorch 2.8.0+cu128).

## Submit

Four LSF jobs, PF only: in-memory IEEE, in-memory GOC, from-disk IEEE, from-disk GOC. Resources match the paper runs: 1 exclusive H100, 128G, 40 slots, one host, CCC 7xx (`cccxc702`–`716`).

```bash
bash scripts/runtime/submit_pf_matrix.sh
```

| Job name | Launcher | Paper job |
|----------|----------|-----------|
| `genco_pf_in_memory_ieee` | `run_pf_in_memory_ieee.py` | 954945 `matrix_aligned_ram_small` |
| `genco_pf_in_memory_goc` | `run_pf_in_memory_goc.py` | 954946 `matrix_aligned_ram_large` |
| `genco_pf_from_disk_ieee` | `run_pf_from_disk_ieee.py` | 954947 `matrix_aligned_get_small` |
| `genco_pf_from_disk_goc` | `run_pf_from_disk_goc.py` | 954948 `matrix_aligned_get_large` |

Committed CSVs from those jobs are under `genco_pf_in_memory/` and `genco_pf_from_disk/`. Plotters in this directory read them. They do not rerun the benchmark.

## Residuals

Active-power residual figure is not a runtime job. Script and CSVs:

```text
scripts/datakit_pf/plot_grid_scaling_active_residuals_pf.py
scripts/datakit_pf/results/pf_eval_aggregated.csv
```
