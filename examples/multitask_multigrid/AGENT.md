# Multi-task multi-grid case14 / case30 / case57

Instructions for launching the comparison. Do not change the model code unless a run crashes. Do not edit `gridfm_graphkit/models/gnn_heterogeneous_gns.py`. Run one seed only (seed 0, already set in every config). Use one GPU at a time.

## What to train

Plain GENCO, hidden size 12, 12 layers, 8 heads, batch size 64, 50 epochs, AdamW at 5e-4. Each dataset has its own `HeteroDataMVANormalizer`.

| Run | Config | Data | Samples |
|---|---|---|---|
| Mixed PF+OPF on case14, case30, and case57 | `pf_opf_case14_30_57.yaml` | both trees | 1,000 per task per grid (6,000 total) |
| Case57 PF only, same total count | `pf_case57_6000.yaml` | `pf/` | 6,000 PF |
| Case57 OPF only, same total count | `opf_case57_6000.yaml` | `opf/` | 6,000 OPF |
| Case57 PF only, one task-grid count | `pf_case57_1000.yaml` | `pf/` | 1,000 PF |
| Case57 OPF only, one task-grid count | `opf_case57_1000.yaml` | `opf/` | 1,000 OPF |

The mixed model is `GNS_heterogeneous_MultiTask`: the unchanged GENCO forward plus `set_task`, which swaps the power-flow or OPF decoder. The four case57 runs are `GNS_heterogeneous`. The trunk is the same. PF graphs use the power-flow mask and the PF loss (layered physics 0.1, masked bus MSE 0.9). OPF graphs use the OPF mask and the OPF loss (layered physics 0.1, masked generator MSE 0.1, masked bus MSE 0.75, Qg penalty 0.001).

Each training step gets one batch of 32 PF graphs and one of 32 OPF graphs (Lightning `CombinedLoader`, one loader per task). Each task batch mixes case14, case30, and case57 at random. Each batch runs the plain GENCO forward with its task, and the step averages the two task losses, so every update sees both tasks. This is layout 1 in `BATCH_TIMING.md`: on an H100 it took 226 ms per step against 667 ms for six homogeneous forwards. Validation has one loader per task, and `Validation loss` averages over both. A test loader is one grid and one task.

## Data

On the CCC cluster the data is already downloaded and processed under `/dccstor/gridfm/powermodels_data/v4/finetuning/{pf,opf}/{case14,case30,case57}_ieee` (about 200,000 scenarios each). The single-task runs read `.../finetuning/pf` or `.../finetuning/opf` directly. The mixed run reads one directory with a symlink per network name:

```bash
MT_ROOT=/u/apu/multitask/data_pf_opf
mkdir -p $MT_ROOT
for k in pf opf; do for c in case14_ieee case30_ieee case57_ieee; do
  ln -sfn /dccstor/gridfm/powermodels_data/v4/finetuning/$k/$c $MT_ROOT/${c}_$k
done; done
```

Elsewhere, `download_pf_opf.py --dest "$DATA_ROOT"` fetches the Hugging Face slices into `$DATA_ROOT/{pf,opf}/<grid>`. Make the same symlinks from them.

`HeteroGridDatasetDisk` skips processing when `processed/processed_raw_files.done` exists.

## Launch (LSF)

```bash
REPO=/u/apu/multitask/gridfm-graphkit-minimal
VENV=/u/apu/gridfm_model_evaluation/venv
PM=/dccstor/gridfm/powermodels_data/v4/finetuning
LOGS=/u/apu/multitask/logs
GPU='num=1:mode=exclusive_process:gmodel=NVIDIAH10080GBHBM3'

submit() {  # name config data_path
  bsub -q normal -gpu "$GPU" -M 32G -n 8 -J "$1" -o "$LOGS/$1_%J.out" \
    "cd $REPO; source $VENV/bin/activate && export PYTHONPATH=$REPO MLFLOW_ALLOW_FILE_STORE=true && \
     python -u -m gridfm_graphkit train --config $REPO/examples/multitask_multigrid/$2 --data_path $3 \
     --num_workers 2 --exp_name multitask_multigrid --run_name $1 --log_dir /u/apu/multitask/mlruns_multitask_multigrid"
}

submit genco_pf_opf_case14_30_57 pf_opf_case14_30_57.yaml /u/apu/multitask/data_pf_opf
submit genco_pf_case57_6000      pf_case57_6000.yaml      $PM/pf
submit genco_opf_case57_6000     opf_case57_6000.yaml     $PM/opf
submit genco_pf_case57_1000      pf_case57_1000.yaml      $PM/pf
submit genco_opf_case57_1000     opf_case57_1000.yaml     $PM/opf
```

Configs set `accelerator: auto` and `devices: 1`. Each job takes one GPU, and the jobs are independent. Do not start a second seed.

The 1,000-scenario case57 runs shuffle the full case57 cache with seed 0 and keep 1,000 scenarios. That is the same case57 slice as in the mixed run, because all of them read the same processed files. Their test sets are each run's own 10% holdout. The 6,000-scenario test set is larger and is not the same holdout.

## Where to read results

MLflow run artifacts, `artifacts/test/`:

- Mixed model, case57 power flow: `case57_ieee_pf_metrics.csv`
- Mixed model, case57 OPF: `case57_ieee_opf_metrics.csv`
- PF-only runs: `case57_ieee_metrics.csv` under `genco_pf_case57_6000` and `genco_pf_case57_1000`
- OPF-only runs: `case57_ieee_metrics.csv` under `genco_opf_case57_6000` and `genco_opf_case57_1000`

Ignore case14 and case30 CSVs from the mixed run for this comparison.

Compare power-flow metrics (active residual MW, reactive residual MVar, PBE) of the mixed model on case57 against both PF-only runs. Compare OPF metrics (active and reactive residual, RMSE Pg, optimality gap, thermal from/to, mean Qg violation) of the mixed model on case57 against both OPF-only runs. Lower is better.

## If a run crashes

Restart only that run. A `Numpy` invalid-divide warning during test correlation is the same warning as a finished GENCO run and is not a failure. If processing says the scenario ids are not contiguous, the download is missing a partition. If training reports fewer scenarios than the config asked for, the processed cache was built from a shorter download: delete `processed/` and let it rebuild.
