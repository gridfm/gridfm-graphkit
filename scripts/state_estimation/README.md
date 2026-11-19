# State estimation with GENCO

Code for the state estimation (SE) results of *GENCO: A Unified Neural Solver
Embedded in a Development Framework for Steady-State Grid Analysis*
([arXiv:2608.09921](https://arxiv.org/abs/2608.09921)). It trains GENCO to
estimate the grid state from noisy, partial and corrupted SCADA measurements,
and compares it with a weighted least squares (WLS) estimator with bad-data removal.

## Requirements

Python 3.10-3.12 on Linux or macOS. Training uses one GPU per grid. The WLS
baseline and the evaluation run on CPU.

```bash
pip install -e ".[se,test]"
TORCH=$(python -c "import torch; v = torch.__version__; print(v if '+' in v else v + '+cpu')")
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
```

Installation takes a few minutes. To check it, run `python -m pytest tests`.
The tests train and evaluate a small model on the IEEE 14-bus scenarios in
`tests/data`, in about 5 minutes on a laptop CPU.

## Files

| File | Purpose |
| --- | --- |
| `base.yaml`, `generate_configs.py` | Model, measurement and training settings. The script writes `configs/`. |
| `configs/<grid>_base_0.yaml` | State estimation training on IEEE 14, 30, 57 and 118. |
| `configs/<grid>_pe_0.yaml` | Joint state and branch parameter estimation. |
| `generate_datakit_configs.py` | gridfm-datakit configs for the test sets with admittance and topology perturbations. |
| `make_eval_configs.py` | Evaluation settings for one trained model: measurement density, outlier rate, grid parameter errors. Writes one `gridfm_graphkit evaluate` command per setting. |
| `make_figures.py` | Builds the SE figures and the parameter estimation table from the evaluation runs. |
| `export_results.py`, `results/` | Evaluation results of the paper, collected from the evaluation runs into one CSV. |

The model and the WLS baseline are in `gridfm_graphkit/tasks/se_task.py` and
`gridfm_graphkit/models/pandapower_se.py`. Measurement simulation is in
`gridfm_graphkit/datasets/masking.py`.

## Figures of the paper

```bash
python scripts/state_estimation/make_figures.py --results scripts/state_estimation/results/se_evaluation_results.csv --out_dir se_figures
```

This takes a few seconds.

## Reproducing the results

1. **Data.** Training data is on Hugging Face as `gridfm/opf_small_<grid>`.
   The test sets are generated with [gridfm-datakit](https://github.com/gridfm/gridfm-datakit),
   four per grid, one per admittance noise sigma and topology level k (folder `perturbation_<sigma>_<k>`):
   ```bash
   hf download gridfm/opf_small_case14_ieee --repo-type dataset --local-dir data/case14_ieee/raw
   python scripts/state_estimation/generate_datakit_configs.py --train_data_dir data --out_dir data/se_perturbations --grids case14_ieee
   for config in data/se_perturbations/case14_ieee/perturbation_*/config.yaml; do gridfm_datakit generate "$config"; done
   ```
2. **Train.**
   ```bash
   gridfm_graphkit train --config scripts/state_estimation/configs/case14_ieee_base_0.yaml --data_path data
   ```
3. **Evaluate** against WLS. Pass the model and normalizer stats saved by the training run:
   ```bash
   python scripts/state_estimation/make_eval_configs.py --grid case14_ieee \
       --config scripts/state_estimation/configs/case14_ieee_base_0.yaml \
       --model_path <run>/artifacts/model/best_model_state_dict.pt \
       --normalizer_stats <run>/artifacts/stats/normalizer_stats.pt \
       --perturbation_data_dir data/se_perturbations --out_dir se_eval/case14_ieee --exp_name se_eval_case14_ieee
   bash se_eval/case14_ieee/commands.sh
   ```
4. **Figures.** Repeat steps 2 and 3 for `case30_ieee`, `case57_ieee` and
   `case118_ieee`, then:
   ```bash
   python scripts/state_estimation/make_figures.py --mlflow_dir mlruns --out_dir se_figures \
       --experiments case14=se_eval_case14_ieee case30=se_eval_case30_ieee \
                     case57=se_eval_case57_ieee case118=se_eval_case118_ieee
   ```
   This writes the accuracy against measurement density and outlier rate
   (`fig5_base_case.pdf`), the robustness to admittance errors
   (`fig6_unknown_perturbation.pdf`) and the accuracy where WLS does not
   converge (`fig19_non_observable.pdf`). Figure numbers follow arXiv:2608.09921.
   For joint parameter estimation, evaluate each `<grid>_pe_0` model on
   `perturbation_0.2_<k>` and add `--pe_experiment <experiment>`.
