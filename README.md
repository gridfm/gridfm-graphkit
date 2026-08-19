<p align="center">
  <img src="https://raw.githubusercontent.com/gridfm/gridfm-graphkit/refs/heads/main/docs/figs/KIT.png" alt="GridFM logo" style="width: 40%; height: auto;"/>
  <br/>
</p>

<p align="center" style="font-size: 25px;">
    <b>gridfm-graphkit</b>
</p>


[![Docs](https://img.shields.io/badge/docs-available-brightgreen)](https://gridfm.github.io/gridfm-graphkit/)
![Coverage](https://img.shields.io/badge/coverage-83%25-yellowgreen)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/12802/badge)](https://www.bestpractices.dev/projects/12802)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/gridfm/gridfm-graphkit/badge)](https://scorecard.dev/viewer/?uri=github.com/gridfm/gridfm-graphkit)
![Python](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.12-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

This library is brought to you by the GridFM team to train, finetune and interact with a foundation model for the electric power grid.

## Citation

If you use `gridfm-graphkit` in your research, please cite:

```bibtex
@article{puech2026gencounifiedneural,
  title={GENCO - A Unified Neural Solver Embedded in a Development Framework for Steady-State Grid Analysis},
  author={Alban Puech and Matteo Mazzonelli and Tamara R. Govindasamy and Mangaliso Mngomezulu and Héctor Maeso-García and Thomas Tolhurst and Javad Bayazi and Ali Moeini and Naomi Simumba and Celia Cintas and David Nelischer and Romeo Kienzler and Jonas Weiss and Anna Varbella and Florian Dörfler and Gabriela Hug and Martin Mevissen and Juan Bernabé-Moreno and François Mirallès and Hendrik F. Hamann and Etienne Vos and Thomas Brunschwiler},
  journal={arXiv preprint arXiv:2608.09921},
  year={2026},
  url={https://arxiv.org/abs/2608.09921}
}
```

---

# Installation

Create and activate a virtual environment (make sure you use the right python version = 3.10, 3.11 or 3.12. I highly recommend 3.12)
```bash
python -m venv venv
source venv/bin/activate
```

Install gridfm-graphkit from PyPI
```bash
pip install gridfm-graphkit
```

**`torch-scatter` is a required dependency.** It cannot be bundled in `pyproject.toml` because the correct wheel depends on your PyTorch and CUDA versions, so it must be installed separately.

Get PyTorch + CUDA version for torch-scatter
```bash
TORCH_CUDA_VERSION=$(python -c "import torch; print(torch.__version__ + ('+cpu' if torch.version.cuda is None else ''))")
```

Install the correct torch-scatter wheel
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_CUDA_VERSION}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH_CUDA_VERSION}.html
```

If `data.pyg.org` has no prebuilt wheel for your PyTorch version, build from source instead (this
compiles against your installed torch, so it works with any version):
```bash
pip install torch-scatter torch-sparse --no-build-isolation
```
Building from source requires a C++ compiler (and a matching CUDA toolkit with
`nvcc` for GPU builds).


For documentation generation and unit testing, install with the optional `dev` and `test` extras:

```bash
pip install "gridfm-graphkit[dev,test]"
```


# CLI commands

Interface to train, fine-tune, evaluate, and run inference on GridFM models using YAML configs and MLflow tracking.

```bash
gridfm_graphkit <command> [OPTIONS]
```

Available commands:

* `train` - Train a new model from scratch
* `finetune` - Fine-tune an existing pre-trained model
* `evaluate` - Evaluate model performance on a dataset
* `predict` - Run inference and save predictions

---

## Training Models

```bash
gridfm_graphkit train --config path/to/config.yaml
```

### Arguments

| Argument | Type | Description | Default |
| -------- | ---- | ----------- | ------- |
| `--config` | `str` | **Required**. Path to the training configuration YAML file. | `None` |
| `--exp_name` | `str` | MLflow experiment name. | `timestamp` |
| `--run_name` | `str` | MLflow run name. | `run` |
| `--log_dir` | `str` | MLflow tracking/logging directory. | `mlruns` |
| `--data_path` | `str` | Root dataset directory. | `data` |
| `--compile [MODE]` | `str` | Enable `torch.compile` mode. Valid values: `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`. If flag is passed without a value, mode is `default`. | `None` |
| `--bfloat16` | `flag` | Cast model to `torch.bfloat16` (`model.to(torch.bfloat16)`). | `False` |
| `--tf32` | `flag` | Enable TF32 on Ampere+ GPUs via `torch.set_float32_matmul_precision("high")`. | `False` |
| `--dataset_wrapper` | `str` | Registered dataset wrapper name (see `DATASET_WRAPPER_REGISTRY`), e.g. `SharedMemoryCacheDataset`. | `None` |
| `--plugins` | `list[str]` | Python packages to import for plugin registration, e.g. `gridfm_graphkit_ee`. | `[]` |
| `--num_workers` | `int` | Override `data.workers` from YAML. Use `0` to debug worker crashes. | `None` |
| `--dataset_wrapper_cache_dir` | `str` | Disk cache directory for dataset wrapper; cache is loaded from here when present and saved after first population. | `None` |
| `--profiler` | `str` | Enable Lightning profiler (`simple`, `advanced`, `pytorch`). | `None` |
| `--compute_dc_ac_metrics` | `flag` | Compute ground-truth AC/DC power balance metrics on the test split. | `False` |
| `--mp_context` | `str` | DataLoader multiprocessing start method (`spawn`, `fork`, `forkserver`). Defaults to PyTorch's automatic choice. On Linux, `spawn` is recommended for safety (CUDA + fork is unsafe); other choices emit a warning. | `None` |
| `--config_conversion` | `str` | How to handle a config older than the required version (see [Config file versioning](#config-file-versioning)): `no` (fail with instructions), `auto_inline` (convert in memory), `auto_copy` (write an upgraded copy). | `no` |
| `--converted_config_path` | `str` | Destination file for `--config_conversion auto_copy`. Defaults to a sibling `<config>.v<N>.yaml`. | `None` |
### Examples

**Standard Training:**

```bash
gridfm_graphkit train --config examples/config/case30_ieee_base.yaml --data_path examples/data
```

---

## Config file versioning

Config files carry a top-level integer `version` key. The current schema version
is **1** (`version: 1`); all shipped example configs are tagged accordingly.
A config with **no** `version` key is treated as **v0** (the pre-versioning
schema, before the configurable optimizer/scheduler landed in
[#92](https://github.com/gridfm/gridfm-graphkit/pull/92)).

**This version of gridfm-graphkit requires config version 1.** If you pass an
older config, the CLI stops with instructions instead of silently mis-reading it.
Choose how to migrate with `--config_conversion`:

| Mode | Behaviour |
| ---- | --------- |
| `no` *(default)* | Fail with an explanation of the options. |
| `auto_inline` | Convert to v1 in memory and run now; the file on disk is left unchanged. |
| `auto_copy` | Convert, write an upgraded copy (`--converted_config_path`, or a derived `<config>.v1.yaml`), then run from it. |

```bash
# Migrate an old config in place in memory and train immediately
gridfm_graphkit train --config old_v0.yaml --data_path data --config_conversion auto_inline

# Or write an upgraded copy you can inspect and keep
gridfm_graphkit train --config old_v0.yaml --data_path data \
  --config_conversion auto_copy --converted_config_path config_v1.yaml
```

The **v0 → v1** conversion maps the old hard-coded `AdamW` + `ReduceLROnPlateau`
optimizer block onto the explicit v1 schema:

| v0 key | v1 location |
| ------ | ----------- |
| *(implicit AdamW)* | `optimizer.type: AdamW` |
| `learning_rate` | `optimizer.learning_rate` |
| `beta1`, `beta2` | `optimizer.optimizer_params.betas: [beta1, beta2]` |
| *(implicit ReduceLROnPlateau, mode=min)* | `optimizer.scheduler_type` + `scheduler_params.mode: min` |
| `lr_decay` | `optimizer.scheduler_params.factor` |
| `lr_patience` | `optimizer.scheduler_params.patience` |

---

## Fine-Tuning Models

```bash
gridfm_graphkit finetune --config path/to/config.yaml --model_path path/to/model.pt
```

### Arguments

| Argument | Type | Description | Default |
| -------- | ---- | ----------- | ------- |
| `--config` | `str` | **Required**. Fine-tuning configuration file. | `None` |
| `--model_path` | `str` | **Required**. Path to a pre-trained model state dict. | `None` |
| `--exp_name` | `str` | MLflow experiment name. | `timestamp` |
| `--run_name` | `str` | MLflow run name. | `run` |
| `--log_dir` | `str` | MLflow logging directory. | `mlruns` |
| `--data_path` | `str` | Root dataset directory. | `data` |
| `--compile [MODE]` | `str` | Enable `torch.compile` mode. Valid values: `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`. If flag is passed without a value, mode is `default`. | `None` |
| `--bfloat16` | `flag` | Cast model to `torch.bfloat16` (`model.to(torch.bfloat16)`). | `False` |
| `--tf32` | `flag` | Enable TF32 on Ampere+ GPUs via `torch.set_float32_matmul_precision("high")`. | `False` |
| `--dataset_wrapper` | `str` | Registered dataset wrapper name (see `DATASET_WRAPPER_REGISTRY`), e.g. `SharedMemoryCacheDataset`. | `None` |
| `--plugins` | `list[str]` | Python packages to import for plugin registration, e.g. `gridfm_graphkit_ee`. | `[]` |
| `--num_workers` | `int` | Override `data.workers` from YAML. Use `0` to debug worker crashes. | `None` |
| `--dataset_wrapper_cache_dir` | `str` | Disk cache directory for dataset wrapper; cache is loaded from here when present and saved after first population. | `None` |
| `--profiler` | `str` | Enable Lightning profiler (`simple`, `advanced`, `pytorch`). | `None` |
| `--compute_dc_ac_metrics` | `flag` | Compute ground-truth AC/DC power balance metrics on the test split. | `False` |
| `--mp_context` | `str` | DataLoader multiprocessing start method (`spawn`, `fork`, `forkserver`). Defaults to PyTorch's automatic choice. On Linux, `spawn` is recommended for safety (CUDA + fork is unsafe); other choices emit a warning. | `None` |

---

## Evaluating Models

```bash
gridfm_graphkit evaluate --config path/to/eval.yaml --model_path path/to/model.pt
```

### Arguments

| Argument | Type | Description | Default |
| -------- | ---- | ----------- | ------- |
| `--config` | `str` | **Required**. Path to evaluation config. | `None` |
| `--model_path` | `str` | Path to the trained model state dict. | `None` |
| `--normalizer_stats` | `str` | Path to `normalizer_stats.pt` from a training run. Restores `fit_on_train` normalizers from saved statistics instead of re-fitting on current split. | `None` |
| `--exp_name` | `str` | MLflow experiment name. | `timestamp` |
| `--run_name` | `str` | MLflow run name. | `run` |
| `--log_dir` | `str` | MLflow logging directory. | `mlruns` |
| `--data_path` | `str` | Dataset directory. | `data` |
| `--compile [MODE]` | `str` | Enable `torch.compile` mode. Valid values: `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`. If flag is passed without a value, mode is `default`. | `None` |
| `--bfloat16` | `flag` | Cast model to `torch.bfloat16` (`model.to(torch.bfloat16)`). | `False` |
| `--tf32` | `flag` | Enable TF32 on Ampere+ GPUs via `torch.set_float32_matmul_precision("high")`. | `False` |
| `--dataset_wrapper` | `str` | Registered dataset wrapper name (see `DATASET_WRAPPER_REGISTRY`), e.g. `SharedMemoryCacheDataset`. | `None` |
| `--plugins` | `list[str]` | Python packages to import for plugin registration, e.g. `gridfm_graphkit_ee`. | `[]` |
| `--num_workers` | `int` | Override `data.workers` from YAML. Use `0` to debug worker crashes. | `None` |
| `--dataset_wrapper_cache_dir` | `str` | Disk cache directory for dataset wrapper; cache is loaded from here when present and saved after first population. | `None` |
| `--profiler` | `str` | Enable Lightning profiler (`simple`, `advanced`, `pytorch`). | `None` |
| `--compute_dc_ac_metrics` | `flag` | Compute ground-truth AC/DC power balance metrics on the test split. | `False` |
| `--save_output` | `flag` | Save predictions as `<grid_name>_predictions.parquet` under MLflow artifacts (`.../artifacts/test`). | `False` |
| `--mp_context` | `str` | DataLoader multiprocessing start method (`spawn`, `fork`, `forkserver`). Defaults to PyTorch's automatic choice. On Linux, `spawn` is recommended for safety (CUDA + fork is unsafe); other choices emit a warning. | `None` |

### Example with saved normalizer stats

When evaluating a model on a dataset, you can pass the normalizer statistics from the original training run to ensure the same normalization parameters are used:

```bash
gridfm_graphkit evaluate \
  --config examples/config/HGNS_PF_datakit_case118.yaml \
  --model_path mlruns/<experiment_id>/<run_id>/artifacts/model/best_model_state_dict.pt \
  --normalizer_stats mlruns/<experiment_id>/<run_id>/artifacts/stats/normalizer_stats.pt \
  --data_path data
```

> **Note:** The `--normalizer_stats` flag only affects normalizers with `fit_strategy = "fit_on_train"` (e.g. `HeteroDataMVANormalizer`). Per-sample normalizers (`HeteroDataPerSampleMVANormalizer`) always recompute their statistics from the current dataset regardless of this flag.

---

## Running Predictions

```bash
gridfm_graphkit predict --config path/to/config.yaml --model_path path/to/model.pt
```

### Arguments

| Argument | Type | Description | Default |
| -------- | ---- | ----------- | ------- |
| `--config` | `str` | **Required**. Path to prediction config file. | `None` |
| `--model_path` | `str` | Path to trained model state dict. Optional; may be defined in config. | `None` |
| `--normalizer_stats` | `str` | Path to `normalizer_stats.pt` from a training run. Restores `fit_on_train` normalizers from saved statistics. | `None` |
| `--exp_name` | `str` | MLflow experiment name. | `timestamp` |
| `--run_name` | `str` | MLflow run name. | `run` |
| `--log_dir` | `str` | MLflow logging directory. | `mlruns` |
| `--data_path` | `str` | Dataset directory. | `data` |
| `--dataset_wrapper` | `str` | Registered dataset wrapper name (see `DATASET_WRAPPER_REGISTRY`), e.g. `SharedMemoryCacheDataset`. | `None` |
| `--plugins` | `list[str]` | Python packages to import for plugin registration, e.g. `gridfm_graphkit_ee`. | `[]` |
| `--num_workers` | `int` | Override `data.workers` from YAML. Use `0` to debug worker crashes. | `None` |
| `--dataset_wrapper_cache_dir` | `str` | Disk cache directory for dataset wrapper; cache is loaded from here when present and saved after first population. | `None` |
| `--output_path` | `str` | Directory where predictions are saved as `<grid_name>_predictions.parquet`. | `data` |
| `--get_embeddings` | `flag` | Export final hidden embeddings to `<grid_name>_bus_embeddings.parquet` (and `<grid_name>_gen_embeddings.parquet` for OPF models that expose gen embeddings) in `--output_path`. | `False` |
| `--compile [MODE]` | `str` | Enable `torch.compile` mode. Valid values: `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`. If flag is passed without a value, mode is `default`. | `None` |
| `--bfloat16` | `flag` | Cast model to `torch.bfloat16` (`model.to(torch.bfloat16)`). | `False` |
| `--tf32` | `flag` | Enable TF32 on Ampere+ GPUs via `torch.set_float32_matmul_precision("high")`. | `False` |
| `--profiler` | `str` | Enable Lightning profiler (`simple`, `advanced`, `pytorch`). | `None` |
| `--mp_context` | `str` | DataLoader multiprocessing start method (`spawn`, `fork`, `forkserver`). Defaults to PyTorch's automatic choice. On Linux, `spawn` is recommended for safety (CUDA + fork is unsafe); other choices emit a warning. | `None` |

---

## Benchmarking Dataloader Throughput

```bash
gridfm_graphkit benchmark --config path/to/config.yaml
```

### Arguments

| Argument | Type | Description | Default |
| -------- | ---- | ----------- | ------- |
| `--config` | `str` | **Required**. Path to configuration YAML file. | `None` |
| `--data_path` | `str` | Root dataset directory. | `data` |
| `--epochs` | `int` | Number of epochs to iterate through the train dataloader. | `3` |
| `--dataset_wrapper` | `str` | Registered dataset wrapper name (see `DATASET_WRAPPER_REGISTRY`), e.g. `SharedMemoryCacheDataset`. | `None` |
| `--dataset_wrapper_cache_dir` | `str` | Directory for dataset wrapper disk cache. | `None` |
| `--num_workers` | `int` | Override `data.workers` from YAML. | `None` |
| `--plugins` | `list[str]` | Python packages to import for plugin registration. | `[]` |
| `--mp_context` | `str` | DataLoader multiprocessing start method (`spawn`, `fork`, `forkserver`). Defaults to PyTorch's automatic choice. On Linux, `spawn` is recommended for safety (CUDA + fork is unsafe); other choices emit a warning. | `None` |

Use built-in help for full command details:

```bash
gridfm_graphkit --help
gridfm_graphkit <command> --help
```

---

## Running Tests

### Unit and Integration Tests

Install the test dependencies first (if not already done):

```bash
pip install -e .[dev,test]
```

Run the full unit test suite:

```bash
pytest ./tests
```

Run the base set integration tests:

```bash
pytest ./integrationtests/test_base_set.py
```

### Running Base Set Tests on an LSF Cluster (GPU)

To submit the base set integration tests as an interactive LSF job with GPU access, use `bsub`. Adjust the paths to match your environment:

```bash
bsub -gpu "num=1" \
     -n 16 \
     -R "rusage[mem=32GB] span[hosts=1]" \
     -Is \
     -J gridfm_base_set_tests \
     /bin/bash -c "
       cd /path/to/gridfm-graphkit && \
       export PATH=/path/to/cuda/bin:\$PATH \
               CUDA_HOME=/path/to/cuda \
               LD_LIBRARY_PATH=/path/to/cuda/lib64:\$LD_LIBRARY_PATH && \
       source /path/to/venv/bin/activate && \
       pytest ./integrationtests/test_base_set.py
     "
```

Key `bsub` options used above:

| Option | Description |
| ------ | ----------- |
| `-gpu "num=1"` | Request 1 GPU |
| `-n 16` | Request 16 CPU slots |
| `-R "rusage[mem=32GB] span[hosts=1]"` | Reserve 32 GB of memory on a single host |
| `-Is` | Run as an interactive shell session |
| `-J <job_name>` | Assign a name to the job |

**Concrete example** (adapt paths to your cluster setup):

```bash
bsub -gpu "num=1" -n 16 -R "rusage[mem=32GB] span[hosts=1]" -Is -J hpo_trial_190 /bin/bash -c "cd /dccstor/terratorch/users/rkie/gitco/gridfm-graphkit && export PATH=/opt/share/cuda-12.8.1/bin:\$PATH CUDA_HOME=/opt/share/cuda-12.8.1 LD_LIBRARY_PATH=/opt/share/cuda-12.8.1/lib64:\$LD_LIBRARY_PATH && source /u/rkie/venvs/venv_gridfm-graphkit/bin/activate && pytest ./integrationtests/test_base_set.py"
```

---
