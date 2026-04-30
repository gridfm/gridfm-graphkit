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
* `benchmark` - Benchmark train-dataloader iteration speed

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

### Examples

**Standard Training:**

```bash
gridfm_graphkit train --config examples/config/case30_ieee_base.yaml --data_path examples/data
```

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
| `--compile [MODE]` | `str` | Enable `torch.compile` mode. Valid values: `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`. If flag is passed without a value, mode is `default`. | `None` |
| `--bfloat16` | `flag` | Cast model to `torch.bfloat16` (`model.to(torch.bfloat16)`). | `False` |
| `--tf32` | `flag` | Enable TF32 on Ampere+ GPUs via `torch.set_float32_matmul_precision("high")`. | `False` |
| `--profiler` | `str` | Enable Lightning profiler (`simple`, `advanced`, `pytorch`). | `None` |

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

Use built-in help for full command details:

```bash
gridfm_graphkit --help
gridfm_graphkit <command> --help
```

---
