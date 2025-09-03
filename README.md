<p align="center">
  <img src="https://raw.githubusercontent.com/gridfm/gridfm-graphkit/refs/heads/main/docs/figs/KIT.png" alt="GridFM logo" style="width: 40%; height: auto;"/>
  <br/>
</p>

<p align="center" style="font-size: 25px;">
    <b>gridfm-graphkit</b>
</p>


[![DOI](https://zenodo.org/badge/1007159095.svg)](https://doi.org/10.5281/zenodo.17016737)
[![Docs](https://img.shields.io/badge/docs-available-brightgreen)](https://gridfm.github.io/gridfm-graphkit/)
![Coverage](https://img.shields.io/badge/coverage-83%25-yellowgreen)
![Python](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.12-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

This library is brought to you by the GridFM team to train, finetune and interact with a foundation model for the electric power grid.

---

# Installation

You can install `gridfm-graphkit` directly from PyPI:

```bash
pip install gridfm-graphkit
```

To contribute or develop locally, clone the repository and install in editable mode:

```bash
git clone git@github.com:gridfm/gridfm-graphkit.git
cd gridfm-graphkit
python -m venv venv
source venv/bin/activate
pip install -e .
```

For documentation generation and unit testing, install with the optional `dev` and `test` extras:

```bash
pip install -e .[dev,test]
```


# CLI commands

An interface to train, fine-tune, and evaluate GridFM models using configurable YAML files and MLflow tracking.

```bash
gridfm_graphkit <command> [OPTIONS]
```

Available commands:

* `train` – Train a new model from scrathc
* `finetune` – Fine-tune an existing pre-trained model
* `evaluate` – Evaluate model performance on a dataset
* `predict` – Run inference and save predictions

---

## Training Models

```bash
gridfm_graphkit train --config path/to/config.yaml
```

### Arguments

| Argument         | Type   | Description                                                      | Default |
| ---------------- | ------ | ---------------------------------------------------------------- | ------- |
| `--config`       | `str`  | **Required**. Path to the training configuration YAML file.    | `None`  |
| `--exp_name`     | `str`  | **Optional**. MLflow experiment name.                            | `timestamp`  |
| `--run_name`     | `str`  | **Optional**. MLflow run name.                                   | `run`  |
| `--log_dir  `    | `str`  | **Optional**. MLflow logging directory.                              | `mlruns`  |
| `--data_path`    | `str`  | **Optional**. Root dataset directory.                            | `data`  |

### Examples

**Standard Training:**

```bash
gridfm_graphkit train --config examples/config/case30_ieee_base.yaml --data_path examples/data
```

---

## Fine-Tuning Models

```bash
gridfm_graphkit finetune --config path/to/config.yaml --model_path path/to/model.pth
```

### Arguments

| Argument       | Type  | Description                                     | Default   |
| -------------- | ----- | ----------------------------------------------- | --------- |
| `--config`     | `str` | **Required**. Fine-tuning configuration file.   | `None`    |
| `--model_path` | `str` | **Required**. Path to a pre-trained model file. | `None`    |
| `--exp_name`   | `str` | MLflow experiment name.                         | timestamp |
| `--run_name`   | `str` | MLflow run name.                                | `run`     |
| `--log_dir`    | `str` | MLflow logging directory.                       | `mlruns`  |
| `--data_path`  | `str` | Root dataset directory.                         | `data`    |


---

## Evaluating Models

```bash
gridfm_graphkit evaluate --config path/to/eval.yaml --model_path path/to/model.pth
```

### Arguments

| Argument       | Type  | Description                              | Default   |
| -------------- | ----- | ---------------------------------------- | --------- |
| `--config`     | `str` | **Required**. Path to evaluation config. | `None`    |
| `--model_path` | `str` | Path to the trained model file.          | `None`    |
| `--exp_name`   | `str` | MLflow experiment name.                  | timestamp |
| `--run_name`   | `str` | MLflow run name.                         | `run`     |
| `--log_dir`    | `str` | MLflow logging directory.                | `mlruns`  |
| `--data_path`  | `str` | Dataset directory.                       | `data`    |

---

## Running Predictions

```bash
gridfm_graphkit predict --config path/to/config.yaml --model_path path/to/model.pth
```

### Arguments

| Argument        | Type  | Description                                   | Default   |
| --------------- | ----- | --------------------------------------------- | --------- |
| `--config`      | `str` | **Required**. Path to prediction config file. | `None`    |
| `--model_path`  | `str` | Path to the trained model file.               | `None`    |
| `--exp_name`    | `str` | MLflow experiment name.                       | timestamp |
| `--run_name`    | `str` | MLflow run name.                              | `run`     |
| `--log_dir`     | `str` | MLflow logging directory.                     | `mlruns`  |
| `--data_path`   | `str` | Dataset directory.                            | `data`    |
| `--output_path` | `str` | Directory where predictions are saved.        | `data`    |

---
