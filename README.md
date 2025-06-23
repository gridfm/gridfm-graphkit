# gridfm-graphkit
[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://gridfm.github.io/gridfm-graphkit/)

This library is brought to you by the GridFM team to train, finetune and interact with a foundation model for the electric power grid.

---

<p align="center">
  <img src="docs/figs/pre_training.png" alt="GridFM logo"/>
  <br/>
</p>

# Installation

Create a python virtual environment and install the requirements
```bash
git clone git@github.com:gridfm/gridfm-graphkit.git
cd gridfm-graphkit
python -m venv venv
source venv/bin/activate
pip install .
```

Install the package in editable mode during development phase:

```bash
pip install -e .[dev,test]
```

# gridfm-graphkit CLI

An interface to train, fine-tune, and evaluate GridFM models using configurable YAML files and MLflow tracking.

```bash
gridfm_graphkit <command> [OPTIONS]
```

Available commands:

* `train` – Train a new model
* `predict` – Evaluate an existing model
* `finetune` – Fine-tune a pre-trained model

---

## Training Models

```bash
gridfm_graphkit train --config path/to/config.yaml
```

### Arguments

| Argument         | Type   | Description                                                      | Default |
| ---------------- | ------ | ---------------------------------------------------------------- | ------- |
| `--config`       | `str`  | **Required for standard training**. Path to base config YAML.    | `None`  |
| `--grid`         | `str`  | **Optional**. Path to grid search YAML. Not supported with `-c`. | `None`  |
| `--exp`          | `str`  | **Optional**. MLflow experiment name. Defaults to a timestamp.   | `None`  |
| `--data_path`    | `str`  | **Optional**. Root dataset directory.                            | `data`  |
| `-c`             | `flag` | **Optional**. Enable checkpoint mode.                            | `False` |
| `--model_exp_id` | `str`  | **Required if `-c` is used**. MLflow experiment ID.              | `None`  |
| `--model_run_id` | `str`  | **Required if `-c` is used**. MLflow run ID.                     | `None`  |

### Examples

**Standard Training:**

```bash
gridfm_graphkit train --config config/train.yaml --exp "run1"
```

**Grid Search Training:**

```bash
gridfm_graphkit train --config config/train.yaml --grid config/grid.yaml
```

**Training from Checkpoint:**

```bash
gridfm_graphkit train -c --model_exp_id 123 --model_run_id abc
```

---

## Evaluating Models

```bash
gridfm_graphkit predict --model_path model.pth --config config/eval.yaml --eval_name run_eval
```

### Arguments

| Argument         | Type  | Description                                                       | Default      |
| ---------------- | ----- | ----------------------------------------------------------------- | ------------ |
| `--model_path`   | `str` | **Optional**. Path to a model file.                               | `None`       |
| `--model_exp_id` | `str` | **Required if `--model_path` is not used**. MLflow experiment ID. | `None`       |
| `--model_run_id` | `str` | **Required if `--model_path` is not used**. MLflow run ID.        | `None`       |
| `--model_name`   | `str` | **Optional**. Filename inside MLflow artifacts.                   | `best_model` |
| `--config`       | `str` | **Required**. Path to evaluation config.                          | `None`       |
| `--eval_name`    | `str` | **Required**. Name of the evaluation run in MLflow.               | `None`       |
| `--data_path`    | `str` | **Optional**. Path to dataset directory.                          | `data`       |

### Examples

**Evaluate a Logged MLflow Model:**

```bash
gridfm_graphkit predict --config config/eval.yaml --eval_name run_eval --model_exp_id 1 --model_run_id abc
```

---

## Fine-Tuning Models

```bash
gridfm_graphkit finetune --config path/to/config.yaml --model_path path/to/model.pth
```

### Arguments

| Argument       | Type  | Description                                     | Default |
| -------------- | ----- | ----------------------------------------------- | ------- |
| `--config`     | `str` | **Required**. Fine-tuning configuration file.   | `None`  |
| `--model_path` | `str` | **Required**. Path to a pre-trained model file. | `None`  |
| `--exp`        | `str` | **Optional**. MLflow experiment name.           | `None`  |
| `--data_path`  | `str` | **Optional**. Root dataset directory.           | `data`  |
