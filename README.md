# GridFM model evaluation

Create a python virtual environment and install the requirements
```bash
virtualenv venv
pip install -r requirements.txt
```

Install the package in editable mode during development phase:

```bash
pip install -e .
```


---

# Training Script with optional grid search

This script allows you to train models with configurable parameters, optional grid search, and experiment tracking using **MLflow**.

## Usage

```bash
python train.py --config path/to/config.yaml --grid path/to/grid.yaml --exp my_experiment --data_path /path/to/data
```

## Command-Line Arguments

| Argument        | Type   | Default                  | Description                                                         |
|---------------------------------|--------|--------------------------|--------------------------------------------------------------|
| `--config`      | `str`  | `config/default.yaml`    | **(Required)** Path to the base configuration YAML file.            |
| `--grid`        | `str`  | `None`                   | **(Optional)** Path to the grid search YAML file. If not provided, grid search is not performed. |
| `--exp`         | `str`  | `None`                   | **(Optional)** Experiment name for **MLflow** tracking. If not provided, the run won't be logged under a specific experiment and the name assigned to it will be equal to the current timestamp |
| `--data_path`   | `str`  | `../data` | **(Optional)** Root directory of the dataset. Defaults to the `data` folder one level up from the current working directory. |

---

## Example Commands

### **1. Basic Training Run**
```bash
python train.py --config config/gridFMv0.1_pretraining.yaml --exp "GridFMv0.1_pretraining"
```

### **2. Training with Grid Search**
```bash
python train.py --config config/gridFMv0.1_pretraining.yaml --grid config/grid_search_baseline.yaml
```

### **3. Custom Data Path**
```bash
python train.py --config config/gridFMv0.1_pretraining.yaml --data_path /dccstor/gridfm/PowerGraph
```


# Evaluation Script

This script evaluates pre-trained models using **MLflow** for experiment tracking.

## Usage

```bash
python eval.py --config path/to/config.yaml --eval_name my_evaluation --model_exp_id my_MLflow_experiment_id --model_run_id my_MLflow_run_id --model_name model_name_in_MLflow
```

## Command-Line Arguments

| Argument          | Type   | Default                  | Description                                                                                                                     |
|-------------------|--------|--------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| `--model_path`    | `str`  | `None`                   | **(Optional)** Direct path to the model file. If provided, a new MLflow experiment is created for evaluation.                   |
| `--model_exp_id`  | `str`  | `None`                   | **(Required if `--model_path` is not provided)** ID of the MLflow experiment associated with the logged model.                  |
| `--model_run_id`  | `str`  | `None`                   | **(Required if `--model_path` is not provided)** Run ID within the MLflow experiment that contains the model.                   |
| `--model_name`    | `str`  | `best_model`             | **(Optional)** Name of the model file within the MLflow artifacts directory.                                                     |
| `--config`        | `str`  | `None`                   | **(Required)** Path to the configuration YAML file.                                                                              |
| `--eval_name`     | `str`  | `None`                   | **(Required)** Name for the evaluation run in MLflow.                                                                            |
| `--data_path`     | `str`  | `../data`                | **(Optional)** Path to the dataset directory. Defaults to `../data`.                                                             |

---

## Example Commands

### **1. Evaluate Using a Direct Model Path**
```bash
python eval.py --model_path GridFM_v01.pth --config config/case300_ieee_base.yaml --eval_name eval_case300
```

### **2. Evaluate a Logged MLflow Model**
```bash
python eval.py --config config/case300_ieee_base.yaml --eval_name eval_case300 --model_exp_id 1 --model_run_id abcdef123456 --model_name best_model
```

---

## Notes

- **Model Selection:**
  - Use `--model_path` for standalone model files.
  - Use `--model_exp_id`, `--model_run_id`, and `--model_name` for models stored in MLflow.

- **Outputs:**
  - Evaluation results are saved in MLflow.
  - Optional plots are stored as **HTML** files


