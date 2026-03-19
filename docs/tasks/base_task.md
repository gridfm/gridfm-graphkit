# Base Task

The `BaseTask` class is an abstract base class that provides the foundation for all task implementations in GridFM-GraphKit. It extends PyTorch Lightning's `LightningModule` and defines the common interface and shared functionality for training, validation, and testing.

## Overview

`BaseTask` serves as the parent class for all task-specific implementations, providing:

- **Abstract method definitions**: Enforces implementation of core methods in subclasses
- **Optimizer configuration**: Sets up AdamW optimizer with learning rate scheduling
- **Normalization statistics logging**: Saves normalization parameters for reproducibility
- **Hyperparameter management**: Automatically saves hyperparameters for experiment tracking

## BaseTask Class

::: gridfm_graphkit.tasks.base_task.BaseTask
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - forward
        - training_step
        - validation_step
        - test_step
        - predict_step
        - on_fit_start
        - configure_optimizers

## Methods

### `__init__(args, data_normalizers)`

Initialize the base task with configuration and normalizers.

**Parameters:**

- `args` (NestedNamespace): Experiment configuration containing all hyperparameters
- `data_normalizers` (list): List of normalizer objects, one per dataset

**Attributes Set:**

- `self.args`: Stores the configuration
- `self.data_normalizers`: Stores the normalizers
- Automatically calls `save_hyperparameters()` for experiment tracking

---

### `forward(*args, **kwargs)` (Abstract)

Defines the forward pass through the model. Must be implemented by subclasses.

**Returns:**

- Model output (structure depends on task implementation)

---

### `training_step(batch)` (Abstract)

Executes one training step. Must be implemented by subclasses.

**Parameters:**

- `batch`: A batch of data from the training dataloader

**Returns:**

- Loss tensor for backpropagation

---

### `validation_step(batch, batch_idx)` (Abstract)

Executes one validation step. Must be implemented by subclasses.

**Parameters:**

- `batch`: A batch of data from the validation dataloader
- `batch_idx` (int): Index of the current batch

**Returns:**

- Loss tensor or metrics dictionary

---

### `test_step(batch, batch_idx, dataloader_idx=0)` (Abstract)

Executes one test step. Must be implemented by subclasses.

**Parameters:**

- `batch`: A batch of data from the test dataloader
- `batch_idx` (int): Index of the current batch
- `dataloader_idx` (int): Index of the dataloader (for multiple test datasets)

**Returns:**

- Metrics dictionary or None

---

### `predict_step(batch, batch_idx, dataloader_idx=0)` (Abstract)

Executes one prediction step. Must be implemented by subclasses.

**Parameters:**

- `batch`: A batch of data from the prediction dataloader
- `batch_idx` (int): Index of the current batch
- `dataloader_idx` (int): Index of the dataloader

**Returns:**

- Predictions dictionary

---

### `on_fit_start()`

Called at the beginning of training. Saves normalization statistics to disk.

**Behavior:**

- Creates a `stats` directory in the logging directory
- Saves human-readable normalization statistics to `normalization_stats.txt`
- Saves machine-loadable statistics to `normalizer_stats.pt` (PyTorch format)
- Only executes on rank 0 in distributed training (via `@rank_zero_only` decorator)

**Output Files:**

1. **`normalization_stats.txt`**: Human-readable text file with statistics for each dataset
2. **`normalizer_stats.pt`**: PyTorch file containing a dictionary keyed by network name

---

### `configure_optimizers()`

Configures the optimizer and learning rate scheduler.

**Optimizer:**

- **Type**: AdamW
- **Learning Rate**: From `args.optimizer.learning_rate`
- **Betas**: From `args.optimizer.beta1` and `args.optimizer.beta2`

**Scheduler:**

- **Type**: ReduceLROnPlateau
- **Mode**: Minimize
- **Factor**: From `args.optimizer.lr_decay`
- **Patience**: From `args.optimizer.lr_patience`
- **Monitored Metric**: "Validation loss"

**Returns:**

- Dictionary with optimizer and lr_scheduler configuration

## Usage

`BaseTask` is not used directly. Instead, create a subclass that implements all abstract methods:

```python
from gridfm_graphkit.tasks.base_task import BaseTask

class MyCustomTask(BaseTask):
    def __init__(self, args, data_normalizers):
        super().__init__(args, data_normalizers)
        # Initialize task-specific components

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, mask_dict):
        # Implement forward pass
        pass

    def training_step(self, batch):
        # Implement training logic
        pass

    def validation_step(self, batch, batch_idx):
        # Implement validation logic
        pass

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # Implement test logic
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Implement prediction logic
        pass
```

## Configuration Example

The base task uses the following configuration sections:

```yaml
optimizer:
  learning_rate: 0.001
  beta1: 0.9
  beta2: 0.999
  lr_decay: 0.7
  lr_patience: 5

data:
  networks:
    - case14_ieee
    - case118_ieee
```

## Related

- [Reconstruction Task](reconstruction_task.md): Base class for reconstruction tasks
- [Power Flow Task](power_flow.md): Concrete implementation for power flow
- [Optimal Power Flow Task](optimal_power_flow.md): Concrete implementation for OPF
- [State Estimation Task](state_estimation.md): Concrete implementation for state estimation
