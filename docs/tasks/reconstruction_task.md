# Reconstruction Task

The `ReconstructionTask` class is a concrete implementation of `BaseTask` that provides the foundation for node feature reconstruction on power grid graphs. It wraps a GridFM model and defines the training, validation, and testing logic for reconstructing masked node features.

## Overview

`ReconstructionTask` serves as the base class for all reconstruction-based tasks in GridFM-GraphKit, including:

- Power Flow (PF)
- Optimal Power Flow (OPF)
- State Estimation (SE)

It provides:

- **Model integration**: Loads and wraps the GridFM model
- **Loss function handling**: Configures and applies loss functions
- **Shared training logic**: Common training and validation steps
- **Test output management**: Collects and manages test outputs for analysis

## ReconstructionTask Class

::: gridfm_graphkit.tasks.reconstruction_tasks.ReconstructionTask
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - forward
        - shared_step
        - training_step
        - validation_step
        - on_test_end

## Methods

### `__init__(args, data_normalizers)`

Initialize the reconstruction task with model, loss function, and configuration.

**Parameters:**

- `args` (NestedNamespace): Experiment configuration with fields like:
    - `training.batch_size`: Batch size for training
    - `optimizer.*`: Optimizer configuration
    - `model.*`: Model architecture configuration
    - `training.losses`: List of loss functions to use
    - `data.networks`: List of network names
- `data_normalizers` (list): One normalizer per dataset for feature normalization/denormalization

**Attributes Set:**

- `self.model`: GridFM model loaded via `load_model()`
- `self.loss_fn`: Loss function resolved from configuration via `get_loss_function()`
- `self.batch_size`: Training batch size
- `self.test_outputs`: Dictionary to store test outputs per dataset (keyed by dataloader index)

**Example:**

```python
task = ReconstructionTask(args, data_normalizers)
```

---

### `forward(x_dict, edge_index_dict, edge_attr_dict, mask_dict)`

Forward pass through the model.

**Parameters:**

- `x_dict` (dict): Node features dictionary with keys like `"bus"`, `"gen"`
- `edge_index_dict` (dict): Edge indices dictionary for heterogeneous edges
- `edge_attr_dict` (dict): Edge attributes dictionary
- `mask_dict` (dict): Masking dictionary indicating which features are masked

**Returns:**

- Model output dictionary with predicted node features

**Example:**

```python
output = task.forward(
    x_dict=batch.x_dict,
    edge_index_dict=batch.edge_index_dict,
    edge_attr_dict=batch.edge_attr_dict,
    mask_dict=batch.mask_dict
)
```

---

### `shared_step(batch)`

Common logic for training and validation steps.

**Parameters:**

- `batch`: A batch from the dataloader containing:
    - `x_dict`: Input node features
    - `y_dict`: Target node features
    - `edge_index_dict`: Edge connectivity
    - `edge_attr_dict`: Edge attributes
    - `mask_dict`: Feature masks

**Returns:**

- `output` (dict): Model predictions
- `loss_dict` (dict): Dictionary containing:
    - `"loss"`: Total loss value
    - Additional loss components (if applicable)

**Behavior:**

1. Performs forward pass through the model
2. Computes loss using the configured loss function
3. Returns both predictions and loss dictionary

**Example:**

```python
output, loss_dict = task.shared_step(batch)
total_loss = loss_dict["loss"]
```

---

### `training_step(batch)`

Execute one training step.

**Parameters:**

- `batch`: Training batch from dataloader

**Returns:**

- Loss tensor for backpropagation

**Logged Metrics:**

- `"Training Loss"`: Total training loss
- `"Learning Rate"`: Current learning rate

**Logging Configuration:**

- `batch_size`: Number of graphs in batch
- `sync_dist=False`: No synchronization across GPUs during training
- `on_epoch=False`: Log per step, not per epoch
- `on_step=True`: Log at each training step
- `prog_bar=False`: Don't show in progress bar
- `logger=True`: Send to logger (e.g., MLflow)

---

### `validation_step(batch, batch_idx)`

Execute one validation step.

**Parameters:**

- `batch`: Validation batch from dataloader
- `batch_idx` (int): Index of the current batch

**Returns:**

- Loss tensor

**Logged Metrics:**

- `"Validation loss"`: Total validation loss
- Additional loss components (if multiple losses are used)

**Logging Configuration:**

- `batch_size`: Number of graphs in batch
- `sync_dist=True`: Synchronize metrics across GPUs
- `on_epoch=True`: Aggregate and log at epoch end
- `on_step=False`: Don't log individual steps
- `logger=True`: Send to logger

**Note:** The validation loss is monitored by the learning rate scheduler for automatic learning rate reduction.

---

### `on_test_end()`

Called at the end of testing. Clears stored test outputs.

**Behavior:**

- Clears the `self.test_outputs` dictionary
- Only executes on rank 0 in distributed training (via `@rank_zero_only` decorator)
- Subclasses typically override this to add custom analysis, plotting, and CSV generation

**Note:** This is a minimal implementation. Task-specific subclasses (PowerFlowTask, OptimalPowerFlowTask, StateEstimationTask) override this method to:

- Generate detailed metrics CSV files
- Create visualization plots
- Save analysis results

---

## Usage

`ReconstructionTask` can be used directly for simple reconstruction tasks, but is typically subclassed for specific power system tasks:

```python
from gridfm_graphkit.tasks.reconstruction_tasks import ReconstructionTask

# Direct usage (simple reconstruction)
task = ReconstructionTask(args, data_normalizers)

# Or create a subclass for custom behavior
class CustomReconstructionTask(ReconstructionTask):
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # Custom test logic
        output, loss_dict = self.shared_step(batch)
        # Add custom metrics
        return loss_dict["loss"]
    
    def on_test_end(self):
        # Custom analysis and visualization
        super().on_test_end()
```

## Configuration Example

```yaml
task:
  task_name: Reconstruction  # Or PowerFlow, OptimalPowerFlow, StateEstimation

model:
  type: GNS_heterogeneous
  hidden_size: 48
  num_layers: 12
  attention_head: 8

training:
  batch_size: 64
  epochs: 100
  losses:
    - MaskedMSE
  loss_weights:
    - 1.0

optimizer:
  learning_rate: 0.001
  beta1: 0.9
  beta2: 0.999
  lr_decay: 0.7
  lr_patience: 5
```

## Loss Functions

The reconstruction task supports various loss functions configured via the YAML file:

- **MaskedMSE**: Mean squared error on masked features only
- **MaskedBusMSE**: MSE specifically for bus node features
- **LayeredWeightedPhysics**: Physics-based loss with layer-wise weighting
- **PBE**: Power Balance Error loss

Multiple losses can be combined with weights:

```yaml
training:
  losses:
    - LayeredWeightedPhysics
    - MaskedBusMSE
  loss_weights:
    - 0.1
    - 0.9
  loss_args:
    - base_weight: 0.5
    - {}
```

## Subclasses

The following task classes extend `ReconstructionTask`:

- **[PowerFlowTask](power_flow.md)**: Adds power flow-specific metrics and physics validation
- **[OptimalPowerFlowTask](optimal_power_flow.md)**: Adds economic optimization metrics and constraint violation tracking
- **[StateEstimationTask](state_estimation.md)**: Adds measurement-based estimation and outlier handling

## Related

- [Base Task](base_task.md): Abstract base class for all tasks
- [Power Flow Task](power_flow.md): Power flow analysis implementation
- [Optimal Power Flow Task](optimal_power_flow.md): OPF optimization implementation
- [State Estimation Task](state_estimation.md): State estimation implementation
- [Loss Functions](../training/loss.md): Available loss functions