# Task Classes Overview

GridFM-GraphKit provides a hierarchical task system for power grid analysis. All tasks inherit from a common base class and share core functionality while implementing domain-specific logic.

## Task Hierarchy

```
BaseTask (Abstract)
    └── ReconstructionTask
            ├── PowerFlowTask
            ├── OptimalPowerFlowTask
            └── StateEstimationTask
```

## Available Task Classes

### Base Classes

- **[BaseTask](base_task.md)**: Abstract base class providing common functionality for all tasks
    - Optimizer configuration
    - Learning rate scheduling
    - Normalization statistics logging
    - Abstract method definitions

- **[ReconstructionTask](reconstruction_task.md)**: Base class for feature reconstruction tasks
    - Model integration
    - Loss function handling
    - Shared training/validation logic
    - Test output management

### Concrete Task Implementations

- **[PowerFlowTask](power_flow.md)**: Power flow analysis
    - Computes voltage profiles and power flows
    - Physics-based validation with Power Balance Error (PBE)
    - Separate metrics for PQ, PV, and REF buses
    - Detailed per-bus predictions

- **[OptimalPowerFlowTask](optimal_power_flow.md)**: Optimal power flow with economic optimization
    - Minimizes generation costs
    - Tracks optimality gap
    - Monitors constraint violations (thermal, voltage, angle)
    - Evaluates reactive power limits

- **[StateEstimationTask](state_estimation.md)**: State estimation from noisy measurements
    - Handles measurement noise and outliers
    - Separate evaluation for outliers, masked values, and clean measurements
    - Correlation analysis between predictions, measurements, and targets

## Quick Reference

### Method Overview

All task classes implement the following core methods:

| Method | Purpose | Implemented In |
|--------|---------|----------------|
| `__init__` | Initialize task with config and normalizers | All classes |
| `forward` | Forward pass through model | ReconstructionTask+ |
| `training_step` | Execute one training step | ReconstructionTask+ |
| `validation_step` | Execute one validation step | ReconstructionTask+ |
| `test_step` | Execute one test step | Concrete tasks |
| `predict_step` | Execute one prediction step | Concrete tasks |
| `on_fit_start` | Save normalization stats before training | BaseTask |
| `on_test_end` | Generate reports and plots after testing | Concrete tasks |
| `configure_optimizers` | Setup optimizer and scheduler | BaseTask |

### Task Selection

Tasks are automatically selected based on your YAML configuration:

```yaml
task:
  task_name: PowerFlow  # or OptimalPowerFlow, StateEstimation
```

The task registry automatically instantiates the correct task class based on the `task_name` field.

## Common Features

All tasks share these features:

### 1. Distributed Training Support
- Multi-GPU training with proper metric synchronization
- Rank 0 handles logging and file I/O
- Automatic gathering of test outputs across ranks

### 2. Comprehensive Logging
- Training and validation metrics logged to MLflow or TensorBoard
- Automatic hyperparameter tracking
- Normalization statistics saved for reproducibility

### 3. Test Outputs
- CSV reports with detailed metrics
- Visualization plots (when `verbose=True`)
- Per-dataset analysis for multiple test sets

### 4. Physics-Based Evaluation
- Power balance error computation
- Branch flow calculations
- Residual analysis by bus type

## Configuration

### Basic Configuration

```yaml
task:
  task_name: PowerFlow
  verbose: true

training:
  batch_size: 64
  epochs: 100
  losses: ["MaskedMSE", "PBE"]
  loss_weights: [0.01, 0.99]

optimizer:
  learning_rate: 0.001
  beta1: 0.9
  beta2: 0.999
  lr_decay: 0.7
  lr_patience: 5
```

### Task-Specific Options

Each task may have additional configuration options. See the individual task documentation for details:

- [Power Flow Configuration](power_flow.md#configuration-example)
- [Optimal Power Flow Configuration](optimal_power_flow.md#configuration-example)
- [State Estimation Configuration](state_estimation.md#configuration-example)

## Creating Custom Tasks

To create a custom task, extend `ReconstructionTask` or `BaseTask`:

```python
from gridfm_graphkit.tasks.reconstruction_tasks import ReconstructionTask
from gridfm_graphkit.io.registries import TASK_REGISTRY

@TASK_REGISTRY.register("MyCustomTask")
class MyCustomTask(ReconstructionTask):
    def __init__(self, args, data_normalizers):
        super().__init__(args, data_normalizers)
        # Add custom initialization
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # Implement custom test logic
        output, loss_dict = self.shared_step(batch)
        
        # Add custom metrics
        custom_metric = self.compute_custom_metric(output, batch)
        loss_dict["Custom Metric"] = custom_metric
        
        # Log metrics
        for metric, value in loss_dict.items():
            self.log(f"{dataset_name}/{metric}", value)
        
        return loss_dict["loss"]
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Implement custom prediction logic
        output, _ = self.shared_step(batch)
        return {"predictions": output}
    
    def on_test_end(self):
        # Custom analysis and visualization
        # Generate reports, plots, etc.
        super().on_test_end()
```

Then use it in your configuration:

```yaml
task:
  task_name: MyCustomTask
```

## Related Documentation

- [Loss Functions](../training/loss.md): Available loss functions and their configuration
- [Data Modules](../datasets/data_modules.md): Data loading and preprocessing
- [Models](../models/models.md): Available model architectures
- [Quick Start Guide](../quick_start/quick_start.md): Getting started with training
