# State Estimation Task

The `StateEstimationTask` class is a concrete implementation of `ReconstructionTask` for **state estimation** from noisy measurements. It evaluates predictions against ground truth and measurements, with separate handling for outliers, masked values, and clean measurements.

## Overview

`StateEstimationTask` extends `ReconstructionTask` and provides:

- **Measurement-based setup**: Inputs are (noisy) measurements; targets are true states. The model reconstructs the state from measurements.
- **Three-way evaluation**: Comparisons between predictions vs targets, predictions vs measurements, and measurements vs targets, with masks for outliers, masked (hidden) values, and non-outliers.
- **Correlation plots**: When `verbose: true`, scatter plots (pred vs target, pred vs measured, measured vs target) per feature (Vm, Va, Pg, Qg) are saved to MLflow artifacts.

## StateEstimationTask Class

::: gridfm_graphkit.tasks.se_task.StateEstimationTask
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - test_step
        - on_test_end
        - predict_step

## Configuration Example

Use the task by setting `task_name: StateEstimation` in your YAML:

```yaml
task:
  task_name: StateEstimation

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

data:
  networks:
    - case14_ieee
    - case118_ieee

verbose: true
```

## Test Outputs

- **test_step**: Runs the shared reconstruction step, then computes targets and measurements (Vm, Va, P_injection, Q_injection). Uses `mask_dict["outliers_bus"]`, `mask_dict["bus"]`, and non-outlier masks to separate evaluation groups. Stores predictions, targets, and measurements for `on_test_end`.
- **on_test_end**: If `verbose`, writes correlation plots (pred vs target, pred vs measured, measured vs target) per dataset to `artifacts/test_plots/<dataset_name>/`.
- **predict_step**: Currently a stub; override in a subclass or in a future release for custom prediction behavior.

## Related

- [Reconstruction Task](reconstruction_task.md): Base class for reconstruction tasks
- [Power Flow Task](power_flow.md): Power flow analysis
- [Optimal Power Flow Task](optimal_power_flow.md): OPF with cost and constraint metrics
- [Base Task](base_task.md): Abstract base class for all tasks
