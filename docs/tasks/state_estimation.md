# State Estimation Task

The State Estimation task focuses on estimating the true state of a power grid from noisy measurements. This is a critical problem in power system operations, where sensor measurements may contain errors, outliers, or missing data.

## Overview

State estimation in power systems involves determining voltage magnitudes, voltage angles, and power injections at buses from available measurements. The `StateEstimationTask` extends the `ReconstructionTask` to handle:

- **Noisy measurements**: Input features include measurement noise and potential outliers
- **Missing data**: Some measurements may be masked or unavailable
- **Outlier detection**: The task tracks and evaluates performance on outlier measurements separately

## Key Features

- **Measurement-based prediction**: Estimates true grid state from noisy sensor data
- **Outlier handling**: Distinguishes between normal measurements, masked values, and outliers
- **Correlation analysis**: Generates plots comparing predictions vs. targets and predictions vs. measurements
- **Multi-mask evaluation**: Evaluates performance separately for outliers, masked values, and clean measurements

## StateEstimationTask

::: gridfm_graphkit.tasks.se_task.StateEstimationTask

## Metrics

The State Estimation task computes and logs the following metrics during testing:

### Prediction Quality
- **Voltage Magnitude (Vm)**: Accuracy of estimated voltage magnitudes at buses
- **Voltage Angle (Va)**: Accuracy of estimated voltage angles at buses  
- **Active Power Injection (Pg)**: Accuracy of estimated active power at buses
- **Reactive Power Injection (Qg)**: Accuracy of estimated reactive power at buses

### Evaluation Categories
Metrics are computed separately for three categories:
- **Outliers**: Measurements identified as outliers
- **Masked**: Intentionally masked/missing measurements
- **Non-outliers**: Clean measurements without outliers or masking

## Visualization

When `verbose=True` in the configuration, the task generates correlation plots:

1. **Predictions vs. Targets**: Shows how well predictions match ground truth
2. **Predictions vs. Measurements**: Shows how predictions compare to noisy input measurements
3. **Measurements vs. Targets**: Shows the quality of input measurements

These plots are generated for each feature (Vm, Va, Pg, Qg) and saved to the test artifacts directory.

## Configuration Example

```yaml
task:
  name: StateEstimation
  verbose: true

training:
  batch_size: 32
  epochs: 100
  losses: ["MaskedMSE"]
  loss_weights: [1.0]
```

## Usage

The State Estimation task is automatically selected when you specify `task.name: StateEstimation` in your YAML configuration file. The task handles:

1. Forward pass through the model with masked/noisy inputs
2. Inverse normalization of predictions and targets
3. Computation of metrics for different measurement categories
4. Generation of correlation plots and analysis

## Related

- [Base Task](base_task.md): Abstract base class for all tasks
- [Reconstruction Task](reconstruction_task.md): Base class for reconstruction tasks
- [Power Flow Task](power_flow.md): For standard power flow analysis
- [Optimal Power Flow Task](optimal_power_flow.md): For optimization-based power flow
- [Task Overview](feature_reconstruction.md): Overview of all task classes