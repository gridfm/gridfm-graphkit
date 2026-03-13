# Optimal Power Flow Task

The Optimal Power Flow (OPF) task solves the optimization problem of determining the most economical operation of a power system while satisfying physical and operational constraints. This task predicts optimal generator setpoints, voltage profiles, and reactive power dispatch.

## Overview

Optimal Power Flow is a fundamental optimization problem in power systems that minimizes generation costs while ensuring:

- **Power balance**: Supply meets demand at all buses
- **Voltage constraints**: Bus voltages remain within acceptable limits
- **Thermal limits**: Branch flows don't exceed capacity
- **Generator limits**: Active and reactive power generation within bounds
- **Angle difference limits**: Voltage angle differences across branches are acceptable

The `OptimalPowerFlowTask` extends the `ReconstructionTask` to include OPF-specific physics-based constraints and economic metrics.

## Key Features

- **Economic optimization**: Tracks generation costs and optimality gap
- **Constraint violation monitoring**: Measures violations of thermal, voltage, and angle limits
- **Physics-based evaluation**: Computes power balance errors and residuals
- **Bus type differentiation**: Separate metrics for PQ, PV, and REF buses
- **Comprehensive reporting**: Generates detailed CSV reports and correlation plots

## OptimalPowerFlowTask

::: gridfm_graphkit.tasks.opf_task.OptimalPowerFlowTask

## Metrics

The Optimal Power Flow task computes extensive metrics during testing:

### Economic Metrics
- **Optimality Gap (%)**: Percentage difference between predicted and optimal generation costs
- **Generation Cost**: Total cost computed from quadratic cost curves (c₀ + c₁·Pg + c₂·Pg²)

### Power Balance Metrics
- **Active Power Loss (MW)**: Mean absolute active power residual across all buses
- **Reactive Power Loss (MVar)**: Mean absolute reactive power residual across all buses

### Constraint Violations
- **Branch Thermal Violations (MVA)**: 
  - Forward direction: Mean excess flow above thermal limits
  - Reverse direction: Mean excess flow above thermal limits
- **Branch Angle Violations (radians)**: Mean violation of angle difference constraints
- **Reactive Power Violations**:
  - PV buses: Mean Qg violation (exceeding min/max limits)
  - REF buses: Mean Qg violation (exceeding min/max limits)

### Prediction Accuracy (RMSE)
Computed separately for each bus type (PQ, PV, REF):
- **Voltage Magnitude (Vm)**: p.u.
- **Voltage Angle (Va)**: radians
- **Active Power Generation (Pg)**: MW
- **Reactive Power Generation (Qg)**: MVar

### Residual Statistics (when verbose=True)
For each bus type and power type (P, Q):
- Mean residual per graph
- Maximum residual per graph

## Bus Types

The task evaluates performance separately for three bus types:

- **PQ Buses**: Load buses with specified active and reactive power demand
- **PV Buses**: Generator buses with specified active power and voltage magnitude
- **REF Buses**: Reference/slack buses that balance the system

## Outputs

### CSV Reports
Two CSV files are generated per test dataset:

1. **`{dataset}_RMSE.csv`**: RMSE metrics by bus type
   - Columns: Metric, Pg (MW), Qg (MVar), Vm (p.u.), Va (radians)
   - Rows: RMSE-PQ, RMSE-PV, RMSE-REF

2. **`{dataset}_metrics.csv`**: Comprehensive metrics including:
   - Average active/reactive residuals
   - RMSE for generator active power
   - Mean optimality gap
   - Branch thermal violations (from/to)
   - Branch angle difference violations
   - Qg violations for PV and REF buses

### Visualizations (when verbose=True)

1. **Cost Correlation Plot**: Predicted vs. ground truth generation costs with correlation coefficient
2. **Residual Histograms**: Distribution of power balance residuals by bus type
3. **Feature Correlation Plots**: Predictions vs. targets for Vm, Va, Pg, Qg by bus type, including Qg violation highlighting

## Configuration Example

```yaml
task:
  name: OptimalPowerFlow
  verbose: true

training:
  batch_size: 32
  epochs: 100
  losses: ["MaskedMSE", "PBE"]
  loss_weights: [0.01, 0.99]

optimizer:
  name: Adam
  lr: 0.001
```

## Physics-Based Constraints

The task uses specialized layers to compute physical quantities:

- **`ComputeBranchFlow`**: Calculates active (Pft) and reactive (Qft) power flows on branches
- **`ComputeNodeInjection`**: Aggregates branch flows to compute net injections at buses
- **`ComputeNodeResiduals`**: Computes power balance violations (residuals)

These ensure predictions are evaluated not just on accuracy but also on physical feasibility.

## Usage

The Optimal Power Flow task is automatically selected when you specify `task.name: OptimalPowerFlow` in your YAML configuration file. The task:

1. Performs forward pass through the model
2. Inverse normalizes predictions and targets
3. Computes branch flows and power balance residuals
4. Evaluates constraint violations
5. Calculates economic metrics (costs, optimality gap)
6. Generates comprehensive reports and visualizations

## Related

- [Power Flow Task](power_flow.md): For standard power flow analysis without optimization
- [State Estimation Task](state_estimation.md): For state estimation from measurements
- [Feature Reconstruction](feature_reconstruction.md): Base reconstruction task