# Power Flow Task

The Power Flow task solves the fundamental problem of determining the steady-state operating conditions of a power system. Given load demands and generator setpoints, it computes voltage magnitudes, voltage angles, and power flows throughout the network.

## Overview

Power Flow (also known as Load Flow) analysis is essential for power system planning and operation. It determines:

- **Voltage profiles**: Magnitude and angle at each bus
- **Power flows**: Active and reactive power on transmission lines
- **Power injections**: Net power generation/consumption at buses
- **System losses**: Total active and reactive power losses

The `PowerFlowTask` extends the `ReconstructionTask` to include physics-based power balance evaluation and comprehensive metrics for different bus types.

## Key Features

- **Physics-based validation**: Computes power balance errors (PBE) to verify physical consistency
- **Bus type differentiation**: Separate metrics for PQ, PV, and REF buses
- **Distributed training support**: Handles multi-GPU training with proper metric aggregation
- **Detailed predictions**: Provides per-bus predictions with residuals for analysis
- **Comprehensive reporting**: Generates CSV reports and correlation plots

## PowerFlowTask

::: gridfm_graphkit.tasks.pf_task.PowerFlowTask

## Metrics

The Power Flow task computes the following metrics during testing:

### Power Balance Metrics
- **Active Power Loss (MW)**: Mean absolute active power residual across all buses
- **Reactive Power Loss (MVar)**: Mean absolute reactive power residual across all buses
- **PBE Mean**: Mean Power Balance Error magnitude across all buses (√(P² + Q²))
- **PBE Max**: Maximum Power Balance Error across all buses

### Prediction Accuracy (RMSE)
Computed separately for each bus type (PQ, PV, REF):
- **Voltage Magnitude (Vm)**: p.u.
- **Voltage Angle (Va)**: radians
- **Active Power Generation (Pg)**: MW
- **Reactive Power Generation (Qg)**: MVar

### Residual Statistics (when verbose=True)
For each bus type (PQ, PV, REF) and power type (P, Q):
- Mean residual per graph
- Maximum residual per graph

## Bus Types

The task evaluates performance separately for three bus types:

- **PQ Buses**: Load buses with specified active and reactive power demand
- **PV Buses**: Generator buses with specified active power and voltage magnitude
- **REF Buses**: Reference/slack buses that balance the system

## Power Balance Error (PBE)

The Power Balance Error is a critical metric that measures how well predictions satisfy Kirchhoff's laws:

$$
\text{PBE} = \sqrt{(\Delta P)^2 + (\Delta Q)^2}
$$

where:
- $\Delta P$ = Active power residual (generation - demand - losses)
- $\Delta Q$ = Reactive power residual (generation - demand - losses)

Lower PBE values indicate better physical consistency of the predictions.

## Outputs

### CSV Reports
Two CSV files are generated per test dataset:

1. **`{dataset}_RMSE.csv`**: RMSE metrics by bus type
   - Columns: Metric, Pg (MW), Qg (MVar), Vm (p.u.), Va (radians)
   - Rows: RMSE-PQ, RMSE-PV, RMSE-REF

2. **`{dataset}_metrics.csv`**: Power balance metrics
   - Avg. active res. (MW)
   - Avg. reactive res. (MVar)
   - PBE Mean
   - PBE Max

### Visualizations (when verbose=True)

1. **Residual Histograms**: Distribution of power balance residuals by bus type (PQ, PV, REF)
2. **Feature Correlation Plots**: Predictions vs. targets for Vm, Va, Pg, Qg by bus type

### Prediction Output

The `predict_step` method returns detailed per-bus information:

```python
{
    'scenario': scenario IDs,
    'bus': bus indices,
    'pd_mw': active power demand,
    'qd_mvar': reactive power demand,
    'vm_pu_target': target voltage magnitude,
    'va_target': target voltage angle,
    'pg_mw_target': target active power generation,
    'qg_mvar_target': target reactive power generation,
    'is_pq': PQ bus indicator,
    'is_pv': PV bus indicator,
    'is_ref': REF bus indicator,
    'vm_pu': predicted voltage magnitude,
    'va': predicted voltage angle,
    'pg_mw': predicted active power generation,
    'qg_mvar': predicted reactive power generation,
    'active res. (MW)': active power residual,
    'reactive res. (MVar)': reactive power residual,
    'PBE': power balance error magnitude
}
```

## Configuration Example

```yaml
task:
  name: PowerFlow
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

- **`ComputeBranchFlow`**: Calculates active (Pft) and reactive (Qft) power flows on branches using the power flow equations
- **`ComputeNodeInjection`**: Aggregates branch flows to compute net power injections at each bus
- **`ComputeNodeResiduals`**: Computes power balance violations by comparing injections with generation and demand

These layers ensure that predictions are evaluated not only on accuracy but also on their adherence to fundamental power system physics.

## Distributed Training

The PowerFlowTask includes special handling for distributed training:

- **Metric aggregation**: Uses `sync_dist=True` to properly aggregate metrics across GPUs
- **Verbose output gathering**: Collects test outputs from all ranks to rank 0 for complete visualization
- **Max reduction for PBE Max**: Uses `reduce_fx="max"` to find the global maximum PBE across all processes

## Usage

The Power Flow task is automatically selected when you specify `task.name: PowerFlow` in your YAML configuration file. The task:

1. Performs forward pass through the model
2. Inverse normalizes predictions and targets
3. Computes branch flows using power flow equations
4. Calculates power balance residuals and PBE
5. Evaluates metrics separately for each bus type
6. Generates comprehensive reports and visualizations
7. Provides detailed per-bus predictions for analysis

## Related

- [Optimal Power Flow Task](optimal_power_flow.md): For optimization-based power flow with economic objectives
- [State Estimation Task](state_estimation.md): For state estimation from noisy measurements
- [Feature Reconstruction](feature_reconstruction.md): Base reconstruction task