from gridfm_graphkit.datasets.globals import BUS_TYPES, FEATURES_IDX, PQ, PV, REF
from gridfm_graphkit.datasets.data_normalization import BaseMVANormalizer
from gridfm_graphkit.datasets.transforms import AddRandomMask, AddPFMask, AddOPFMask
from gridfm_graphkit.utils.loss import PBELoss

import torch
import numpy as np
import pandas as pd
from typing import List, Tuple
from torch.utils.data import DataLoader
import plotly.graph_objects as go
from torch_geometric.data import Dataset


def get_dist_plot(
    data: np.ndarray,
    data_type: str,
    bus_types: List[str],
    n_buses: int,
) -> go.Figure:
    """
    Generates distribution plots for the different feature and for each bus.

    Args:
        data (np.ndarray): The input data matrix, e.g. residuals or model outputs, of shape (n_buses x len(test_dataset), n_features)
        data_type (str): The type of data being plotted (e.g., 'residuals', 'model outputs').
        bus_types (List[str]): List of bus types for each bus in the graphs
        n_buses (int): The total number of buses in the grid.

    Returns:
        List[go.Figure]: List of Plotly figures, each representing box plots of the distribution of one feature for each of the buses
    """

    figs = []

    for feature, feature_idx in FEATURES_IDX.items():
        fig = go.Figure()

        for bus_idx in range(n_buses):
            # Add box plot of distribution of feature for each bus
            fig.add_trace(
                go.Box(
                    y=data[
                        bus_idx::n_buses,
                        feature_idx,
                    ],  # Slice data for each bus (!!)
                    name=f"Bus {bus_idx} ({bus_types[bus_idx]})",
                ),
            )

        fig.update_layout(
            title="{} {} distribution".format(feature, data_type),
            xaxis_title="Bus Number",
            yaxis_title="{}".format(data_type),
            showlegend=True,
        )
        figs.append(fig)
    return figs


def training_stats_to_dataframe(
    rmse_PQ: np.ndarray,
    rmse_PV: np.ndarray,
    rmse_REF: np.ndarray,
    mae_PQ: np.ndarray,
    mae_PV: np.ndarray,
    mae_REF: np.ndarray,
    overall_RMSE: np.ndarray,
    overall_MAE: np.ndarray,
    overall_active_loss: float,
    overall_reactive_loss: float,
) -> pd.DataFrame:
    """
    Converts training statistics into a pandas DataFrame.

    Args:
        RMSE_loss_PQ (np.ndarray): RMSE losses for each feature at PQ nodes
        RMSE_loss_PV (np.ndarray): RMSE losses for each feature at PV nodes.
        RMSE_loss_REF (np.ndarray): RMSE losses  for each feature at REF nodes.
        MAE_loss_PQ (np.ndarray): MAE losses for each feature at PQ nodes.
        MAE_loss_PV (np.ndarray): MAE losses for each feature at PV nodes.
        MAE_loss_REF (np.ndarray): MAE losses for each feature at REF nodes.
        overall_active_loss (float): Mean active power loss across nodes
        overall_reactive_loss (float): Mean reactive power loss across nodes

    Returns:
        pd.DataFrame: DataFrame containing aggregated statistics.
    """

    data = {
        "Metric": [
            "RMSE-PQ",
            "RMSE-PV",
            "RMSE-REF",
            "MAE-PQ",
            "MAE-PV",
            "MAE-REF",
            "Overall RMSE",
            "Overall MAE",
            "Avg. active res. (MW)",
            "Avg. reactive res. (MVar)",
        ],
        "Pd (MW)": [
            rmse_PQ[0],
            rmse_PV[0],
            rmse_REF[0],
            mae_PQ[0],
            mae_PV[0],
            mae_REF[0],
            overall_RMSE[0],
            overall_MAE[0],
            overall_active_loss,
            overall_reactive_loss,
        ],
        "Qd (MVar)": [
            rmse_PQ[1],
            rmse_PV[1],
            rmse_REF[1],
            mae_PQ[1],
            mae_PV[1],
            mae_REF[1],
            overall_RMSE[1],
            overall_MAE[1],
            " ",
            " ",
        ],
        "Pg (MW)": [
            rmse_PQ[2],
            rmse_PV[2],
            rmse_REF[2],
            mae_PQ[2],
            mae_PV[2],
            mae_REF[2],
            overall_RMSE[2],
            overall_MAE[2],
            " ",
            " ",
        ],
        "Qg (MVar)": [
            rmse_PQ[3],
            rmse_PV[3],
            rmse_REF[3],
            mae_PQ[3],
            mae_PV[3],
            mae_REF[3],
            overall_RMSE[3],
            overall_MAE[3],
            " ",
            " ",
        ],
        "Vm (p.u.)": [
            rmse_PQ[4],
            rmse_PV[4],
            rmse_REF[4],
            mae_PQ[4],
            mae_PV[4],
            mae_REF[4],
            overall_RMSE[4],
            overall_MAE[4],
            " ",
            " ",
        ],
        "Va (degree)": [
            rmse_PQ[5],
            rmse_PV[5],
            rmse_REF[5],
            mae_PQ[5],
            mae_PV[5],
            mae_REF[5],
            overall_RMSE[5],
            overall_MAE[5],
            " ",
            " ",
        ],
    }
    return pd.DataFrame(data)


def eval_node_level_task(
    dataset: Dataset,
    model: torch.nn.Module,
    task: str,
    test_loader: DataLoader,
    mask_dim: int,
    mask_ratio: float,
    node_normalizer: object,
    device: torch.device,
    plot_dist: bool = True,
) -> Tuple[pd.DataFrame, List[go.Figure]]:
    """
    Evaluates the model and computes per-feature statistics.

    Args:
        model (torch.nn.Module): The trained model.
        task: task to evaluate the model on e.g. PF
        test_loader (DataLoader): DataLoader for test data.
        mask_dim (int): number of masked features
        node_normalizer (object): Normalizer for input/output features.
        plot_dist (bool): Whether to generate distribution plots.
        device (torch.device): Device to run the evaluation on.

    Returns:
        Tuple[pd.DataFrame, List[px.Figure]]: DataFrame with evaluation metrics and plotly figure.
    """
    model.eval()

    # Initialize lists to collect outputs and targets
    all_outputs = []
    all_targets = []
    all_mask_PQ = []
    all_mask_PV = []
    all_mask_REF = []
    all_active_loss = []
    all_reactive_loss = []

    loss_PBE = PBELoss()

    # Mask input features
    if task == "PF":
        dataset.change_transform(AddPFMask())
    elif task == "OPF":
        dataset.change_transform(AddOPFMask())
    elif task == "Reconstruction":
        dataset.change_transform(
            AddRandomMask(mask_dim=mask_dim, mask_ratio=mask_ratio),
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            mask_PQ = batch.x[:, PQ] == 1
            mask_PV = batch.x[:, PV] == 1
            mask_REF = batch.x[:, REF] == 1

            mask_value_expanded = model.mask_value.expand(batch.x.shape[0], -1)
            batch.x[:, : batch.mask.shape[1]][batch.mask] = mask_value_expanded[
                batch.mask
            ]

            # Forward pass
            output = model(
                batch.x,
                batch.pe,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
            )

            if isinstance(node_normalizer, BaseMVANormalizer):
                loss_PBE_dict = loss_PBE(
                    output,
                    batch.y,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.mask,
                )
                all_active_loss.append(
                    loss_PBE_dict["Active Power Loss in p.u."]
                    * node_normalizer.baseMVA,
                )
                all_reactive_loss.append(
                    loss_PBE_dict["Reactive Power Loss in p.u."]
                    * node_normalizer.baseMVA,
                )
            else:
                all_active_loss.append(-1.0)
                all_reactive_loss.append(-1.0)

            # Denormalize
            output_denorm = node_normalizer.inverse_transform(output)
            target_denorm = node_normalizer.inverse_transform(batch.y)

            # Collect outputs, targets, and masks
            all_outputs.append(output_denorm)
            all_targets.append(target_denorm)
            all_mask_PQ.append(mask_PQ)
            all_mask_PV.append(mask_PV)
            all_mask_REF.append(mask_REF)

    n_buses = int((batch.batch == 0).sum())  # Number of buses in graph
    bus_types = [
        BUS_TYPES[np.argmax(row[mask_dim:])] for row in batch.x[:n_buses].cpu()
    ]  # Ugly hack to get bus types from input features

    # Concatenate all outputs, targets, and masks
    all_outputs = torch.cat(all_outputs, dim=0).cpu()
    all_targets = torch.cat(all_targets, dim=0).cpu()
    all_mask_PQ = torch.cat(all_mask_PQ, dim=0).cpu()
    all_mask_PV = torch.cat(all_mask_PV, dim=0).cpu()
    all_mask_REF = torch.cat(all_mask_REF, dim=0).cpu()

    # Compute per-feature RMSE and MAE after collecting all batches
    residuals = (all_outputs - all_targets).numpy()
    squared_residuals = residuals**2
    absolute_residuals = np.abs(residuals)

    rmse_PQ = np.sqrt(np.mean(squared_residuals[all_mask_PQ], axis=0))
    rmse_PV = np.sqrt(np.mean(squared_residuals[all_mask_PV], axis=0))
    rmse_REF = np.sqrt(np.mean(squared_residuals[all_mask_REF], axis=0))

    mae_PQ = np.mean(absolute_residuals[all_mask_PQ], axis=0)
    mae_PV = np.mean(absolute_residuals[all_mask_PV], axis=0)
    mae_REF = np.mean(absolute_residuals[all_mask_REF], axis=0)

    overall_RMSE = np.sqrt(np.mean(squared_residuals, axis=0))
    overall_MAE = np.mean(absolute_residuals, axis=0)

    overall_active_loss = np.mean(all_active_loss)
    overall_reactive_loss = np.mean(all_reactive_loss)

    figs = []

    if plot_dist:
        figs.extend(get_dist_plot(residuals, "residuals", bus_types, n_buses))
        figs.extend(get_dist_plot(all_outputs, "model outputs", bus_types, n_buses))

    # Convert statistics to a DataFrame
    df = training_stats_to_dataframe(
        rmse_PQ,
        rmse_PV,
        rmse_REF,
        mae_PQ,
        mae_PV,
        mae_REF,
        overall_RMSE,
        overall_MAE,
        overall_active_loss,
        overall_reactive_loss,
    )
    dataset.reset_transform()
    return df, figs
