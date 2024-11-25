import torch
import pandas as pd

def compute_mean_loss(loss_list):
    return torch.sqrt(torch.cat(loss_list, dim=0).mean(dim=0)) if loss_list else None

def training_stats_to_dataframe(RMSE_loss_PQ, RMSE_loss_PV, RMSE_loss_REF, MAE_loss_PQ, MAE_loss_PV, MAE_loss_REF):
    RMSE_pq = compute_mean_loss(RMSE_loss_PQ).tolist()
    RMSE_pv = compute_mean_loss(RMSE_loss_PV).tolist()
    RMSE_ref = compute_mean_loss(RMSE_loss_REF).tolist()

    MAE_pq = torch.cat(MAE_loss_PQ, dim=0).mean(dim=0).tolist() if MAE_loss_PQ else None
    MAE_pv = torch.cat(MAE_loss_PV, dim=0).mean(dim=0).tolist() if MAE_loss_PV else None
    MAE_ref = torch.cat(MAE_loss_REF, dim=0).mean(dim=0).tolist() if MAE_loss_REF else None

    overall_RMSE_loss = RMSE_loss_PQ + RMSE_loss_PV + RMSE_loss_REF
    overall_RMSE = compute_mean_loss(overall_RMSE_loss).tolist()

    # Calculate overall MAE by combining all individual losses across node types
    overall_MAE_loss = MAE_loss_PQ + MAE_loss_PV + MAE_loss_REF
    overall_MAE = torch.cat(overall_MAE_loss, dim=0).mean(dim=0).tolist() if overall_MAE_loss else None

    # Prepare the data in the desired format
    data = {
        'Metric': [
            'RMSE-PQ', 'RMSE-PV', 'RMSE-REF',
            'MAE-PQ', 'MAE-PV', 'MAE-REF',
            'Overall RMSE', 'Overall MAE',
        ],
        'Pd (MW)': [
            RMSE_pq[0], RMSE_pv[0], RMSE_ref[0],
            MAE_pq[0], MAE_pv[0], MAE_ref[0],
            overall_RMSE[0], overall_MAE[0],
        ],
        'Qd (MVar)': [
            RMSE_pq[1], RMSE_pv[1], RMSE_ref[1],
            MAE_pq[1], MAE_pv[1], MAE_ref[1],
            overall_RMSE[1], overall_MAE[1],
        ],
        'Pg (MW)': [
            RMSE_pq[2], RMSE_pv[2], RMSE_ref[2],
            MAE_pq[2], MAE_pv[2], MAE_ref[2],
            overall_RMSE[2], overall_MAE[2],
        ],
        'Qg (MVar)': [
            RMSE_pq[3], RMSE_pv[3], RMSE_ref[3],
            MAE_pq[3], MAE_pv[3], MAE_ref[3],
            overall_RMSE[3], overall_MAE[3],
        ],
        'Vm (p.u.)': [
            RMSE_pq[4], RMSE_pv[4], RMSE_ref[4],
            MAE_pq[4], MAE_pv[4], MAE_ref[4],
            overall_RMSE[4], overall_MAE[4],
        ],
        'Va (degree)': [
            RMSE_pq[5], RMSE_pv[5], RMSE_ref[5],
            MAE_pq[5], MAE_pv[5], MAE_ref[5],
            overall_RMSE[5], overall_MAE[5],
        ]
    }
    return pd.DataFrame(data)
