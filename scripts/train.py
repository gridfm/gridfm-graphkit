from gridFM.models.graphTransformer import GraphTransformer
from gridFM.datasets.powergrid import GridDataset
from gridFM.io.param_handler import parse_yaml, load_normalizer
from gridFM.utils.loss import masked_loss
from gridFM.datasets.data_normalization import *
from gridFM.datasets.globals import *
from train_post_processing import *
import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import os
import mlflow
from datetime import datetime
import mlflow.pytorch
import argparse
from gridFM.training.trainer import Trainer
from gridFM.training.plugins import MLflowLoggerPlugin
from gridFM.datasets.utils import split_dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import shutil
import yaml


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Log directories
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("../runs", run_name)
    config_dir = os.path.join(run_dir, "config")
    data_dir = os.path.join(run_dir, "data_idx")
    test_dir = os.path.join(run_dir, "test")

    # Create log directories if they don't exists
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Load and parse config file
    config_dest = os.path.join(config_dir, "config.yaml")
    shutil.copyfile(args.config, config_dest)
    args = parse_yaml(args.config)

    # Create normalizer
    node_normalizer, edge_normalizer = load_normalizer(args=args)

    # Create torch dataset and split
    data_path = os.path.join(os.getcwd(), "..", "data", args.network)
    dataset = GridDataset(
        root=data_path,
        scenarios=args.data.scenarios,
        norm_method=args.data.normalization,
        node_normalizer=node_normalizer,
        edge_normalizer=edge_normalizer,
        mask_ratio=args.data.mask_ratio,
        mask_dim=args.data.mask_dim,
        mask_value=args.data.mask_value
    )

    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset, data_dir, args.data_split.val_ratio, args.data_split.test_ratio
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.training.batch_size, shuffle=False
    )

    # Create model
    model = GraphTransformer(
        input_dim=args.data.input_dim,
        hidden_dim=args.training.hidden_size,
        output_dim=args.data.output_dim,
        edge_dim=args.data.edge_dim,
        heads=args.training.attention_head,
        num_layers=args.training.num_layers,
    ).to(device)

    # Optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.optimizer.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.optimizer.lr_decay,
        patience=args.optimizer.lr_patience,
    )

    # Initialize mlflow
    mlflow.set_tracking_uri("file:../mlruns")
    experiment_name = f"exp_{run_name}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        run_metadata = {
            "experiment_id": experiment_id,
            "run_id": run.info.run_id,
        }

        # Dump mlflow metadata
        metadata_path = os.path.join(config_dir, "mlflow_metadata.yaml")
        with open(metadata_path, "w") as f:
            yaml.dump(run_metadata, f)

        mlflow_plugin = MLflowLoggerPlugin(steps=10, params=args.flatten())
        trainer = Trainer(
            model,
            optimizer,
            device,
            masked_loss,
            train_loader,
            val_loader,
            lr_scheduler=scheduler,
            plugins=[mlflow_plugin],
        )
        trainer.train(0, args.training.epochs)
        torch.save(model, os.path.join(run_dir, "model.pth"))
    # Initialize lists to store losses for each node type
    RMSE_loss_PQ = []
    RMSE_loss_PV = []
    RMSE_loss_REF = []
    MAE_loss_PQ = []
    MAE_loss_PV = []
    MAE_loss_REF = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Prepare input features with the same masking logic as in training
            input_features = torch.cat((batch.y, batch.x[:, 6:]), dim=1)
            
            # Masking logic with random values
            mask_PQ = input_features[:, PQ] == 1
            mask_PV = input_features[:, PV] == 1
            mask_REF = input_features[:, REF] == 1

            input_features[mask_PQ, VM] = args.data.mask_value
            input_features[mask_PQ, VA] = args.data.mask_value

            input_features[mask_PV, QG] = args.data.mask_value
            input_features[mask_PV, VA] = args.data.mask_value

            input_features[mask_REF, PG] = args.data.mask_value
            input_features[mask_REF, QG] = args.data.mask_value


            # Forward pass
            output = model(input_features, batch.edge_index, batch.edge_attr)
            
            # Denormalize the output and target
            output_denorm = node_normalizer.inverse_transform(output.cpu())
            target_denorm = node_normalizer.inverse_transform(batch.y.cpu())
            
            # Compute per-feature RMSE and MAE for each node type
            if mask_PQ.any():  # Check if any nodes of type PQ exist in this batch
                RMSE_loss_PQ.append(F.mse_loss(output_denorm[mask_PQ.cpu()], target_denorm[mask_PQ.cpu()], reduction='none'))
                MAE_loss_PQ.append(torch.abs(output_denorm[mask_PQ.cpu()] - target_denorm[mask_PQ.cpu()]))

            if mask_PV.any():  # Check if any nodes of type PV exist in this batch
                RMSE_loss_PV.append(F.mse_loss(output_denorm[mask_PV.cpu()], target_denorm[mask_PV.cpu()], reduction='none'))
                MAE_loss_PV.append(torch.abs(output_denorm[mask_PV.cpu()] - target_denorm[mask_PV.cpu()]))

            if mask_REF.any():  # Check if any nodes of type REF exist in this batch
                RMSE_loss_REF.append(F.mse_loss(output_denorm[mask_REF.cpu()], target_denorm[mask_REF.cpu()], reduction='none'))
                MAE_loss_REF.append(torch.abs(output_denorm[mask_REF.cpu()] - target_denorm[mask_REF.cpu()]))
    df = postprocess_results(RMSE_loss_PQ,RMSE_loss_PV,RMSE_loss_REF,MAE_loss_PQ,MAE_loss_PV,MAE_loss_REF)
    df_path = os.path.join(test_dir, "metrics_results.csv")
    df.to_csv(df_path)
    log_file_path = os.path.join(test_dir, "stats.log")

    # Write the print statements to the log file
    with open(log_file_path, "w") as log_file:
        log_file.write("Dataset node_stats: " + str(dataset.node_stats) + "\n")
        log_file.write("Dataset edge_stats: " + str(dataset.edge_stats) + "\n")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--config", type=str, default="config/default.yaml", help="Path to config file"
    )
    args = parser.parse_args()
    main(args)
