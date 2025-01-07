from gridFM.models.graphTransformer import GraphTransformer
from gridFM.datasets.powergrid import GridDatasetMem
from gridFM.io.param_handler import *
from gridFM.datasets.data_normalization import *
from gridFM.datasets.globals import *
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
from gridFM.training.callbacks import EarlyStopper
from gridFM.datasets.utils import split_dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset
import yaml
import random
from gridFM.evaluation.node_level import eval_node_level_task
import plotly.io as pio
import warnings
from torch.utils.data import Subset


def run_training(config_path, grid_params, device):

    with mlflow.start_run() as run:

        # Define log directories
        artifact_dir = os.path.join(
            "../mlruns", run.info.experiment_id, run.info.run_id, "artifacts"
        )
        config_dir = os.path.join(artifact_dir, "config")
        model_dir = os.path.join(artifact_dir, "model")
        data_dir = os.path.join(artifact_dir, "data_idx")
        test_dir = os.path.join(artifact_dir, "test")

        # Create log directories if they don't exists
        os.makedirs(artifact_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Load the base config
        with open(config_path, "r") as f:
            base_config = yaml.safe_load(f)

        # Deep merge the base config with grid parameters
        merge_dict(base_config, grid_params)

        # Save updated config file
        config_dest = os.path.join(config_dir, "config.yaml")
        with open(config_dest, "w") as f:
            yaml.dump(base_config, f)

        args = NestedNamespace(**base_config)

        # Fix random seed
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        node_normalizers = []
        edge_normalizers = []
        datasets = []
        train_datasets = []
        val_datasets = []
        test_datasets = []

        for i, network in enumerate(args.networks):
            node_normalizer, edge_normalizer = load_normalizer(args=args)
            node_normalizers.append(node_normalizer)
            edge_normalizers.append(edge_normalizer)

            # Create torch dataset and split
            data_path = os.path.join(os.getcwd(), "..", "data", network)
            dataset = GridDatasetMem(
                root=data_path,
                norm_method=args.data.normalization,
                node_normalizer=node_normalizer,
                edge_normalizer=edge_normalizer,
                mask_ratio=args.data.mask_ratio,
                mask_dim=args.data.mask_dim,
            )
            datasets.append(dataset)
            
            num_scenarios = args.data.scenarios[i]
            if num_scenarios > len(dataset):
                warnings.warn(
                    f"Requested number of scenarios ({num_scenarios}) exceeds dataset size ({len(dataset)}). "
                    "Using the full dataset instead."
                )
                num_scenarios = len(dataset)

            # Create a subset
            subset_indices = list(range(num_scenarios))
            dataset = Subset(dataset, subset_indices)

            train_dataset, val_dataset, test_dataset = split_dataset(
                dataset, data_dir, args.data_split.val_ratio, args.data_split.test_ratio
            )

            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            test_datasets.append(test_dataset)

        train_dataset_multi = ConcatDataset(train_datasets)
        val_dataset_multi = ConcatDataset(val_datasets)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset_multi, batch_size=args.training.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset_multi, batch_size=args.training.batch_size, shuffle=False
        )
        test_loaders = [DataLoader(i, batch_size=args.training.batch_size, shuffle=False) for i in test_datasets]


        # Create model
        model = GraphTransformer(
            input_dim=args.data.input_dim,
            hidden_dim=args.training.hidden_size,
            output_dim=args.data.output_dim,
            edge_dim=args.data.edge_dim,
            heads=args.training.attention_head,
            num_layers=args.training.num_layers,
            mask_dim=args.data.mask_dim,
            mask_value=args.data.mask_value,
            learn_mask=args.data.learn_mask,
        ).to(device)

        # Optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.optimizer.learning_rate,
            betas=(args.optimizer.beta1, args.optimizer.beta2),
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.optimizer.lr_decay,
            patience=args.optimizer.lr_patience,
        )

        best_model_path = os.path.join(model_dir, "best_model.pth")
        early_stopper = EarlyStopper(
            best_model_path, args.callbacks.patience, args.callbacks.tol
        )

        for i in range(len(node_normalizers)):
            node_normalizers[i].to(device)
            edge_normalizers[i].to(device)

        loss_fn = get_loss_function(args)

        mlflow_plugin = MLflowLoggerPlugin(steps=10, params=args.flatten())
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=device,
            loss_fn=loss_fn,
            early_stopper=early_stopper,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            lr_scheduler=scheduler,
            plugins=[mlflow_plugin],
        )
        # Train model
        trainer.train(0, args.training.epochs)

        # Save model
        model_path = os.path.join(model_dir, "model.pth")
        torch.save(model, model_path)

        # Save mask
        if args.data.learn_mask:
            mask_path = os.path.join(model_dir, "mask_value.txt")
            np.savetxt(mask_path, model.mask_value.numpy(force=True))

        # load best_model
        best_model = torch.load(best_model_path, weights_only=False)

        # Save best_mask
        if args.data.learn_mask:
            best_mask_path = os.path.join(model_dir, "best_mask_value.txt")
            np.savetxt(best_mask_path, best_model.mask_value.numpy(force=True))

        for i, network in enumerate(args.networks):
            for task in ["PF", "OPF"]:
                df, figs = eval_node_level_task(
                    best_model,
                    task,
                    test_loaders[i],
                    args.data.mask_dim,
                    node_normalizers[i],
                    device,
                    args.verbose
                )
                # Log metric results
                df_path = os.path.join(test_dir, f"{task}_metrics_results_{network}.csv")
                df.to_csv(df_path)

                plot_paths = os.path.join(test_dir, f"{task}_evaluation_plots_{network}.html")
                with open(plot_paths, "a") as f:
                    for fig in figs:
                        f.write(pio.to_html(fig, full_html=False, include_plotlyjs="cdn"))

            # Log node and edge stats
            log_file_path = os.path.join(artifact_dir, f"stats_{network}.log")
            with open(log_file_path, "w") as log_file:
                log_file.write("Dataset node_stats: " + str(datasets[i].node_stats) + "\n")
                log_file.write("Dataset edge_stats: " + str(datasets[i].edge_stats) + "\n")

        eval_cmd_path = os.path.join(artifact_dir, "EVAL_CMD.txt")
        with open(eval_cmd_path, "w") as f:
            f.write(
                f"python3 eval.py --model_exp_id {run.info.experiment_id} --model_run_id {run.info.run_id} --model_name best_model --config {config_dest} --eval_name YOUR_EVAL_NAME \n"
            )
            f.write(
                f"python eval.py --model_exp_id {run.info.experiment_id} --model_run_id {run.info.run_id} --model_name best_model --config {config_dest} --eval_name YOUR_EVAL_NAME"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script with grid search")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to base config file",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default=None,
        help="Path to grid.yaml file. No grid search by default",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        help="Experiment name for mlflow, None by default",
    )
    args = parser.parse_args()

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize mlflow
    mlflow.set_tracking_uri("file:../mlruns")

    exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.exp is None:
        experiment_name = f"exp_{exp_name}"
    else:
        experiment_name = f"{args.exp}"

    mlflow.set_experiment(experiment_name)

    if args.grid:  # Only perform grid search if grid parameter file is provided
        print(f"Grid search enabled. Using grid file: {args.grid}")

        # Parse grid parameters
        with open(args.grid, "r") as f:
            grid_config = yaml.safe_load(f)

        grid_combinations = param_combination_gen(grid_config)

        # Run experiments for all combinations
        for i, grid_params in enumerate(grid_combinations):
            print(
                f"\nGrid search: {i + 1}/{len(grid_combinations)} with params: {grid_params}"
            )
            run_training(args.config, grid_params, device)
    else:
        print("No grid search config file provided. Running single training")
        run_training(args.config, {}, device)
