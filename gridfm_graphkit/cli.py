from gridfm_graphkit.datasets.powergrid import GridDatasetMem
from gridfm_graphkit.training.trainer import Trainer
from gridfm_graphkit.training.plugins import MLflowLoggerPlugin, CheckpointerPlugin
from gridfm_graphkit.training.callbacks import EarlyStopper
from gridfm_graphkit.datasets.utils import split_dataset
from gridfm_graphkit.evaluation.node_level import eval_node_level_task
from gridfm_graphkit.io.param_handler import (
    NestedNamespace,
    merge_dict,
    load_normalizer,
    load_model,
    get_loss_function,
    get_transform,
    param_combination_gen,
)

import torch
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset
import numpy as np
import os
import mlflow
from datetime import datetime
import yaml
import random
import plotly.io as pio
import warnings


def run_training(
    config_path,
    grid_params,
    data_path,
    device,
    run,
    checkpoint_flag,
    model_path=None,
):
    # Define log directories
    artifact_dir = os.path.join(
        "mlruns",
        run.info.experiment_id,
        run.info.run_id,
        "artifacts",
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
    if checkpoint_flag:
        config_path = os.path.join(config_dir, "config.yaml")
        with open(config_path, "r") as f:
            base_config = yaml.safe_load(f)
    else:
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

    for i, network in enumerate(args.data.networks):
        node_normalizer, edge_normalizer = load_normalizer(args=args)
        node_normalizers.append(node_normalizer)
        edge_normalizers.append(edge_normalizer)

        # Create torch dataset and split
        data_path_network = os.path.join(data_path, network)
        print(f"Loading {network} dataset")
        dataset = GridDatasetMem(
            root=data_path_network,
            norm_method=args.data.normalization,
            node_normalizer=node_normalizer,
            edge_normalizer=edge_normalizer,
            pe_dim=args.model.pe_dim,
            mask_dim=args.data.mask_dim,
            transform=get_transform(args=args),
        )
        datasets.append(dataset)

        num_scenarios = args.data.scenarios[i]
        if num_scenarios > len(dataset):
            warnings.warn(
                f"Requested number of scenarios ({num_scenarios}) exceeds dataset size ({len(dataset)}). "
                "Using the full dataset instead.",
            )
            num_scenarios = len(dataset)

        # Create a subset
        subset_indices = list(range(num_scenarios))
        dataset = Subset(dataset, subset_indices)

        node_normalizer.to(device)
        edge_normalizer.to(device)

        train_dataset, val_dataset, test_dataset = split_dataset(
            dataset,
            data_dir,
            args.data.val_ratio,
            args.data.test_ratio,
        )

        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        test_datasets.append(test_dataset)

    train_dataset_multi = ConcatDataset(train_datasets)
    val_dataset_multi = ConcatDataset(val_datasets)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset_multi,
        batch_size=args.training.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset_multi,
        batch_size=args.training.batch_size,
        shuffle=False,
    )
    test_loaders = [
        DataLoader(i, batch_size=args.training.batch_size, shuffle=False)
        for i in test_datasets
    ]

    # Create model
    if model_path:
        model = torch.load(model_path, weights_only=False, map_location=device)
    else:
        model = load_model(args=args)

    print(model)
    print(
        "Model parameters: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    if checkpoint_flag:
        checkpoint_path = os.path.join(model_dir, "checkpoint_last_epoch.pth")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)

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

    if checkpoint_flag:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    best_model_path = os.path.join(model_dir, "best_model.pth")
    early_stopper = EarlyStopper(
        best_model_path,
        args.callbacks.patience,
        args.callbacks.tol,
    )

    loss_fn = get_loss_function(args)

    mlflow_plugin = MLflowLoggerPlugin(steps=10, params=args.flatten())
    checkpointer = CheckpointerPlugin(model_dir, steps=None)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        loss_fn=loss_fn,
        early_stopper=early_stopper,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        lr_scheduler=scheduler,
        plugins=[mlflow_plugin, checkpointer],
    )
    # Train model
    if checkpoint_flag:
        trainer.train(
            checkpoint["epoch"] + 1,
            args.training.epochs - checkpoint["epoch"] - 1,
        )
    else:
        trainer.train(0, args.training.epochs)

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

    for i, network in enumerate(args.data.networks):
        for task in ["PF", "OPF", "Reconstruction"]:
            mask_ratio = getattr(
                args.data,
                "mask_ratio",
                0.5,
            )  # Default to 0.5 if mask_ratio doesn't exist
            df, figs = eval_node_level_task(
                dataset=datasets[i],
                model=best_model,
                task=task,
                test_loader=test_loaders[i],
                mask_dim=args.data.mask_dim,
                mask_ratio=mask_ratio,
                node_normalizer=node_normalizers[i],
                device=device,
                plot_dist=args.verbose,
            )

            # Log metric results
            df_path = os.path.join(test_dir, f"{task}_metrics_results_{network}.csv")
            df.to_csv(df_path)

            plot_paths = os.path.join(
                test_dir,
                f"{task}_evaluation_plots_{network}.html",
            )
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
            f"gridFM predict --model_exp_id {run.info.experiment_id} --model_run_id {run.info.run_id} --model_name best_model --config {config_dest} --eval_name YOUR_EVAL_NAME \n",
        )
        f.write(
            f"gridFM predict --model_exp_id {run.info.experiment_id} --model_run_id {run.info.run_id} --model_name best_model --config {config_dest} --eval_name YOUR_EVAL_NAME",
        )


def main_standard(args, device):
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
                f"\nGrid search: {i + 1}/{len(grid_combinations)} with params: {grid_params}",
            )
            with mlflow.start_run() as run:
                run_training(
                    args.config,
                    grid_params,
                    args.data_path,
                    device,
                    run,
                    checkpoint_flag=False,
                )
    else:
        print("No grid search config file provided. Running single training")
        with mlflow.start_run() as run:
            run_training(
                args.config,
                {},
                args.data_path,
                device,
                run,
                checkpoint_flag=False,
            )


def main_checkpoint(args, device):
    if args.grid:
        warnings.warn("Grid search not supported with model checkpoint")
    if args.config:
        warnings.warn("No need to specify config file")

    with mlflow.start_run(
        experiment_id=args.model_exp_id,
        run_id=args.model_run_id,
    ) as run:
        run_training(args.config, {}, args.data_path, device, run, checkpoint_flag=True)


def main_fine_tuning(args, device):
    exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.exp is None:
        experiment_name = f"exp_{exp_name}"
    else:
        experiment_name = f"{args.exp}"

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        run_training(
            args.config,
            {},
            args.data_path,
            device,
            run,
            checkpoint_flag=False,
            model_path=args.model_path,
        )


def eval(model_path, run, config_path, data_path, device):
    model = torch.load(model_path, weights_only=False, map_location=device).to(device)

    artifact_dir = os.path.join(
        "mlruns",
        run.info.experiment_id,
        run.info.run_id,
        "artifacts",
    )

    # Define log directories
    config_dir = os.path.join(artifact_dir, "config")
    data_dir = os.path.join(artifact_dir, "data_idx")
    test_dir = os.path.join(artifact_dir, "test")

    # Create log directories if they don't exist
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Load the base config
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Save config file
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
    test_datasets = []

    for i, network in enumerate(args.data.networks):
        node_normalizer, edge_normalizer = load_normalizer(args=args)
        node_normalizers.append(node_normalizer)
        edge_normalizers.append(edge_normalizer)

        # Create torch dataset and split
        data_path_network = os.path.join(data_path, network)
        print(f"Loading {network} dataset")
        dataset = GridDatasetMem(
            root=data_path_network,
            norm_method=args.data.normalization,
            node_normalizer=node_normalizer,
            edge_normalizer=edge_normalizer,
            pe_dim=args.model.pe_dim,
            mask_dim=args.data.mask_dim,
            transform=get_transform(args=args),
        )
        datasets.append(dataset)

        num_scenarios = args.data.scenarios[i]
        if num_scenarios > len(dataset):
            warnings.warn(
                f"Requested number of scenarios ({num_scenarios}) exceeds dataset size ({len(dataset)}). "
                "Using the full dataset instead.",
            )
            num_scenarios = len(dataset)

        subset_indices = list(range(num_scenarios))
        dataset = Subset(dataset, subset_indices)

        node_normalizer.to(device)
        edge_normalizer.to(device)

        _, _, test_dataset = split_dataset(
            dataset,
            data_dir,
            args.data.val_ratio,
            args.data.test_ratio,
        )

        test_datasets.append(test_dataset)

    test_loaders = [
        DataLoader(i, batch_size=args.training.batch_size, shuffle=False)
        for i in test_datasets
    ]

    mlflow.log_params(args.flatten())
    for i, network in enumerate(args.data.networks):
        for task in ["PF", "OPF", "Reconstruction"]:
            mask_ratio = getattr(
                args.data,
                "mask_ratio",
                0.5,
            )  # Default to 0.5 if mask_ratio doesn't exist
            df, figs = eval_node_level_task(
                dataset=datasets[i],
                model=model,
                task=task,
                test_loader=test_loaders[i],
                mask_dim=args.data.mask_dim,
                mask_ratio=mask_ratio,
                node_normalizer=node_normalizers[i],
                device=device,
                plot_dist=args.verbose,
            )
            # Log metric results
            df_path = os.path.join(test_dir, f"{task}_metrics_results_{network}.csv")
            df.to_csv(df_path)

            plot_paths = os.path.join(
                test_dir,
                f"{task}_evaluation_plots_{network}.html",
            )
            with open(plot_paths, "a") as f:
                for fig in figs:
                    f.write(pio.to_html(fig, full_html=False, include_plotlyjs="cdn"))

        # Log node and edge stats
        log_file_path = os.path.join(artifact_dir, f"stats_{network}.log")

        # Write the print statements to the log file
        with open(log_file_path, "w") as log_file:
            log_file.write("Dataset node_stats: " + str(datasets[i].node_stats) + "\n")
            log_file.write("Dataset edge_stats: " + str(datasets[i].edge_stats) + "\n")


def main_eval(
    args,
    device,
):
    if args.model_path is None and (
        (args.model_exp_id is None)
        or (args.model_run_id is None)
        or (args.model_name is None)
    ):
        raise ValueError(
            "Either model_path or (model_exp_id, model_run_id, model_name) must be provided",
        )
    if args.model_path is not None:
        mlflow.set_experiment(args.eval_name)
        with mlflow.start_run() as run:
            eval(args.model_path, run, args.config, args.data_path, device)

    else:
        # Start the parent run using the provided experiment ID and run ID
        # This is necessary to create a child run
        with mlflow.start_run(
            experiment_id=args.model_exp_id,
            run_id=args.model_run_id,
        ) as parent_run:
            # Start a nested run
            with mlflow.start_run(
                experiment_id=args.model_exp_id,
                parent_run_id=args.model_run_id,
                run_name=args.eval_name,
                nested=True,
            ) as nested_run:
                # load model from parent run artifact dir
                model_path = os.path.join(
                    "mlruns",
                    parent_run.info.experiment_id,
                    parent_run.info.run_id,
                    "artifacts",
                    "model",
                    args.model_name + ".pth",
                )

                eval(model_path, nested_run, args.config, args.data_path, device)
