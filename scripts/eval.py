from gridFM.datasets.powergrid import GridDatasetMem
from gridFM.io.param_handler import *
from gridFM.datasets.data_normalization import *
from gridFM.datasets.globals import *
import torch
from torch_geometric.loader import DataLoader
import os
import mlflow
import mlflow.pytorch
import argparse
from gridFM.datasets.utils import split_dataset
import yaml
import random
from gridFM.evaluation.node_level import eval_node_level_task
import plotly.io as pio
from torch.utils.data import ConcatDataset, Subset
import warnings


def eval(model_path, run, config_path, data_path, args, device):

    model = torch.load(model_path, weights_only=False, map_location=device).to(device)

    artifact_dir = os.path.join(
        "../mlruns",
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
            df, figs = eval_node_level_task(
                model=model,
                task=task,
                test_loader=test_loaders[i],
                mask_dim=args.data.mask_dim,
                mask_ratio=args.data.mask_ratio,
                node_normalizer=node_normalizers[i],
                device=device,
                plot_dist=args.verbose,
            )
            # Log metric results
            df_path = os.path.join(test_dir, f"{task}_metrics_results_{network}.csv")
            df.to_csv(df_path)

            plot_paths = os.path.join(
                test_dir, f"{task}_evaluation_plots_{network}.html"
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


def main(
    model_path,
    config_path,
    eval_name,
    model_exp_id,
    model_run_id,
    model_name,
    data_path,
    device,
):

    if model_path is not None:
        mlflow.set_experiment(eval_name)
        with mlflow.start_run() as run:
            eval(model_path, run, config_path, data_path, args, device)

    else:

        # Start the parent run using the provided experiment ID and run ID
        # This is necessary to create a child run
        with mlflow.start_run(
            experiment_id=model_exp_id, run_id=model_run_id
        ) as parent_run:

            # Start a nested run
            with mlflow.start_run(
                experiment_id=model_exp_id,
                parent_run_id=model_run_id,
                run_name=eval_name,
                nested=True,
            ) as nested_run:

                # load model from parent run artifact dir
                model_path = os.path.join(
                    "../mlruns",
                    parent_run.info.experiment_id,
                    parent_run.info.run_id,
                    "artifacts",
                    "model",
                    model_name + ".pth",
                )

                eval(model_path, nested_run, config_path, data_path, args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script")

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path of the model to use. If specified, the evaluation is ran with this model and a new mlflow experiment is created. Otherwise, model_exp_id, model_run_id and model_name need to be specified and a logged model will be used for evaluation",
    )

    parser.add_argument(
        "--model_exp_id",
        type=str,
        help="ID of the experiment associated with the model to be used",
        default=None,
    )

    parser.add_argument(
        "--model_run_id",
        type=str,
        help="ID of the run associated with the model to be used",
        default=None,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="best_model",
        help="Name of the model to use",
    )

    parser.add_argument(
        "--eval_name",
        type=str,
        required=True,
        default=None,
        help="Name to give to the eval",
    )

    default_data_path = os.path.join(os.getcwd(), "..", "data")
    parser.add_argument(
        "--data_path",
        type=str,
        default=default_data_path,
        help=f"Data root directory (default: {default_data_path})",
    )

    args = parser.parse_args()

    if args.model_path is None and (
        (args.model_exp_id is None)
        or (args.model_run_id is None)
        or (args.model_name is None)
    ):
        raise ValueError(
            "Either model_path or (model_exp_id, model_run_id, model_name) must be provided"
        )

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Evaluating {args.model_name}")

    mlflow.set_tracking_uri("file:../mlruns")

    main(
        args.model_path,
        args.config,
        args.eval_name,
        args.model_exp_id,
        args.model_run_id,
        args.model_name,
        args.data_path,
        device,
    )
