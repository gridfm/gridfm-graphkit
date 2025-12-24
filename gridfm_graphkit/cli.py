from gridfm_graphkit.datasets.powergrid_datamodule import LitGridDataModule
from gridfm_graphkit.io.param_handler import NestedNamespace
from gridfm_graphkit.training.callbacks import get_training_callbacks
from gridfm_graphkit.iterate import run_iterate_experiments
import numpy as np
import os
import yaml
import torch
import random
import pandas as pd
from gridfm_graphkit.tasks import FeatureReconstructionTask
from lightning.pytorch.loggers import MLFlowLogger
import lightning as L

from jsonargparse import Namespace


def main_cli(args):
    logger = MLFlowLogger(
        save_dir=args.log_dir,
        experiment_name=args.exp_name,
        run_name=args.run_name,
    )

    subcommand = args.subcommand
    args = args[subcommand]

    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    config_args = NestedNamespace(**base_config)

    torch.manual_seed(config_args.seed)
    random.seed(config_args.seed)
    np.random.seed(config_args.seed)

    litGrid = LitGridDataModule(config_args, args.data_path)
    model = FeatureReconstructionTask(
        config_args,
        litGrid.node_normalizers,
        litGrid.edge_normalizers,
    )
    if subcommand != "train":
        print(f"Loading model weights from {args.model_path}")
        state_dict = torch.load(args.model_path)
        model.load_state_dict(state_dict)

    trainer = L.Trainer(
        logger=logger,
        accelerator=config_args.training.accelerator,
        devices=config_args.training.devices,
        strategy=config_args.training.strategy,
        log_every_n_steps=1,
        default_root_dir=args.log_dir,
        max_epochs=config_args.training.epochs,
        callbacks=get_training_callbacks(config_args.callbacks),
    )
    if subcommand == "train" or subcommand == "finetune":
        trainer.fit(model=model, datamodule=litGrid)

    if subcommand != "predict":
        trainer.test(model=model, datamodule=litGrid)

    if subcommand == "predict":
        predictions = trainer.predict(model=model, datamodule=litGrid)
        all_outputs = []
        all_scenarios = []
        all_bus_numbers = []

        for batch in predictions:
            all_outputs.append(batch["output"])
            all_scenarios.append(batch["scenario_id"])
            all_bus_numbers.append(batch["bus_number"])

        # Concatenate all
        outputs = np.concatenate(all_outputs, axis=0)  # shape: [num_nodes, 6]
        scenario_ids = np.concatenate(all_scenarios, axis=0)
        bus_numbers = np.concatenate(all_bus_numbers, axis=0)

        # Build DataFrame
        df = pd.DataFrame(
            {
                "scenario": scenario_ids,
                "bus": bus_numbers,
                "PD_pred": outputs[:, 0],
                "QD_pred": outputs[:, 1],
                "PG_pred": outputs[:, 2],
                "QG_pred": outputs[:, 3],
                "VM_pred": outputs[:, 4],
                "VA_pred": outputs[:, 5],
            },
        )

        # Save CSV
        output_dir = os.path.join(args.output_path)
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "predictions.csv")
        df.to_csv(csv_path, index=False)

        print(f"Saved predictions to {csv_path}")


def iterate_cli(config_args):
    # validate inputs
    if config_args.seed is not None:
        assert isinstance(config_args.seed, int), "seed must be an integer"
    torch.manual_seed(config_args.seed)
    random.seed(config_args.seed)
    np.random.seed(config_args.seed)

    return run_iterate_experiments(
        args=config_args,  # TODO
        model_spec=config_args.model,
        training_spec=config_args.training,
        optimizer_spec=config_args.optimizer,
        callbacks_spec=config_args.callbacks,
        hpo_spec=config_args.hpo_spec,
        tasks=config_args.tasks,
        seed=config_args.seed,
    )
