from gridfm_graphkit.datasets.powergrid_datamodule import LitGridDataModule
from gridfm_graphkit.io.param_handler import NestedNamespace
from gridfm_graphkit.training.callbacks import SaveBestModelStateDict
import numpy as np
import os
import yaml
import torch
import random
import pandas as pd

from gridfm_graphkit.tasks.feature_reconstruction_task import FeatureReconstructionTask
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import lightning as L


def get_training_callbacks(args):
    early_stop_callback = EarlyStopping(
        monitor="Validation loss",
        min_delta=args.callbacks.tol,
        patience=args.callbacks.patience,
        verbose=False,
        mode="min",
    )

    save_best_model_callback = SaveBestModelStateDict(
        monitor="Validation loss",
        mode="min",
        filename="best_model_state_dict.pt",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="Validation loss",  # or whichever metric you track
        mode="min",
        save_last=True,
        save_top_k=0,
    )

    return [early_stop_callback, save_best_model_callback, checkpoint_callback]


def main_cli(args):
    logger = MLFlowLogger(
        save_dir=args.log_dir,
        experiment_name=args.exp_name,
        run_name=args.run_name,
    )

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
    if args.command != "train":
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
        callbacks=get_training_callbacks(config_args),
    )
    if args.command == "train" or args.command == "finetune":
        trainer.fit(model=model, datamodule=litGrid)

    if args.command != "predict":
        trainer.test(model=model, datamodule=litGrid)

    if args.command == "predict":
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
