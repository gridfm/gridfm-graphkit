from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.io.param_handler import NestedNamespace
from gridfm_graphkit.io.registries import DATASET_WRAPPER_REGISTRY
from gridfm_graphkit.training.callbacks import (
    SaveBestModelStateDict,
    EpochTimerCallback,
)
import importlib
import numpy as np
import os
import time
import yaml
import torch
import torch.distributed as dist
import pandas as pd

from gridfm_graphkit.io.param_handler import get_task
from gridfm_graphkit.tasks.compute_ac_dc_metrics import compute_ac_dc_metrics
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import lightning as L


def _load_plugins(plugins: list[str]) -> None:
    """Import plugin packages so their registry decorators fire."""
    for plugin_pkg in plugins:
        try:
            importlib.import_module(plugin_pkg)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Plugin package '{plugin_pkg}' could not be imported: {e}. "
                "Make sure it is installed in the current environment.",
            ) from e


def _validate_dataset_wrapper(name: str | None) -> None:
    """Raise a helpful error if *name* is not registered in DATASET_WRAPPER_REGISTRY."""
    if name is None:
        return
    if name not in DATASET_WRAPPER_REGISTRY:
        available = list(DATASET_WRAPPER_REGISTRY)
        raise KeyError(
            f"Dataset wrapper '{name}' is not registered. "
            f"Available wrappers: {available}. "
            "If it lives in a plugin package, pass it via --plugins.",
        )


def benchmark_cli(args):
    """Benchmark train-dataloader iteration speed over one or more epochs."""
    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    config_args = NestedNamespace(**base_config)

    num_workers_override = getattr(args, "num_workers", None)
    if num_workers_override is not None:
        config_args.data.workers = num_workers_override

    _load_plugins(getattr(args, "plugins", []))

    dataset_wrapper = getattr(args, "dataset_wrapper", None)
    dataset_wrapper_cache_dir = getattr(args, "dataset_wrapper_cache_dir", None)
    _validate_dataset_wrapper(dataset_wrapper)

    print("Setting up datamodule...")
    t0 = time.perf_counter()
    dm = LitGridHeteroDataModule(
        config_args,
        args.data_path,
        dataset_wrapper=dataset_wrapper,
        dataset_wrapper_cache_dir=dataset_wrapper_cache_dir,
    )
    dm.setup(stage="fit")
    setup_time = time.perf_counter() - t0
    print(f"  Setup time        : {setup_time:.2f}s")

    loader = dm.train_dataloader()
    num_batches = len(loader)
    print(f"  Train batches     : {num_batches}")
    print(f"  Batch size        : {config_args.training.batch_size}")
    print(f"  Workers           : {config_args.data.workers}")
    print(f"  Dataset wrapper   : {dataset_wrapper or 'none'}")
    print()

    epoch_times = []
    for epoch in range(args.epochs):
        t_start = time.perf_counter()
        for _batch in loader:
            pass
        elapsed = time.perf_counter() - t_start
        per_batch = elapsed / num_batches if num_batches > 0 else 0.0
        epoch_times.append(elapsed)
        print(
            f"Epoch {epoch:>3}: {elapsed:7.3f}s total  "
            f"{per_batch:.4f}s/batch  ({num_batches} batches)",
        )

    if args.epochs > 1:
        avg = sum(epoch_times) / len(epoch_times)
        print(f"\nAverage over {args.epochs} epochs: {avg:.3f}s")


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
    if getattr(args, "tf32", False):
        torch.set_float32_matmul_precision("high")  # enables TF32 on Ampere+ GPUs

    logger = MLFlowLogger(
        save_dir=args.log_dir,
        experiment_name=args.exp_name,
        run_name=args.run_name,
    )

    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    config_args = NestedNamespace(**base_config)

    L.seed_everything(config_args.seed, workers=True)

    normalizer_stats_path = getattr(args, "normalizer_stats", None)
    dataset_wrapper = getattr(args, "dataset_wrapper", None)
    dataset_wrapper_cache_dir = getattr(args, "dataset_wrapper_cache_dir", None)

    # CLI --num_workers overrides the YAML value (useful for debugging with 0)
    num_workers_override = getattr(args, "num_workers", None)
    if num_workers_override is not None:
        config_args.data.workers = num_workers_override

    _load_plugins(getattr(args, "plugins", []))
    _validate_dataset_wrapper(dataset_wrapper)

    litGrid = LitGridHeteroDataModule(
        config_args,
        args.data_path,
        normalizer_stats_path=normalizer_stats_path,
        dataset_wrapper=dataset_wrapper,
        dataset_wrapper_cache_dir=dataset_wrapper_cache_dir,
    )
    model = get_task(config_args, litGrid.data_normalizers)
    if args.command != "train":
        print(f"Loading model weights from {args.model_path}")
        state_dict = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(state_dict)

    precision = "bf16-true" if getattr(args, "bfloat16", False) else None
    if precision:
        print("Using bfloat16 precision (via Lightning Trainer precision='bf16-true')")

    compile_mode = getattr(args, "compile", None)
    if compile_mode is not None:
        if compile_mode in ("max-autotune", "max-autotune-no-cudagraphs"):
            # Allow ATen GEMM as fallback so Triton configs that exceed GPU
            # shared-memory limits (e.g. triton_mm OOM) are skipped gracefully
            # instead of causing autotuning errors.
            import torch._inductor.config as inductor_cfg

            inductor_cfg.max_autotune_gemm_backends = "ATEN,TRITON"
        print(f"Compiling model with torch.compile(mode='{compile_mode}')")
        model.model = torch.compile(model.model, mode=compile_mode)

    trainer_kwargs = {}
    if precision:
        trainer_kwargs["precision"] = precision
    profiler = getattr(args, "profiler", None)

    report_performance = getattr(args, "report_performance", False)
    epoch_timer = EpochTimerCallback() if report_performance else None

    training_callbacks = get_training_callbacks(config_args)
    if epoch_timer is not None:
        training_callbacks = training_callbacks + [epoch_timer]

    trainer = L.Trainer(
        logger=logger,
        accelerator=config_args.training.accelerator,
        devices=config_args.training.devices,
        strategy=config_args.training.strategy,
        log_every_n_steps=1000,
        default_root_dir=args.log_dir,
        max_epochs=config_args.training.epochs,
        callbacks=training_callbacks,
        **trainer_kwargs,
        profiler=profiler,
    )
    if args.command == "train" or args.command == "finetune":
        trainer.fit(model=model, datamodule=litGrid)
        if (
            report_performance
            and epoch_timer is not None
            and epoch_timer.last_epoch_time is not None
        ):
            print(f"[performance] last epoch time : {epoch_timer.last_epoch_time:.3f}s")
            if (
                epoch_timer.last_epoch_iters_per_sec is not None
                and epoch_timer._last_batch_count > 0
            ):
                print(
                    f"[performance] last epoch it/s : {epoch_timer.last_epoch_iters_per_sec:.2f}",
                )

    if args.command != "predict":
        # Reuse the fit trainer when coming from train/finetune so that
        # torch.compile kernel caches are already warm (avoids a second
        # AUTOTUNE pass on the first test batch).
        if args.command in ("train", "finetune"):
            test_trainer = trainer
        else:
            test_trainer = L.Trainer(
                logger=logger,
                accelerator=config_args.training.accelerator,
                devices=1,
                num_nodes=1,
                log_every_n_steps=1,
                default_root_dir=args.log_dir,
                **trainer_kwargs,
                profiler=profiler,
            )
        test_results = test_trainer.test(model=model, datamodule=litGrid)
        if report_performance:
            # test_results[0] may be empty when metrics are routed to the logger
            # only; fall back to trainer.callback_metrics which always has them.
            metrics = (
                test_results[0]
                if test_results and test_results[0]
                else dict(test_trainer.callback_metrics)
            )
            if metrics:
                first_metric, first_value = next(iter(metrics.items()))
                print(f"[performance] {first_metric} : {first_value}")
            else:
                print("[performance] no test metrics available")

    artifacts_dir = None
    is_rank0 = (
        not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
    )
    if is_rank0:
        artifacts_dir = os.path.join(
            logger.save_dir,
            logger.experiment_id,
            logger.run_id,
            "artifacts",
        )

    compute_dc_ac = getattr(args, "compute_dc_ac_metrics", False)
    if is_rank0 and compute_dc_ac:
        sn_mva = config_args.data.baseMVA
        for grid_name in config_args.data.networks:
            raw_dir = os.path.join(args.data_path, grid_name, "raw")
            print(f"\nComputing ground-truth AC/DC metrics for {grid_name}...")
            compute_ac_dc_metrics(artifacts_dir, raw_dir, grid_name, sn_mva)

    save_output = getattr(args, "save_output", False) or args.command == "predict"
    if is_rank0 and save_output:
        if len(config_args.data.networks) > 1:
            raise NotImplementedError(
                "Predict/save_output with multiple grids is not yet supported.",
            )

        predict_trainer = L.Trainer(
            logger=logger,
            accelerator=config_args.training.accelerator,
            devices=1,
            num_nodes=1,
            log_every_n_steps=1,
            default_root_dir=args.log_dir,
            **trainer_kwargs,
            profiler=profiler,
        )
        predictions = predict_trainer.predict(model=model, datamodule=litGrid)

        rows = {key: [] for key in predictions[0].keys()}
        for batch in predictions:
            for key in rows:
                rows[key].append(batch[key])

        df = pd.DataFrame({key: np.concatenate(vals) for key, vals in rows.items()})

        grid_name = config_args.data.networks[0]
        if args.command == "predict":
            output_dir = args.output_path
        else:
            output_dir = os.path.join(artifacts_dir, "test")
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{grid_name}_predictions.parquet")
        df.to_parquet(out_path, index=False)
        print(f"Saved predictions to {out_path}")
