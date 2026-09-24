#!/usr/bin/env python3
"""Benchmark PF inference with sample loading performed inside Dataset.get()."""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader

import benchmark_pf_inference as base


class DiskBackedHeteroDataset(Dataset):
    """Dataset that loads each processed sample from disk in get()."""

    def __init__(self, sample_paths: list[Path], virtual_length: int | None = None):
        if not sample_paths:
            raise ValueError("DiskBackedHeteroDataset requires at least one sample path.")
        self.sample_paths = sample_paths
        self.num_loaded_samples = len(sample_paths)
        self.virtual_length = self.num_loaded_samples if virtual_length is None else virtual_length
        super().__init__(root=None, transform=None)

    def len(self) -> int:
        return self.virtual_length

    def get(self, idx: int) -> HeteroData:
        sample_path = self.sample_paths[idx % self.num_loaded_samples]
        data_dict = torch.load(sample_path, map_location="cpu", weights_only=True)
        return HeteroData.from_dict(data_dict)


def build_dataloader_from_paths(
    samples: list[Path],
    batch_size: int,
    virtual_num_samples: int | None = None,
) -> DataLoader:
    dataset = DiskBackedHeteroDataset(sample_paths=samples, virtual_length=virtual_num_samples)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": base.DEFAULT_NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": base.DEFAULT_NUM_WORKERS > 0,
    }
    if base.DEFAULT_NUM_WORKERS > 0:
        loader_kwargs["multiprocessing_context"] = "fork"
    return DataLoader(**loader_kwargs)


def clone_paths(sample_paths: list[Path]) -> list[Path]:
    return list(sample_paths)


def build_sample_paths(
    config,
    data_path: str | Path,
    num_samples_to_materialize: int,
) -> list[Path]:
    _, processed_dir = base.get_network_paths(config, data_path)
    return [processed_dir / f"data_index_{idx}.pt" for idx in range(num_samples_to_materialize)]


def main() -> None:
    args = base.parse_args()
    batch_sizes = base.parse_batch_sizes(args.batch_sizes)
    device = base.resolve_device(args.device)
    config = base.load_config(args.config)
    base.validate_task(config)

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.warmup_batches < 0:
        raise ValueError("--warmup-batches must be non-negative.")
    if args.preload_workers <= 0:
        raise ValueError("--preload-workers must be positive.")
    if args.max_preloaded_samples <= 0:
        raise ValueError("--max-preloaded-samples must be positive.")

    print(f"Building model for {config.task.task_name}...")
    compile_mode = None if args.no_compile else args.compile
    model, num_params = base.configure_model(config=config, device=device, compile_mode=compile_mode)
    print(f"Model parameters: {num_params:,}")
    transforms = base.get_task_transforms(config)

    preload_num_samples = min(args.num_samples, args.max_preloaded_samples)
    print(
        f"Preparing {preload_num_samples} sample paths for on-demand disk loading "
        "(disk I/O is included in outer elapsed timing)."
    )
    sample_paths = build_sample_paths(
        config=config,
        data_path=args.data_path,
        num_samples_to_materialize=preload_num_samples,
    )
    print("Loading reference samples only for normalizer fit (excluded from timing)...")
    normalizer_samples = base.preload_raw_samples(
        config,
        args.data_path,
        preload_num_samples,
        args.preload_workers,
    )
    normalizer = base.build_normalizer(config=config, raw_samples=normalizer_samples)

    if args.num_samples > preload_num_samples:
        print(
            f"Using virtual dataset length {args.num_samples} by reusing "
            f"{preload_num_samples} on-disk samples modulo dataset size."
        )

    # Reuse benchmark logic while swapping the dataset behavior.
    base.build_dataloader = build_dataloader_from_paths
    base.clone_samples = clone_paths

    results = []
    for batch_size in batch_sizes:
        num_batches = (args.num_samples + batch_size - 1) // batch_size
        try:
            print(f"\nRunning warmup for batch size {batch_size}...")
            base.warmup(
                raw_samples=sample_paths,
                num_samples=args.num_samples,
                batch_size=batch_size,
                warmup_batches=args.warmup_batches,
                model=model,
                normalizer=normalizer,
                transforms=transforms,
                device=device,
            )

            print(f"Benchmarking batch size {batch_size}...")
            result = base.benchmark_batch_size(
                raw_samples=sample_paths,
                num_samples=args.num_samples,
                batch_size=batch_size,
                model=model,
                normalizer=normalizer,
                transforms=transforms,
                device=device,
                breakdown=args.breakdown,
            )
            result["batch_size"] = batch_size
            result["num_samples"] = args.num_samples
            result["num_batches"] = num_batches
            result["num_params"] = num_params
            result["device"] = str(device)
            result["status"] = "ok"
            result["error_type"] = ""
            result["error_message"] = ""
        except (RuntimeError, MemoryError) as exc:
            base.clear_device_cache(device)
            status = "oom" if base.is_out_of_memory_error(exc) else "failed"
            result = base.make_failed_result(
                batch_size=batch_size,
                num_samples=args.num_samples,
                num_batches=num_batches,
                num_params=num_params,
                device=device,
                status=status,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            print(
                f"Skipping batch size {batch_size} after {status}: "
                f"{type(exc).__name__}: {exc}"
            )
        base.print_result(result)
        results.append(result)
        base.write_results_csv(
            output_csv=args.output_csv,
            results=results,
            args=args,
            config=config,
            warmup_batches=args.warmup_batches,
            compile_mode=compile_mode,
            num_params=num_params,
        )

    print(f"\nResults written to {args.output_csv}")


if __name__ == "__main__":
    main()
