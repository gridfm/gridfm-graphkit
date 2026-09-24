#!/usr/bin/env python3
"""Benchmark graph-model inference throughput over a batch-size sweep."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import itertools
import os
import random
import time
from pathlib import Path

import torch
import yaml
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from gridfm_graphkit.datasets.globals import VN_KV
from gridfm_graphkit.io.param_handler import (
    NestedNamespace,
    get_task_transforms,
    load_model,
    load_normalizer,
)


DEFAULT_CONFIG = "scripts/benchmark_inference/config_pf_base.yaml"
DEFAULT_OUTPUT_CSV = "scripts/benchmark_inference/benchmark_results.csv"
DEFAULT_BATCH_SIZES = "1024,2048,4096,8192, 16384, 32768, 65536" #"16,32,64,128,256,512,1024,2048,4096,8192"
DEFAULT_NUM_SAMPLES = 1_000_000
DEFAULT_MAX_PRELOADED_SAMPLES = 10_000
DEFAULT_WARMUP_BATCHES = 20
DEFAULT_PRELOAD_WORKERS = max(1, min(84, os.cpu_count() or 1))
DEFAULT_NUM_WORKERS = 32
DEFAULT_COMPILE = "reduce-overhead"
DEFAULT_DEVICE = "cuda"
DEFAULT_DATA_PATH = None
COMPILE_CHOICES = [
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark preprocessing + H2D + forward throughput for PF or OPF.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument(
        "--max-preloaded-samples",
        type=int,
        default=DEFAULT_MAX_PRELOADED_SAMPLES,
        help="Maximum number of processed samples to preload into RAM.",
    )
    parser.add_argument("--batch-sizes", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--warmup-batches", type=int, default=DEFAULT_WARMUP_BATCHES)
    parser.add_argument(
        "--preload-workers",
        type=int,
        default=DEFAULT_PRELOAD_WORKERS,
        help="Number of threads used to preload processed samples into RAM.",
    )
    parser.add_argument(
        "--compile",
        type=str,
        default=DEFAULT_COMPILE,
        nargs="?",
        const="default",
        choices=COMPILE_CHOICES,
        help="Enable torch.compile with the given mode (omit value for 'default').",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile.",
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help=(
            "Record per-stage host/CUDA timings with per-stage GPU synchronizes. "
            "Default: outer elapsed only, with one synchronize after all batches."
        ),
    )
    return parser.parse_args()


def parse_batch_sizes(raw: str) -> list[int]:
    batch_sizes = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not batch_sizes:
        raise ValueError("At least one batch size must be provided.")
    if any(batch_size <= 0 for batch_size in batch_sizes):
        raise ValueError("Batch sizes must be positive integers.")
    return batch_sizes


def load_config(path: str | Path) -> NestedNamespace:
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    return NestedNamespace(**config_dict)


def resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return device


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    elif device.type == "mps":
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def is_out_of_memory_error(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda error: out of memory" in msg
    return False


def get_network_paths(config: NestedNamespace, data_path: str | Path) -> tuple[Path, Path]:
    network = config.data.networks[0]
    network_dir = Path(data_path) / network
    return network_dir / "raw", network_dir / "processed"


def validate_task(config: NestedNamespace) -> None:
    supported_tasks = {"PowerFlow", "OptimalPowerFlow"}
    task_name = config.task.task_name
    if task_name not in supported_tasks:
        raise ValueError(
            f"Unsupported task_name '{task_name}'. Expected one of {sorted(supported_tasks)}.",
        )


def build_normalizer(
    config: NestedNamespace,
    raw_samples: list[HeteroData],
):
    normalizer = load_normalizer(config)
    random_base_mva = random.Random(config.seed).uniform(100.0, 150.0) # we use a random base MVA instead of fitting the normalizer on the data
    vn_kv_max = max(sample.x_dict["bus"][:, VN_KV].max().item() for sample in raw_samples)
    normalizer.fit_from_dict(
        {
            "baseMVA_orig": torch.tensor(float(getattr(config.data, "baseMVA", 100.0))),
            "baseMVA": torch.tensor(float(random_base_mva)),
            "vn_kv_max": torch.tensor(float(vn_kv_max)),
        },
    )
    return normalizer


def count_model_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def configure_model(
    config: NestedNamespace,
    device: torch.device,
    compile_mode: str | None,
) -> tuple[torch.nn.Module, int]:
    model = load_model(config)
    num_params = count_model_params(model)
    model.eval()
    model.to(device)
    if compile_mode is not None:
        # Match the compile-related setup in the main CLI.
        torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
        if compile_mode in ("max-autotune", "max-autotune-no-cudagraphs"):
            import torch._inductor.config as inductor_cfg

            inductor_cfg.max_autotune_gemm_backends = "ATEN,TRITON"
        print(f"Compiling model with torch.compile(mode='{compile_mode}')")
        model = torch.compile(model, mode=compile_mode)
    return model, num_params


def load_processed_sample(sample_path: Path) -> HeteroData:
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing processed sample: {sample_path}")
    data_dict = torch.load(sample_path, map_location="cpu", weights_only=True)
    return HeteroData.from_dict(data_dict)


def preload_raw_samples( # data is preloaded in RAM to avoid disk I/O (we factor this out from the runtime analysis)
    config: NestedNamespace,
    data_path: str | Path,
    num_samples: int,
    preload_workers: int,
) -> list[HeteroData]:
    _, processed_dir = get_network_paths(config, data_path)
    sample_paths = [processed_dir / f"data_index_{idx}.pt" for idx in range(num_samples)]
    workers = max(1, preload_workers)

    if workers == 1:
        return [
            load_processed_sample(sample_path)
            for sample_path in tqdm(
                sample_paths,
                desc="Preloading",
                leave=False,
            )
        ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        return list(
            tqdm(
                executor.map(load_processed_sample, sample_paths),
                total=num_samples,
                desc=f"Preloading ({workers} threads)",
                leave=False,
            )
        )


class PreloadedHeteroDataset(Dataset):
    def __init__(self, samples: list[HeteroData], virtual_length: int | None = None):
        if not samples:
            raise ValueError("PreloadedHeteroDataset requires at least one sample.")
        self.samples = samples
        self.num_loaded_samples = len(samples)
        self.virtual_length = self.num_loaded_samples if virtual_length is None else virtual_length
        super().__init__(root=None, transform=None)

    def len(self) -> int:
        return self.virtual_length

    def get(self, idx: int) -> HeteroData:
        data = self.samples[idx % self.num_loaded_samples] # this is to avoid having to load as many samples as virtual_length (which could be very large)
        return data


def clone_samples(raw_samples: list[HeteroData]) -> list[HeteroData]:
    return [sample.clone() for sample in raw_samples]


def build_dataloader(
    samples: list[HeteroData],
    batch_size: int,
    virtual_num_samples: int | None = None,
) -> DataLoader:
    dataset = PreloadedHeteroDataset(samples=samples, virtual_length=virtual_num_samples)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False, # no shuffling needed as we are not training
        "num_workers": DEFAULT_NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": DEFAULT_NUM_WORKERS > 0,
    }
    if DEFAULT_NUM_WORKERS > 0:
        loader_kwargs["multiprocessing_context"] = "fork" # this is faster than 'spawn' and works well with one GPU
    return DataLoader(
        **loader_kwargs,
    )


def run_forward(model, batch):
    return model(
        x_dict=batch.x_dict,
        edge_index_dict=batch.edge_index_dict,
        edge_attr_dict=batch.edge_attr_dict,
        mask_dict=batch.mask_dict,
    )


def run_batch_pipeline(
    cpu_batch,
    model,
    normalizer,
    transforms,
    device: torch.device,
) -> None:
    batch = cpu_batch.to(device)
    normalizer.transform(batch)
    batch = transforms(batch)
    with torch.inference_mode():
        output = run_forward(model, batch)
        normalizer.inverse_output(output, batch)


def empty_breakdown_fields() -> dict[str, float | int | str]:
    return {
        "batch_fetch_elapsed_ms": "",
        "transfer_elapsed_ms": "",
        "transfer_cuda_elapsed_ms": "",
        "normalizer_elapsed_ms": "",
        "normalizer_cuda_elapsed_ms": "",
        "transform_elapsed_ms": "",
        "transform_cuda_elapsed_ms": "",
        "forward_elapsed_ms": "",
        "forward_cuda_elapsed_ms": "",
        "total_elapsed_ms": "",
        "total_cuda_elapsed_ms": "",
        "outer_minus_total_ms": "",
        "outer_equals_total": "",
        "batch_fetch_per_sample_ms": "",
        "transfer_per_sample_ms": "",
        "transfer_cuda_per_sample_ms": "",
        "normalizer_per_sample_ms": "",
        "normalizer_cuda_per_sample_ms": "",
        "transform_per_sample_ms": "",
        "transform_cuda_per_sample_ms": "",
        "forward_per_sample_ms": "",
        "forward_cuda_per_sample_ms": "",
        "total_cuda_per_sample_ms": "",
        "total_per_sample_ms": "",
    }


def warmup(
    raw_samples: list[HeteroData],
    num_samples: int,
    batch_size: int,
    warmup_batches: int,
    model,
    normalizer,
    transforms,
    device: torch.device,
) -> None:
    if warmup_batches <= 0:
        return

    working_samples = clone_samples(raw_samples)
    loader = build_dataloader(
        samples=working_samples,
        batch_size=batch_size,
        virtual_num_samples=num_samples,
    )
    warmup_iter = itertools.islice(itertools.cycle(loader), warmup_batches) # cycle is used to repeat the same batches for the warmup in case the number of batches is less than the warmup batches
    for batch in tqdm(
        warmup_iter,
        total=warmup_batches,
        desc=f"Warmup bs={batch_size}",
        leave=False,
    ):
        batch = batch.to(device)
        normalizer.transform(batch)
        batch = transforms(batch)
        with torch.inference_mode():
            output = run_forward(model, batch)
            normalizer.inverse_output(output, batch)
    synchronize_device(device)


def benchmark_batch_size(
    raw_samples: list[HeteroData],
    num_samples: int,
    batch_size: int,
    model,
    normalizer,
    transforms,
    device: torch.device,
    *,
    breakdown: bool = False,
) -> dict[str, float | int]:
    working_samples = clone_samples(raw_samples)
    loader = build_dataloader(
        samples=working_samples,
        batch_size=batch_size,
        virtual_num_samples=num_samples,
    )
    num_batches = len(loader)
    assert (num_samples + batch_size - 1) // batch_size == num_batches, "num_batches does not match the number of samples and batch size"
    loader_iter = iter(loader)

    if not breakdown:
        outer_started_at = time.perf_counter()
        for _ in tqdm(
            range(num_batches),
            total=num_batches,
            desc=f"Benchmark bs={batch_size}",
            leave=False,
        ):
            run_batch_pipeline(
                next(loader_iter),
                model=model,
                normalizer=normalizer,
                transforms=transforms,
                device=device,
            )
        synchronize_device(device)
        outer_elapsed_s = time.perf_counter() - outer_started_at
        return {
            **empty_breakdown_fields(),
            "outer_elapsed_ms": outer_elapsed_s * 1000.0,
            "samples_per_s": num_samples / outer_elapsed_s,
            "it_per_s": num_batches / outer_elapsed_s,
            "batches_per_s": num_batches / outer_elapsed_s,
        }

    batch_fetch_elapsed_s = 0.0
    transfer_elapsed_s = 0.0
    normalizer_elapsed_s = 0.0
    transform_elapsed_s = 0.0
    forward_elapsed_s = 0.0
    transfer_cuda_elapsed_s = 0.0
    normalizer_cuda_elapsed_s = 0.0
    transform_cuda_elapsed_s = 0.0
    forward_cuda_elapsed_s = 0.0
    use_cuda_events = device.type == "cuda"
    outer_started_at = time.perf_counter()
    for _ in tqdm(
        range(num_batches),
        total=num_batches,
        desc=f"Benchmark bs={batch_size}",
        leave=False,
    ):
        started_at = time.perf_counter()
        cpu_batch = next(loader_iter)
        batch_fetch_elapsed_s += time.perf_counter() - started_at

        started_at = time.perf_counter()
        batch = cpu_batch
        if use_cuda_events:
            transfer_start = torch.cuda.Event(enable_timing=True)
            transfer_end = torch.cuda.Event(enable_timing=True)
            transfer_start.record()
        batch = batch.to(device)
        if use_cuda_events:
            transfer_end.record()
        synchronize_device(device)
        if use_cuda_events:
            transfer_cuda_elapsed_s += transfer_start.elapsed_time(transfer_end) / 1000.0
        transfer_elapsed_s += time.perf_counter() - started_at

        started_at = time.perf_counter()
        if use_cuda_events:
            normalizer_start = torch.cuda.Event(enable_timing=True)
            normalizer_end = torch.cuda.Event(enable_timing=True)
            normalizer_start.record()
        normalizer.transform(batch)
        if use_cuda_events:
            normalizer_end.record()
        synchronize_device(device)
        if use_cuda_events:
            normalizer_cuda_elapsed_s += normalizer_start.elapsed_time(normalizer_end) / 1000.0
        normalizer_elapsed_s += time.perf_counter() - started_at

        started_at = time.perf_counter()
        if use_cuda_events:
            transform_start = torch.cuda.Event(enable_timing=True)
            transform_end = torch.cuda.Event(enable_timing=True)
            transform_start.record()
        batch = transforms(batch)
        if use_cuda_events:
            transform_end.record()
        synchronize_device(device)
        if use_cuda_events:
            transform_cuda_elapsed_s += transform_start.elapsed_time(transform_end) / 1000.0
        transform_elapsed_s += time.perf_counter() - started_at

        started_at = time.perf_counter()
        if use_cuda_events:
            forward_start = torch.cuda.Event(enable_timing=True)
            forward_end = torch.cuda.Event(enable_timing=True)
            forward_start.record()
        with torch.inference_mode():
            output = run_forward(model, batch)
            normalizer.inverse_output(output, batch)
        if use_cuda_events:
            forward_end.record()
        synchronize_device(device)
        if use_cuda_events:
            forward_cuda_elapsed_s += forward_start.elapsed_time(forward_end) / 1000.0
        forward_elapsed_s += time.perf_counter() - started_at

    total_elapsed_s = (
        batch_fetch_elapsed_s
        + transfer_elapsed_s
        + normalizer_elapsed_s
        + transform_elapsed_s
        + forward_elapsed_s
    )
    total_cuda_elapsed_s = (
        batch_fetch_elapsed_s
        + transfer_cuda_elapsed_s
        + normalizer_cuda_elapsed_s
        + transform_cuda_elapsed_s
        + forward_cuda_elapsed_s
    )
    outer_elapsed_s = time.perf_counter() - outer_started_at
    outer_equals_total = int(
        abs(outer_elapsed_s - total_elapsed_s) <= max(1e-9, 1e-6 * outer_elapsed_s),
    )

    return {
        "batch_fetch_elapsed_ms": batch_fetch_elapsed_s * 1000.0,
        "transfer_elapsed_ms": transfer_elapsed_s * 1000.0,
        "transfer_cuda_elapsed_ms": transfer_cuda_elapsed_s * 1000.0 if use_cuda_events else "",
        "normalizer_elapsed_ms": normalizer_elapsed_s * 1000.0,
        "normalizer_cuda_elapsed_ms": normalizer_cuda_elapsed_s * 1000.0 if use_cuda_events else "",
        "transform_elapsed_ms": transform_elapsed_s * 1000.0,
        "transform_cuda_elapsed_ms": transform_cuda_elapsed_s * 1000.0 if use_cuda_events else "",
        "forward_elapsed_ms": forward_elapsed_s * 1000.0,
        "forward_cuda_elapsed_ms": forward_cuda_elapsed_s * 1000.0 if use_cuda_events else "",
        "total_elapsed_ms": total_elapsed_s * 1000.0,
        "total_cuda_elapsed_ms": total_cuda_elapsed_s * 1000.0 if use_cuda_events else "",
        "outer_elapsed_ms": outer_elapsed_s * 1000.0,
        "outer_minus_total_ms": (outer_elapsed_s - total_elapsed_s) * 1000.0,
        "outer_equals_total": outer_equals_total,
        "samples_per_s": num_samples / total_elapsed_s,
        "it_per_s": num_batches / total_elapsed_s,
        "batches_per_s": num_batches / total_elapsed_s,
        "batch_fetch_per_sample_ms": (batch_fetch_elapsed_s / num_samples) * 1000.0,
        "transfer_per_sample_ms": (transfer_elapsed_s / num_samples) * 1000.0,
        "transfer_cuda_per_sample_ms": (transfer_cuda_elapsed_s / num_samples) * 1000.0 if use_cuda_events else "",
        "normalizer_per_sample_ms": (normalizer_elapsed_s / num_samples) * 1000.0,
        "normalizer_cuda_per_sample_ms": (normalizer_cuda_elapsed_s / num_samples) * 1000.0 if use_cuda_events else "",
        "transform_per_sample_ms": (transform_elapsed_s / num_samples) * 1000.0,
        "transform_cuda_per_sample_ms": (transform_cuda_elapsed_s / num_samples) * 1000.0 if use_cuda_events else "",
        "forward_per_sample_ms": (forward_elapsed_s / num_samples) * 1000.0,
        "forward_cuda_per_sample_ms": (forward_cuda_elapsed_s / num_samples) * 1000.0 if use_cuda_events else "",
        "total_cuda_per_sample_ms": (total_cuda_elapsed_s / num_samples) * 1000.0 if use_cuda_events else "",
        "total_per_sample_ms": (total_elapsed_s / num_samples) * 1000.0,
    }


def flatten_for_csv(value, prefix: str = "") -> dict[str, str | int | float | bool]:
    if isinstance(value, Path):
        return {prefix: str(value)}
    if isinstance(value, argparse.Namespace):
        return flatten_for_csv(vars(value), prefix)
    if isinstance(value, NestedNamespace):
        return flatten_for_csv(vars(value), prefix)
    if isinstance(value, dict):
        flat: dict[str, str | int | float | bool] = {}
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}_{key}" if prefix else str(key)
            flat.update(flatten_for_csv(nested_value, nested_prefix))
        return flat
    if isinstance(value, (list, tuple)):
        return {prefix: ",".join(str(item) for item in value)}
    return {prefix: value}


def make_failed_result(
    *,
    batch_size: int,
    num_samples: int,
    num_batches: int,
    num_params: int,
    device: torch.device,
    status: str,
    error_type: str,
    error_message: str,
) -> dict[str, float | int | str]:
    return {
        "batch_size": batch_size,
        "num_samples": num_samples,
        "num_batches": num_batches,
        "num_params": num_params,
        "device": str(device),
        "status": status,
        "error_type": error_type,
        "error_message": error_message,
        **empty_breakdown_fields(),
        "outer_elapsed_ms": "",
        "samples_per_s": "",
        "it_per_s": "",
        "batches_per_s": "",
    }


def write_results_csv(
    output_csv: str | Path,
    results: list[dict[str, float | int | str]],
    args: argparse.Namespace,
    config: NestedNamespace,
    warmup_batches: int,
    compile_mode: str | None,
    num_params: int,
) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args_columns = flatten_for_csv(args, "arg")
    config_columns = flatten_for_csv(config, "config")

    fieldnames = [
        "batch_size",
        "num_samples",
        "num_batches",
        "status",
        "error_type",
        "error_message",
        "warmup_batches",
        "num_params",
        "batch_fetch_elapsed_ms",
        "transfer_elapsed_ms",
        "transfer_cuda_elapsed_ms",
        "normalizer_elapsed_ms",
        "normalizer_cuda_elapsed_ms",
        "transform_elapsed_ms",
        "transform_cuda_elapsed_ms",
        "forward_elapsed_ms",
        "forward_cuda_elapsed_ms",
        "total_elapsed_ms",
        "total_cuda_elapsed_ms",
        "outer_elapsed_ms",
        "outer_minus_total_ms",
        "outer_equals_total",
        "samples_per_s",
        "it_per_s",
        "batches_per_s",
        "batch_fetch_per_sample_ms",
        "transfer_per_sample_ms",
        "transfer_cuda_per_sample_ms",
        "normalizer_per_sample_ms",
        "normalizer_cuda_per_sample_ms",
        "transform_per_sample_ms",
        "transform_cuda_per_sample_ms",
        "forward_per_sample_ms",
        "forward_cuda_per_sample_ms",
        "total_cuda_per_sample_ms",
        "total_per_sample_ms",
        "device",
        "precision",
        "compile_mode",
    ] + list(args_columns.keys()) + list(config_columns.keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = dict(result)
            row.setdefault("status", "ok")
            row.setdefault("error_type", "")
            row.setdefault("error_message", "")
            row["warmup_batches"] = warmup_batches
            row.setdefault("num_params", num_params)
            row["precision"] = "fp32"
            row["compile_mode"] = compile_mode or "none"
            row.update(args_columns)
            row.update(config_columns)
            writer.writerow(row)


def print_result(result: dict[str, float | int | str]) -> None:
    if result.get("status") != "ok":
        print(
            f"batch_size={result['batch_size']:>4}  status={result['status']}  "
            f"{result['error_type']}: {result['error_message']}"
        )
        return
    if result["batch_fetch_elapsed_ms"] == "":
        outer_per_sample_ms = result["outer_elapsed_ms"] / result["num_samples"]
        print(
            f"batch_size={result['batch_size']:>4}  "
            f"outer={result['outer_elapsed_ms']:.3f}ms  "
            f"outer/sample={outer_per_sample_ms:.6f}ms  "
            f"samples/s={result['samples_per_s']:.2f}  "
            f"it/s={result['it_per_s']:.2f}  "
            f"batches={result['num_batches']}"
        )
        return
    msg = (
        f"batch_size={result['batch_size']:>4}  "
        f"fetch/sample={result['batch_fetch_per_sample_ms']:.3f}ms  "
        f"norm/sample={result['normalizer_per_sample_ms']:.3f}ms  "
        f"transform/sample={result['transform_per_sample_ms']:.3f}ms  "
        f"transfer/sample={result['transfer_per_sample_ms']:.3f}ms  "
        f"forward/sample={result['forward_per_sample_ms']:.3f}ms  "
        f"total={result['total_elapsed_ms']:.3f}ms  "
        f"outer={result['outer_elapsed_ms']:.3f}ms  "
        f"it/s={result['it_per_s']:.2f}  "
        f"batches={result['num_batches']}"
    )
    if result["total_cuda_elapsed_ms"] != "":
        msg += (
            f"  cuda_norm/sample={result['normalizer_cuda_per_sample_ms']:.3f}ms"
            f"  cuda_transform/sample={result['transform_cuda_per_sample_ms']:.3f}ms"
            f"  cuda_transfer/sample={result['transfer_cuda_per_sample_ms']:.3f}ms"
            f"  cuda_forward/sample={result['forward_cuda_per_sample_ms']:.3f}ms"
            f"  cuda_total/sample={result['total_cuda_per_sample_ms']:.3f}ms"
        )
    print(msg)


def main() -> None:
    args = parse_args()
    batch_sizes = parse_batch_sizes(args.batch_sizes)
    device = resolve_device(args.device)
    config = load_config(args.config)
    validate_task(config)

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.warmup_batches < 0:
        raise ValueError("--warmup-batches must be non-negative.")
    if args.preload_workers <= 0:
        raise ValueError("--preload-workers must be positive.")
    if args.max_preloaded_samples <= 0:
        raise ValueError("--max-preloaded-samples must be positive.")

    compile_mode = None if args.no_compile else args.compile

    print(f"Building model for {config.task.task_name}...")
    model, num_params = configure_model(config=config, device=device, compile_mode=compile_mode)
    print(f"Model parameters: {num_params:,}")
    transforms = get_task_transforms(config)

    preload_num_samples = min(args.num_samples, args.max_preloaded_samples)
    print(
        f"Preloading {preload_num_samples} processed samples into RAM "
        f"with {args.preload_workers} worker(s)..."
    )
    preload_started_at = time.perf_counter()
    raw_samples = preload_raw_samples(
        config,
        args.data_path,
        preload_num_samples,
        args.preload_workers,
    )
    preload_elapsed_s = time.perf_counter() - preload_started_at
    print(f"Preload completed in {preload_elapsed_s:.2f}s (excluded from timing).")
    if args.num_samples > preload_num_samples:
        print(
            f"Using virtual dataset length {args.num_samples} by reusing "
            f"{preload_num_samples} preloaded samples modulo dataset size."
        )
    normalizer = build_normalizer(config=config, raw_samples=raw_samples)

    results = []
    for batch_size in batch_sizes:
        num_batches = (args.num_samples + batch_size - 1) // batch_size
        try:
            print(f"\nRunning warmup for batch size {batch_size}...")
            warmup(
                raw_samples=raw_samples,
                num_samples=args.num_samples,
                batch_size=batch_size,
                warmup_batches=args.warmup_batches,
                model=model,
                normalizer=normalizer,
                transforms=transforms,
                device=device,
            )

            print(f"Benchmarking batch size {batch_size}...")
            result = benchmark_batch_size(
                raw_samples=raw_samples,
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
            clear_device_cache(device)
            status = "oom" if is_out_of_memory_error(exc) else "failed"
            result = make_failed_result(
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
        print_result(result)
        results.append(result)
        write_results_csv(
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
