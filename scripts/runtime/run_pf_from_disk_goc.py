#!/usr/bin/env python3
"""Run PF inference benchmark matrix on large GOC cases (disk load in get).

Sample counts match gridfm-datakit pure-Julia matrix_config.jl (large scope
for case2000/case10000; case500 uses its small-scope count).
For each network: copy the first 10,000 processed samples to tmp, benchmark all
models/batch sizes while cycling through them, then delete tmp.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from benchmark_tmp_dataset import cleanup_local_tmp_dataset, prepare_local_tmp_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_SCRIPT = SCRIPT_DIR / "benchmark_pf_inference_load_in_get.py"
BASE_CONFIG = SCRIPT_DIR / "config_pf_base.yaml"
CONFIG_DIR = SCRIPT_DIR / "configs/pf_from_disk_goc"
OUTPUT_DIR = SCRIPT_DIR / "genco_pf_from_disk/goc"
MATERIALIZED_SAMPLES = 10_000
BATCH_SIZES = "16,32,64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384"
DEVICE = "cuda"
WARMUP_BATCHES = 20
PRELOAD_WORKERS = 32
COMPILE_MODE = "reduce-overhead"

# (network, num_samples) from matrix_config.jl
NETWORK_SPECS = [
    ("case500_goc", 500_000),
    ("case2000_goc", 50_000),
    ("case10000_goc", 10_000),
]

MODEL_SPECS = [
    ("tiny", 12),
    ("small", 24),
    ("base", 48),
]


def write_config(network: str, hidden_size: int, config_path: Path) -> None:
    with BASE_CONFIG.open("r") as f:
        config = yaml.safe_load(f)

    config["data"]["networks"] = [network]
    config["model"]["hidden_size"] = hidden_size

    with config_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def run_benchmark(
    *,
    config_path: Path,
    data_path: Path,
    output_csv: Path,
    num_samples: int,
    extra_args: list[str],
) -> None:
    command = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--config",
        str(config_path),
        "--data-path",
        str(data_path),
        "--output-csv",
        str(output_csv),
        "--num-samples",
        str(num_samples),
        "--max-preloaded-samples",
        str(MATERIALIZED_SAMPLES),
        "--batch-sizes",
        BATCH_SIZES,
        "--device",
        DEVICE,
        "--warmup-batches",
        str(WARMUP_BATCHES),
        "--preload-workers",
        str(PRELOAD_WORKERS),
        "--compile",
        COMPILE_MODE,
        *extra_args,
    ]
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    args, extra_args = parser.parse_known_args()
    data_path = args.data_path

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    for network, num_samples in NETWORK_SPECS:
        print()
        print(f"=== Preparing tmp dataset for {network} ===")
        local_data_root = prepare_local_tmp_dataset(
            network=network,
            source_data_root=data_path,
            max_samples=MATERIALIZED_SAMPLES,
        )

        try:
            for model_name, hidden_size in MODEL_SPECS:
                config_path = CONFIG_DIR / f"config_{network}_{model_name}.yaml"
                output_csv = OUTPUT_DIR / f"benchmark_{network}_{model_name}.csv"

                write_config(network=network, hidden_size=hidden_size, config_path=config_path)

                print()
                print(
                    f"=== Running {network} / {model_name} "
                    f"(hidden_size={hidden_size}, num_samples={num_samples:,}, "
                    f"materialized={MATERIALIZED_SAMPLES:,}) ==="
                )

                run_benchmark(
                    config_path=config_path,
                    data_path=local_data_root,
                    output_csv=output_csv,
                    num_samples=num_samples,
                    extra_args=extra_args,
                )
        finally:
            cleanup_local_tmp_dataset(network)

    print()
    print("Benchmark matrix complete.")
    print(f"Configs written to {CONFIG_DIR}")
    print(f"CSVs written to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
