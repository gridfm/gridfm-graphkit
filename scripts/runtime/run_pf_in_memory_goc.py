#!/usr/bin/env python3
"""Run PF inference benchmark matrix on large GOC cases.

Sample counts match gridfm-datakit pure-Julia matrix_config.jl (large scope
for case2000/case10000; case500 uses its small-scope count).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_SCRIPT = SCRIPT_DIR / "benchmark_pf_inference.py"
BASE_CONFIG = SCRIPT_DIR / "config_pf_base.yaml"
CONFIG_DIR = SCRIPT_DIR / "configs/pf_in_memory_goc"
OUTPUT_DIR = SCRIPT_DIR / "genco_pf_in_memory/goc"
MAX_PRELOADED_SAMPLES = 10_000
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    args, extra_args = parser.parse_known_args()
    data_path = args.data_path

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    for network, num_samples in NETWORK_SPECS:
        for model_name, hidden_size in MODEL_SPECS:
            config_path = CONFIG_DIR / f"config_{network}_{model_name}.yaml"
            output_csv = OUTPUT_DIR / f"benchmark_{network}_{model_name}.csv"

            write_config(network=network, hidden_size=hidden_size, config_path=config_path)

            print()
            print(
                f"=== Running {network} / {model_name} "
                f"(hidden_size={hidden_size}, num_samples={num_samples:,}) ==="
            )

            command = [
                sys.executable,
                str(BENCHMARK_SCRIPT),
                "--config",
                str(config_path),
                "--data-path",
                data_path,
                "--output-csv",
                str(output_csv),
                "--num-samples",
                str(num_samples),
                "--max-preloaded-samples",
                str(MAX_PRELOADED_SAMPLES),
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

    print()
    print("Benchmark matrix complete.")
    print(f"Configs written to {CONFIG_DIR}")
    print(f"CSVs written to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
