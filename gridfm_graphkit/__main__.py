import argparse
from datetime import datetime
from gridfm_graphkit.cli import main_cli, benchmark_cli


def main():
    parser = argparse.ArgumentParser(
        prog="gridfm_graphkit",
        description="gridfm-graphkit CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    _compile_kwargs = dict(
        type=str,
        default=None,
        nargs="?",
        const="default",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        help="Enable torch.compile with the given mode (omit value for 'default').",
    )
    _bfloat16_kwargs = dict(
        action="store_true",
        default=False,
        help="Cast model to bfloat16 (model.to(torch.bfloat16)).",
    )
    _tf32_kwargs = dict(
        action="store_true",
        default=False,
        help="Enable TF32 on Ampere+ GPUs via torch.set_float32_matmul_precision('high').",
    )

    # ---- TRAIN SUBCOMMAND ----
    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument("--config", type=str, required=True)
    train_parser.add_argument("--exp_name", type=str, default=exp_name)
    train_parser.add_argument("--run_name", type=str, default="run")
    train_parser.add_argument("--log_dir", type=str, default="mlruns")
    train_parser.add_argument("--data_path", type=str, default="data")
    train_parser.add_argument("--compile", **_compile_kwargs)
    train_parser.add_argument("--bfloat16", **_bfloat16_kwargs)
    train_parser.add_argument("--tf32", **_tf32_kwargs)
    train_parser.add_argument(
        "--dataset_wrapper",
        type=str,
        default=None,
        help="Registered name of a dataset wrapper (see DATASET_WRAPPER_REGISTRY), e.g. SharedMemoryCacheDataset",
    )
    train_parser.add_argument(
        "--plugins",
        nargs="*",
        default=[],
        help="Python packages to import for plugin registration, e.g. gridfm_graphkit_ee",
    )
    train_parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override data.workers from the YAML config. Use 0 to debug worker crashes.",
    )
    train_parser.add_argument(
        "--dataset_wrapper_cache_dir",
        type=str,
        default=None,
        help="Directory for the dataset wrapper's disk cache. If set, cache is loaded from here when present and saved here after first population.",
    )
    train_parser.add_argument(
        "--profiler",
        type=str,
        default=None,
        choices=["simple", "advanced", "pytorch"],
        help="Enable Lightning profiler: 'simple', 'advanced', or 'pytorch'.",
    )
    train_parser.add_argument(
        "--report-performance",
        dest="report_performance",
        action="store_true",
        help="Print the last training epoch time and a single test metric to stdout.",
    )
    finetune_parser = subparsers.add_parser("finetune", help="Run fine-tuning")
    finetune_parser.add_argument("--config", type=str, required=True)
    finetune_parser.add_argument("--model_path", type=str, required=True)
    finetune_parser.add_argument("--exp_name", type=str, default=exp_name)
    finetune_parser.add_argument("--run_name", type=str, default="run")
    finetune_parser.add_argument("--log_dir", type=str, default="mlruns")
    finetune_parser.add_argument("--data_path", type=str, default="data")
    finetune_parser.add_argument("--compile", **_compile_kwargs)
    finetune_parser.add_argument("--bfloat16", **_bfloat16_kwargs)
    finetune_parser.add_argument("--tf32", **_tf32_kwargs)
    finetune_parser.add_argument(
        "--dataset_wrapper",
        type=str,
        default=None,
        help="Registered name of a dataset wrapper (see DATASET_WRAPPER_REGISTRY), e.g. SharedMemoryCacheDataset",
    )
    finetune_parser.add_argument(
        "--plugins",
        nargs="*",
        default=[],
        help="Python packages to import for plugin registration, e.g. gridfm_graphkit_ee",
    )
    finetune_parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override data.workers from the YAML config. Use 0 to debug worker crashes.",
    )
    finetune_parser.add_argument(
        "--dataset_wrapper_cache_dir",
        type=str,
        default=None,
        help="Directory for the dataset wrapper's disk cache. If set, cache is loaded from here when present and saved here after first population.",
    )
    finetune_parser.add_argument(
        "--profiler",
        type=str,
        default=None,
        choices=["simple", "advanced", "pytorch"],
        help="Enable Lightning profiler: 'simple', 'advanced', or 'pytorch'.",
    )
    finetune_parser.add_argument(
        "--report-performance",
        dest="report_performance",
        action="store_true",
        help="Print the last training epoch time and a single test metric to stdout.",
    )

    # ---- EVALUATE SUBCOMMAND ----
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate model performance",
    )
    evaluate_parser.add_argument("--model_path", type=str, default=None)
    evaluate_parser.add_argument(
        "--normalizer_stats",
        type=str,
        default=None,
        help="Path to normalizer_stats.pt from a training run.",
    )
    evaluate_parser.add_argument("--config", type=str, required=True)
    evaluate_parser.add_argument("--exp_name", type=str, default=exp_name)
    evaluate_parser.add_argument("--run_name", type=str, default="run")
    evaluate_parser.add_argument("--log_dir", type=str, default="mlruns")
    evaluate_parser.add_argument("--data_path", type=str, default="data")
    evaluate_parser.add_argument("--compile", **_compile_kwargs)
    evaluate_parser.add_argument("--bfloat16", **_bfloat16_kwargs)
    evaluate_parser.add_argument("--tf32", **_tf32_kwargs)
    evaluate_parser.add_argument(
        "--dataset_wrapper",
        type=str,
        default=None,
        help="Registered name of a dataset wrapper (see DATASET_WRAPPER_REGISTRY), e.g. SharedMemoryCacheDataset",
    )
    evaluate_parser.add_argument(
        "--plugins",
        nargs="*",
        default=[],
        help="Python packages to import for plugin registration, e.g. gridfm_graphkit_ee",
    )
    evaluate_parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override data.workers from the YAML config. Use 0 to debug worker crashes.",
    )
    evaluate_parser.add_argument(
        "--dataset_wrapper_cache_dir",
        type=str,
        default=None,
        help="Directory for the dataset wrapper's disk cache. If set, cache is loaded from here when present and saved here after first population.",
    )
    evaluate_parser.add_argument(
        "--profiler",
        type=str,
        default=None,
        choices=["simple", "advanced", "pytorch"],
        help="Enable Lightning profiler: 'simple', 'advanced', or 'pytorch'.",
    )
    evaluate_parser.add_argument(
        "--compute_dc_ac_metrics",
        action="store_true",
    )
    evaluate_parser.add_argument(
        "--save_output",
        action="store_true",
    )

    # ---- PREDICT SUBCOMMAND ----
    predict_parser = subparsers.add_parser("predict", help="Run prediction")
    predict_parser.add_argument("--model_path", type=str, required=False)
    predict_parser.add_argument("--normalizer_stats", type=str, default=None)
    predict_parser.add_argument("--config", type=str, required=True)
    predict_parser.add_argument("--exp_name", type=str, default=exp_name)
    predict_parser.add_argument("--run_name", type=str, default="run")
    predict_parser.add_argument("--log_dir", type=str, default="mlruns")
    predict_parser.add_argument("--data_path", type=str, default="data")
    predict_parser.add_argument(
        "--dataset_wrapper",
        type=str,
        default=None,
        help="Registered name of a dataset wrapper (see DATASET_WRAPPER_REGISTRY), e.g. SharedMemoryCacheDataset",
    )
    predict_parser.add_argument(
        "--plugins",
        nargs="*",
        default=[],
        help="Python packages to import for plugin registration, e.g. gridfm_graphkit_ee",
    )
    predict_parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override data.workers from the YAML config. Use 0 to debug worker crashes.",
    )
    predict_parser.add_argument(
        "--dataset_wrapper_cache_dir",
        type=str,
        default=None,
        help="Directory for the dataset wrapper's disk cache. If set, cache is loaded from here when present and saved here after first population.",
    )
    predict_parser.add_argument("--output_path", type=str, default="data")
    predict_parser.add_argument("--compile", **_compile_kwargs)
    predict_parser.add_argument("--bfloat16", **_bfloat16_kwargs)
    predict_parser.add_argument("--tf32", **_tf32_kwargs)
    predict_parser.add_argument(
        "--profiler",
        type=str,
        default=None,
        choices=["simple", "advanced", "pytorch"],
    )

    # ---- BENCHMARK SUBCOMMAND ----
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark train-dataloader iteration speed",
    )
    benchmark_parser.add_argument("--config", type=str, required=True)
    benchmark_parser.add_argument("--data_path", type=str, default="data")
    benchmark_parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs to iterate through the train dataloader.",
    )
    benchmark_parser.add_argument(
        "--dataset_wrapper",
        type=str,
        default=None,
        help="Registered name of a dataset wrapper (see DATASET_WRAPPER_REGISTRY), e.g. SharedMemoryCacheDataset",
    )
    benchmark_parser.add_argument(
        "--dataset_wrapper_cache_dir",
        type=str,
        default=None,
        help="Directory for the dataset wrapper's disk cache.",
    )
    benchmark_parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override data.workers from the YAML config.",
    )
    benchmark_parser.add_argument(
        "--plugins",
        nargs="*",
        default=[],
        help="Python packages to import for plugin registration.",
    )

    args = parser.parse_args()

    if args.command == "benchmark":
        benchmark_cli(args)
    else:
        main_cli(args)


if __name__ == "__main__":
    main()
