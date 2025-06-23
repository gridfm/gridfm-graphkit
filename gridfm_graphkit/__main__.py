import argparse
import torch
import mlflow
from gridfm_graphkit.cli import (
    main_standard,
    main_checkpoint,
    main_eval,
    main_fine_tuning,
)


def main():
    parser = argparse.ArgumentParser(
        prog="gridfm_graphkit",
        description="gridfm-graphkit CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- TRAIN SUBCOMMAND ----
    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument("--config", type=str, default=None)
    train_parser.add_argument("--grid", type=str, default=None)
    train_parser.add_argument("--exp", type=str, default=None)
    train_parser.add_argument("--data_path", type=str, default="data")
    train_parser.add_argument("-c", action="store_true", help="Start from checkpoint")
    train_parser.add_argument("--model_exp_id", type=str, default=None)
    train_parser.add_argument("--model_run_id", type=str, default=None)

    # ---- FINETUNE SUBCOMMAND ----
    train_parser = subparsers.add_parser("finetune", help="Run fine-tuning")
    train_parser.add_argument("--config", type=str, required=True)
    train_parser.add_argument("--model_path", type=str, required=True)
    train_parser.add_argument("--exp", type=str, default=None)
    train_parser.add_argument("--data_path", type=str, default="data")

    # ---- PREDICT SUBCOMMAND ----
    predict_parser = subparsers.add_parser("predict", help="Run prediction")
    predict_parser.add_argument("--model_path", type=str, default=None)
    predict_parser.add_argument("--config", type=str, required=True)
    predict_parser.add_argument("--eval_name", type=str, required=True)
    predict_parser.add_argument("--model_exp_id", type=str, default=None)
    predict_parser.add_argument("--model_run_id", type=str, default=None)
    predict_parser.add_argument("--model_name", type=str, default="best_model")
    predict_parser.add_argument("--data_path", type=str, default="data")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlflow.set_tracking_uri("file:mlruns")

    if args.command == "train":
        if args.c:
            main_checkpoint(args, device)
        else:
            main_standard(args, device)
    elif args.command == "predict":
        main_eval(args, device)
    elif args.command == "finetune":
        main_fine_tuning(args, device)


if __name__ == "__main__":
    main()
