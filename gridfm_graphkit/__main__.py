import argparse
from datetime import datetime
from gridfm_graphkit.cli import main_cli


def main():
    parser = argparse.ArgumentParser(
        prog="gridfm_graphkit",
        description="gridfm-graphkit CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ---- TRAIN SUBCOMMAND ----
    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument("--config", type=str, required=True)
    train_parser.add_argument("--exp_name", type=str, default=exp_name)
    train_parser.add_argument("--run_name", type=str, default="run")
    train_parser.add_argument("--log_dir", type=str, default="mlruns")
    train_parser.add_argument("--data_path", type=str, default="data")

    # ---- FINETUNE SUBCOMMAND ----
    finetune_parser = subparsers.add_parser("finetune", help="Run fine-tuning")
    finetune_parser.add_argument("--config", type=str, required=True)
    finetune_parser.add_argument("--model_path", type=str, required=True)
    finetune_parser.add_argument("--exp_name", type=str, default=exp_name)
    finetune_parser.add_argument("--run_name", type=str, default="run")
    finetune_parser.add_argument("--log_dir", type=str, default="mlruns")
    finetune_parser.add_argument("--data_path", type=str, default="data")

    # ---- EVALUATE SUBCOMMAND ----
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate model performance",
    )
    evaluate_parser.add_argument("--model_path", type=str, default=None)
    evaluate_parser.add_argument("--config", type=str, required=True)
    evaluate_parser.add_argument("--exp_name", type=str, default=exp_name)
    evaluate_parser.add_argument("--run_name", type=str, default="run")
    evaluate_parser.add_argument("--log_dir", type=str, default="mlruns")
    evaluate_parser.add_argument("--data_path", type=str, default="data")

    # ---- PREDICT SUBCOMMAND ----
    predict_parser = subparsers.add_parser("predict", help="Evaluate model performance")
    predict_parser.add_argument("--model_path", type=str, required=None)
    predict_parser.add_argument("--config", type=str, required=True)
    predict_parser.add_argument("--exp_name", type=str, default=exp_name)
    predict_parser.add_argument("--run_name", type=str, default="run")
    predict_parser.add_argument("--log_dir", type=str, default="mlruns")
    predict_parser.add_argument("--data_path", type=str, default="data")
    predict_parser.add_argument("--output_path", type=str, default="data")

    args = parser.parse_args()
    main_cli(args)


if __name__ == "__main__":
    main()
