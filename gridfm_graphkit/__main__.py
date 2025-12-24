from datetime import datetime
from gridfm_graphkit.cli import main_cli, iterate_cli
from jsonargparse import ArgumentParser, Namespace


from gridfm_graphkit.utils.types import (
    HyperParameterOptmizerSpec, TaskSpec, CallbackSpec,
    OptimizerSpec, ModelSpec, TrainingSpec, DataSpec
    )


def main():
    # parser = argparse.ArgumentParser(
    #     prog="gridfm_graphkit",
    #     description="gridfm-graphkit CLI",
    # )
    # subparsers = parser.add_subparsers(dest="command", required=True)
    exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ---- TRAIN SUBCOMMAND ----
    # train_parser = subparsers.add_parser("train", help="Run training")
    train_parser = ArgumentParser()
    train_parser.add_argument("--config", type=str, required=True)
    train_parser.add_argument("--exp_name", type=str, default=exp_name)
    train_parser.add_argument("--run_name", type=str, default="run")
    train_parser.add_argument("--log_dir", type=str, default="mlruns")
    train_parser.add_argument("--data_path", type=str, default="data")

    # ---- FINETUNE SUBCOMMAND ----
    # finetune_parser = subparsers.add_parser("finetune", help="Run fine-tuning")
    finetune_parser = ArgumentParser()
    finetune_parser.add_argument("--config", type=str, required=True)
    finetune_parser.add_argument("--model_path", type=str, required=True)
    finetune_parser.add_argument("--exp_name", type=str, default=exp_name)
    finetune_parser.add_argument("--run_name", type=str, default="run")
    finetune_parser.add_argument("--log_dir", type=str, default="mlruns")
    finetune_parser.add_argument("--data_path", type=str, default="data")

    # ---- EVALUATE SUBCOMMAND ----
    # evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate model performance")
    evaluate_parser = ArgumentParser()
    evaluate_parser.add_argument("--config", type=str, required=True)
    evaluate_parser.add_argument("--model_path", type=str, required=True)
    evaluate_parser.add_argument("--exp_name", type=str, default=exp_name)
    evaluate_parser.add_argument("--run_name", type=str, default="run")
    evaluate_parser.add_argument("--log_dir", type=str, default="mlruns")
    evaluate_parser.add_argument("--data_path", type=str, default="data")

    # ---- PREDICT SUBCOMMAND ----
    # predict_parser = subparsers.add_parser("predict", help="Evaluate model performance")
    predict_parser = ArgumentParser()
    predict_parser.add_argument("--model_path", type=str, required=None)
    predict_parser.add_argument("--config", type=str, required=True)
    predict_parser.add_argument("--exp_name", type=str, default=exp_name)
    predict_parser.add_argument("--run_name", type=str, default="run")
    predict_parser.add_argument("--log_dir", type=str, default="mlruns")
    predict_parser.add_argument("--data_path", type=str, default="data")
    predict_parser.add_argument("--output_path", type=str, default="data")

    # ---- ITERATE SUBCOMMAND ----
    # iterate_parser = subparsers.add_parser("iterate", help="Run model benchmarking")
    iterate_parser = ArgumentParser()
    iterate_parser.add_argument("--config", action="config")
    iterate_parser.add_argument("--seed", type=int) 
    iterate_parser.add_argument("--hpo_spec", type=HyperParameterOptmizerSpec)
    iterate_parser.add_argument("--tasks", type=list[TaskSpec])
    iterate_parser.add_argument("--model", type=ModelSpec)
    iterate_parser.add_argument("--optimizer", type=OptimizerSpec)
    iterate_parser.add_argument("--training", type=TrainingSpec)
    iterate_parser.add_argument("--callbacks", type=CallbackSpec)
    

    parser = ArgumentParser(
        prog="gridfm_graphkit",
        description="gridfm-graphkit CLI",
    )
    subcommands = parser.add_subcommands()
    subcommands.add_subcommand('train', train_parser)
    subcommands.add_subcommand('finetune', finetune_parser)
    subcommands.add_subcommand('evaluate', finetune_parser)
    subcommands.add_subcommand('predict', predict_parser)
    subcommands.add_subcommand('iterate', iterate_parser)

    args = parser.parse_args()
    if args.subcommand == "iterate":
        experiment_ids = iterate_cli(args.iterate)
        return experiment_ids
    else:
        main_cli(args)


if __name__ == "__main__":
    main()
