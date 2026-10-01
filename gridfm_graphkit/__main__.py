import argparse
from datetime import datetime
from gridfm_graphkit.cli import main_cli

import subprocess
import os


def is_lsf():
    return (
        os.environ.get("LSB_JOBID") is not None
        and os.environ.get("LSB_MCPU_HOSTS") is not None
        and "LSF_ENVDIR" in os.environ  # strong LSF indicator
    )


def fix_infiniband():
    ibv = subprocess.run("ibv_devinfo", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = ibv.stdout.decode("utf-8").split("\n")
    exclude = ""
    for line in lines:
        if "hca_id:" in line:
            name = line.split(":")[1].strip()
        if "\tport:" in line:
            port = line.split(":")[1].strip()
        if "link_layer:" in line and "Ethernet" in line:
            exclude = exclude + f"{name}:{port},"

    if exclude:
        exclude = "^" + exclude[:-1]
        os.environ["NCCL_IB_HCA"] = exclude


def set_env():
    # print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")
    LSB_MCPU_HOSTS = os.environ[
        "LSB_MCPU_HOSTS"
    ].split(
        " ",
    )  # Parses Node list set by LSF, in format hostname proceeded by number of cores requested
    HOST_LIST = LSB_MCPU_HOSTS[::2]  # Strips the cores per node items in the list
    LSB_JOBID = os.environ[
        "LSB_JOBID"
    ]  # Parses Node list set by LSF, in format hostname proceeded by number of cores requested
    os.environ["MASTER_ADDR"] = HOST_LIST[
        0
    ]  # Sets the MasterNode to thefirst node on the list of hosts
    os.environ["MASTER_PORT"] = "5" + LSB_JOBID[-5:-1]
    os.environ["NODE_RANK"] = str(
        HOST_LIST.index(os.environ["HOSTNAME"]),
    )  # Uses the list index for node rank, master node rank must be 0
    os.environ["NCCL_SOCKET_IFNAME"] = (
        "ib,bond"  # avoids using docker of loopback interface
    )
    os.environ["NCCL_IB_CUDA_SUPPORT"] = "1"  # Force use of infiniband


def main():
    if is_lsf():
        print("Using LSF")
        set_env()
        fix_infiniband()
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
