#!/bin/bash

cd /u/albanpuech/gridfm_model_evaluation/scripts

# Define the path to the virtual environment and Python script
VENV_NAME="myenv"
PYTHON_SCRIPT="train.py"

# Check if grid configuration and additional config file are provided
GRID_CONFIG_FILE=""
CONFIG_FILE=""
EXP_NAME=""

if [ $# -eq 3 ]; then
    CONFIG_FILE=$1
    GRID_CONFIG_FILE=$2
    EXP_NAME=$3
fi

# Display GPU status
nvidia-smi

# Activate the virtual environment
conda init bash
echo "Activating virtual environment at $VENV_PATH"
conda activate $VENV_NAME

# Run the Python script
if [ -n "$GRID_CONFIG_FILE" ] && [ -n "$CONFIG_FILE" ] && [ -n "$EXP_NAME" ]; then
    echo "Running $PYTHON_SCRIPT with config: $CONFIG_FILE and additional grid configuration: $GRID_CONFIG_FILE"
    python "$PYTHON_SCRIPT" --config "config/$CONFIG_FILE" --grid "config/$GRID_CONFIG_FILE" --exp "$EXP_NAME"
else
    
    echo "Running $PYTHON_SCRIPT without grid or additional configuration"
    python "$PYTHON_SCRIPT"
fi

# Deactivate the virtual environment
conda deactivate
echo "Virtual environment deactivated."
