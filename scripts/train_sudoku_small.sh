#!/usr/bin/env bash
set -euo pipefail

# Trains TRM on the small Sudoku dataset and evaluates on the test split.
#
# Notes:
# - This repo assumes CUDA; `pretrain.py` constructs the model on "cuda".
# - W&B is optional. Set WANDB_MODE=disabled to avoid login requirements.

export WANDB_MODE="${WANDB_MODE:-disabled}"

DEVICE="${1:-cuda}"

python3 pretrain.py \
  --config-path config/experiments \
  --config-name sudoku_small \
  device="${DEVICE}" \
  +run_name="sudoku-small-demo"
