#!/usr/bin/env bash
set -euo pipefail

# Builds a tiny Sudoku dataset (train/test) for quick TRM experiments.
# Downloads the source dataset from HuggingFace via `huggingface_hub`.

OUT_DIR="${1:-data/sudoku-small}"

python3 dataset/build_sudoku_dataset.py \
  --output-dir "${OUT_DIR}" \
  --subsample-size 128 \
  --test-subsample-size 128 \
  --num-aug 8

echo ""
echo "Built dataset at: ${OUT_DIR}"
echo "Train split: ${OUT_DIR}/train"
echo "Test  split: ${OUT_DIR}/test"
