# Reimplementation of Tiny Recursion Model (TRM)

## Structure
- `dataset/`: Scripts to build datasets (e.g., Sudoku).
- `models/`: The TRM model implementation, including layers, embeddings, and losses.
- `utils/`: Helper functions.
- `puzzle_dataset.py`: PyTorch IterableDataset implementation.
- `pretrain.py`: Main training script.

## Usage

1. **Build Dataset**:
   ```bash
   python3 -m reimplementation.dataset.build_sudoku --subsample_size 1000 --output_dir data/reimpl_sudoku
   ```

2. **Train**:
   ```bash
   python3 -m reimplementation.pretrain --data_dir data/reimpl_sudoku --epochs 10
   ```

## Notes
- This is a simplified reimplementation for educational purposes.
- Some distributed training features are simplified.
- Hydra configuration is replaced with `pydantic` + `argparse`.
