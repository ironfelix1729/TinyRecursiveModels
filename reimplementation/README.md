# Tiny Recursion Model (TRM) - Clean Reimplementation

This repository contains a clean, from-scratch reimplementation of the paper **"Less is More: Recursive Reasoning with Tiny Networks"**. 

The Tiny Recursion Model (TRM) challenges the notion that massive models are required for complex reasoning. Instead, it uses a tiny neural network (~7M parameters) that recursively updates its internal state ("latent thought") to improve its answer over multiple steps.

## 🧠 Core Concept: Recursive Reasoning

TRM improves its predictions by iterating on them. Unlike a standard Transformer that processes input once to produce output, TRM works in cycles:

1.  **Input**: The puzzle `x` is embedded.
2.  **Latent State (`z`)**: The model maintains a latent state `z` and a candidate answer `y`.
3.  **Recursive Update (L-cycles)**: The latent state is updated `L` times through the reasoning module (Transformer or MLP) without changing the candidate answer.
4.  **Answer Update (H-cycles)**: After `L` latent updates, the candidate answer `y` is updated based on the current latent state.
5.  **Output**: The final state produces the solution.

This structure allows the model to "think" for longer on harder problems without increasing the number of parameters.

## 📂 Repository Structure

### 1. `models/` - The Neural Network
This directory contains the PyTorch implementation of the architecture.

- **`trm.py`**: The heart of the project.
    - `TinyRecursiveReasoningModel_ACTV1_Inner`: Implements the recursive loop (H-cycles and L-cycles).
    - `TinyRecursiveReasoningModel_ACTV1`: Wraps the inner model with the Adaptive Computation Time (ACT) logic (halting mechanism).
- **`layers.py`**: Custom implementations of Transformer components:
    - **Rotary Embeddings (RoPE)**: For positional encoding.
    - **SwiGLU**: The activation function used in the feed-forward blocks.
    - **RMSNorm**: Root Mean Square Normalization.
    - **Attention**: Standard Multi-Head Attention optimized for this architecture.
- **`losses.py`**:
    - **ACTLossHead**: Computes the language modeling loss (Cross Entropy) and the Halting loss (Q-learning based) to teach the model when to stop thinking.
- **`sparse_embedding.py`**:
    - Handles specialized sparse embeddings for puzzle identifiers, crucial for preventing overfitting when training on specific puzzle instances.

### 2. `dataset/` - Data Generation
- **`build_sudoku.py`**: Downloads the "Sudoku-Extreme" dataset from HuggingFace, applies Sudoku-preserving augmentations (digit permutations, band/stack shuffling), and saves it in a memory-mapped NumPy format for efficient training.

### 3. `puzzle_dataset.py` - Data Loading
- Implements a PyTorch `IterableDataset`.
- Unlike standard map-style datasets, this loader handles the specific "Grouped" structure of the training data, where we shuffle groups of augmentations rather than individual samples to maintain training stability.

### 4. `pretrain.py` - Training Loop
- The main entry point.
- Initializes the model, optimizer (AdamW or Adam-Atan2), and the training loop.
- Handles **EMA (Exponential Moving Average)** updates, which are critical for the stability of recursive models.

## ⚖️ Differences from Original Implementation

This reimplementation focuses on clarity and educational value while maintaining the core mathematical logic.

| Feature | Original Implementation | This Reimplementation |
|---------|------------------------|-----------------------|
| **Configuration** | Uses **Hydra** & OmegaConf. Configs are spread across multiple YAML files and composed at runtime. | Uses **Pydantic** & **Argparse**. Configuration is explicit, type-checked, and contained within `pretrain.py` for easy reading. |
| **Distributed Training** | Complex `torchrun` setup with custom process groups and specialized scatter/gather operations for metrics. | Simplified `DistributedDataParallel` support (or single GPU). Retains the custom distributed optimizer logic for sparse embeddings but simplifies the loop. |
| **Datasets** | Supports ARC-AGI, Maze, and Sudoku with complex selection logic. | Focuses on **Sudoku** as the primary example to demonstrate the recursive reasoning capabilities. |
| **Logging** | Deep integration with WandB and custom console loggers. | Simplified WandB logging and standard `tqdm` progress bars. |
| **Dependencies** | Requires `hydra-core`, `coolname`, etc. | Minimal dependencies (`torch`, `numpy`, `pydantic`). |

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ and PyTorch installed.

```bash
pip install torch numpy tqdm pydantic huggingface_hub einops wandb adam-atan2
```
*Note: `adam-atan2` is the optimizer used in the paper, but the code falls back to `AdamW` if it's not installed.*

### 2. Generate the Dataset
We use the Sudoku-Extreme dataset. The script will download it and preprocess it.

```bash
# Generate a small dataset for testing (1000 original puzzles + augmentations)
python3 -m reimplementation.dataset.build_sudoku \
    --subsample_size 1000 \
    --output_dir data/reimpl_sudoku
```

### 3. Train the Model
Run the pretraining script.

```bash
python3 -m reimplementation.pretrain \
    --data_dir data/reimpl_sudoku \
    --epochs 50
```

### 4. Understanding the Output
During training, watch for:
- **`lm_loss`**: The language modeling loss (should decrease).
- **`accuracy`**: The next-token prediction accuracy.
- **`exact_accuracy`**: The percentage of puzzles perfectly solved.

Because the model uses recursive reasoning, you might see `accuracy` stagnate while `exact_accuracy` jumps later as the model learns to effectively use its "thinking time" (L-cycles).
