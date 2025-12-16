### From-scratch walkthrough (single Sudoku + minimal TRM)

This folder is a **small, readable “starting point”** for understanding and modifying the TRM idea in this repository.

It contains:
- A **minimal contract-matching TRM** (`TRMMin`) + loss head (`ACTLossHeadMin`) you can modify safely.
- A **single-Sudoku step-by-step runner** that uses the **original repo TRM** (`TinyRecursiveReasoningModel_ACTV1`) and prints what happens at each recursion step.

The goal is clarity and iteration speed, **not** reproduction of paper results.

---

### What the original repo does (high-level)

The main training entrypoint is `pretrain.py`.

At a high level:
- **Data**: `PuzzleDataset` (`puzzle_dataset.py`) yields batches shaped like:
  - `inputs`: token ids `[B, seq_len]`
  - `labels`: token ids `[B, seq_len]`
  - `puzzle_identifiers`: ids `[B]`
- **Model**: a TRM variant (e.g. `models/recursive_reasoning/trm.py`) exposes:
  - `initial_carry(batch)`
  - `forward(carry, batch) -> (new_carry, outputs)`
- **Loss head**: `models/losses.py::ACTLossHead` wraps the model and returns what `pretrain.py` expects:
  - `(carry, loss, metrics, detached_outputs, all_finish)`

Key mechanics in the original TRM (`TinyRecursiveReasoningModel_ACTV1`):
- Maintains latents `z_H` and `z_L` in a carry.
- Runs recursive reasoning for `H_cycles` outer cycles and `L_cycles` inner cycles.
- Uses a Q-head (`q_halt_logits` / `q_continue_logits`) to decide when to halt (ACT-like).
- Computes LM loss + halting loss.

---

### What this folder adds

#### 1) `run_one_sudoku.py`: step-by-step recursion on one Sudoku (using original TRM)

File: `from_scratch/run_one_sudoku.py`

What it does:
- Downloads **one** example from the Sudoku Extreme dataset (`sapientinc/sudoku-extreme`).
- Builds a single-item batch:
  - `inputs`: shape `[1, 81]`
  - `labels`: shape `[1, 81]`
  - `puzzle_identifiers`: shape `[1]` (always 0)
- Instantiates the repo’s **original TRM** class:
  - `models.recursive_reasoning.trm.TinyRecursiveReasoningModel_ACTV1`
- Runs forward passes **step-by-step** for `K` recursion steps and prints:
  - predicted 9×9 grid
  - per-step loss vs the solution
  - token accuracy and exact-match accuracy
  - halt logit and whether all sequences are halted

Token mapping (Sudoku datasets in this repo):
- `0` = PAD (shown as `_`)
- `1` = blank cell (shown as `.`)
- `2..10` = digits `1..9` (shown as `1..9`)

Run:

```bash
cd /workspace
python3 -m from_scratch.run_one_sudoku --K 6 --mlp-t --device cpu
```

Notes:
- With **random weights** (no checkpoint), predictions look like noise; this run is for understanding **the mechanics**: recursion, carry updates, logits→preds, halting.
- The script runs the model in `eval()` mode so the loop is deterministic and ends after exactly `K` steps.


#### 2) `trm_min.py`: minimal TRM-like model with the same interface

File: `from_scratch/trm_min.py`

Contains:
- `TRMMin`: a small TRM-like model that preserves the key interface:
  - `initial_carry(batch)`
  - `forward(carry, batch) -> (carry, outputs)`
- `ACTLossHeadMin`: a minimal wrapper that matches the **training-loop contract** used by `pretrain.py`:
  - returns `(carry, loss, metrics, detached_outputs, all_finish)`

This is meant to be the “sandbox” where you can simplify/modify:
- replace blocks (MLP → attention)
- remove halting
- change recurrence schedule
- change loss


#### 3) `puzzle_batch.py`: tiny batch utilities

File: `from_scratch/puzzle_batch.py`

Contains:
- `synthetic_batch(...)`: generate a shaped batch without any dataset
- `iter_first_real_batch(dataset_path, ...)`: load a *simple* first batch from `.npy` files

This is intentionally **much simpler** than the official `PuzzleDataset` sampler.


#### 4) `demo_one_step.py`: forward→loss→backward sanity run

File: `from_scratch/demo_one_step.py`

- Runs `TRMMin + ACTLossHeadMin` on either:
  - synthetic batch, or
  - first batch from a dataset directory
- Runs until halting, then performs one optimizer step.

Run:

```bash
cd /workspace
python3 -m from_scratch.demo_one_step --batch-size 4 --device cpu
```

---

### Differences vs the original TRM implementation (important)

This folder is **not** a reimplementation of the full repo training system. It is a comprehension + iteration aid.

#### Differences in `TRMMin` (minimal model)
- **Blocks**: uses simple feed-forward blocks (LayerNorm + Linear + GELU) instead of the repo’s `SwiGLU` + optional attention blocks.
- **Positional encoding**: none (the repo supports RoPE/learned).
- **Precision**:
  - CPU uses float32 to avoid bf16 dtype issues.
  - Repo TRM uses casted modules and often runs bf16 on GPU.
- **Puzzle embeddings**:
  - supported in `TRMMin` but default off.
  - repo has special sparse embedding machinery and optional SignSGD.
- **Halting**:
  - `TRMMin` uses a simplified halting policy (halt when `q_halt_logits > 0` or max steps).
  - repo includes optional Q-continue bootstrapping logic.
- **Metrics/loss**:
  - `ACTLossHeadMin` supports a basic softmax CE only.
  - repo supports stablemax + extra logging and Q-loss terms.

#### Differences in `run_one_sudoku.py`
- Uses the repo TRM model but:
  - runs on **one sample**
  - runs in **eval mode** with a fixed step count
  - prints human-readable grids after each recursion step

#### Differences in data loading
- `puzzle_dataset.py` in the repo:
  - packs puzzles into batches via group/puzzle indices
  - supports distributed ranks
  - maps ignore ids to `IGNORE_LABEL_ID = -100`
- `from_scratch/puzzle_batch.py`:
  - either synthetic, or reads a simple slice from `.npy`
  - no grouping/distributed sampling

---

### Recommended learning path (practical)

1) **Understand mechanics** (no training):
- Run `run_one_sudoku.py` and watch how outputs change per step.

2) **Confirm the training contract**:
- Run `demo_one_step.py` to see `(carry, loss, metrics, all_finish)` wiring.

3) **Simplify**:
- Start deleting pieces from `TRMMin` (e.g. remove halting) while keeping the interface stable.

4) **Only later**:
- Connect back to `pretrain.py` once you’re comfortable with the interface and batch format.
