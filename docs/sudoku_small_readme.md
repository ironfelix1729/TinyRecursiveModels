## Small Sudoku TRM run (train + test) — and what the code is doing

This walkthrough shows how to run a **small** TRM training job on Sudoku and then read the **test** results. It also explains (in plain language) how the repo handles the fact that **some puzzles can be “solved” in fewer refinement steps than others**.

This is meant for understanding the training algorithm, not for reaching the paper’s best accuracy.

---

## 0) Prerequisites

- **GPU required**: `pretrain.py` constructs the model on CUDA (`torch.device("cuda")`).
- Python deps:

```bash
pip install -r requirements.txt
pip install --no-cache-dir --no-build-isolation adam-atan2
```

If `adam-atan2` fails to import on Colab (common on Python 3.12), the training script will automatically fall back to `torch.optim.AdamW` so you can still run the demo. (This is fine for understanding the training loop; it may change final accuracy.)

- Optional: disable Weights & Biases logging (recommended for a quick local run):

```bash
export WANDB_MODE=disabled
```

---

## 1) Build a tiny Sudoku dataset

This repo includes `dataset/build_sudoku_dataset.py`, which downloads Sudoku-Extreme from HuggingFace and writes a standardized on-disk format used by `PuzzleDataset`.

Run:

```bash
bash scripts/build_sudoku_small.sh
```

It will create `data/sudoku-small/{train,test}/...` with:

- `all__inputs.npy`: tokenized Sudoku boards (questions)
- `all__labels.npy`: tokenized solutions (answers)
- `all__puzzle_identifiers.npy`: per-puzzle identifier (Sudoku uses a single "<blank>" id)
- `all__puzzle_indices.npy`, `all__group_indices.npy`: indexing structures for batching
- `dataset.json`: metadata (`seq_len`, `vocab_size`, etc.)

---

## 2) Train and evaluate (small demo run)

Run:

```bash
bash scripts/train_sudoku_small.sh
```

If you *don’t* have a GPU, you can try CPU mode (much slower):

```bash
bash scripts/train_sudoku_small.sh cpu
```

This uses the Hydra config `config/cfg_sudoku_small.yaml`. During training it will periodically print evaluation progress like:

- `Processing batch ...`
- `Completed inference in N steps`

And it will log (at least) these metrics (computed in `models/losses.py`):

- `accuracy`: average token accuracy (masked/padded positions excluded)
- `exact_accuracy`: fraction of examples where **all** output tokens match the label
- `steps`: number of refinement steps used (summed over evaluated examples)

For exact counts (how many puzzles/examples) and the precise tensor shapes/hyperparameters used by the demo, see:

- `docs/sudoku_small_details.md`

Checkpoints are written under:

- `checkpoints/<project>/<run_name>/step_<k>`

(`project`/`run_name` are set in `pretrain.py`; the script sets `+run_name="sudoku-small-demo"`.)

---

## 3) What “variable solve time in a batch” means here

TRM in this repo is trained as a **refinement model**:

- Input: `inputs` (a Sudoku board)
- Target: `labels` (the solved board)
- The model produces a full predicted solution grid in parallel (not autoregressive)
- It can run for multiple **refinement steps**

But different puzzles may become correct at different refinement steps. The repo handles that with a **streaming batch**.

### 3.1 The streaming batch: persistent “slots”

Instead of treating a batch as “64 independent examples that reset every step”, the model keeps a persistent `carry` across optimizer steps. Conceptually:

- You have `B` **slots** (equal to batch size).
- Each slot holds:
  - latent state (`z_H`, `z_L`)
  - the current puzzle data (`current_data`)
  - a step counter (`steps`)
  - a boolean flag (`halted`)

This is the dataclass `TinyRecursiveReasoningModel_ACTV1Carry` in `models/recursive_reasoning/trm.py`.

### 3.2 Refilling only the finished slots

At each optimizer step, the dataloader yields a fresh batch `batch`.

The model then does:

- If a slot is `halted=True`: **load the new puzzle** from `batch` into that slot and reset its latent state.
- If a slot is `halted=False`: **keep the old puzzle** and ignore the new `batch` item for that slot.

That “refill only halted slots” merge is implemented with `torch.where(...)` over the batch tensors.

Result: even though the outer training loop always runs fixed-size batches, *internally* each slot can spend a variable number of refinement steps on its current puzzle before it gets replaced with a new one.

### 3.3 How a slot decides to halt (during training)

The TRM wrapper produces:

- `logits`: token predictions for the full output grid
- `q_halt_logits`: a learned scalar “should I stop refining?”

During training, a slot halts if:

- it hits `halt_max_steps`, or
- the model predicts halt (by default config: `q_halt_logits > 0`), plus some exploration noise

During evaluation, the model does **not** use early halting; it runs a fixed number of steps (`halt_max_steps`) so batching is deterministic.

---

## 4) What is TRM actually doing inside one refinement step?

Inside `models/recursive_reasoning/trm.py`, the “inner” network keeps two latent tensors per slot:

- `z_L`: “local” latent
- `z_H`: “global” latent

Each forward pass does a nested recurrence controlled by:

- `L_cycles`: how many times to update `z_L`
- `H_cycles`: how many times to alternate and update `z_H`

To reduce memory/compute, only the **final** `H` cycle is differentiated; earlier cycles run under `torch.no_grad()` (truncated backprop through the internal recurrence).

---

## 5) What losses are used?

The loss head is `ACTLossHead` in `models/losses.py`:

1) **Token loss (`lm_loss`)**
- masked cross entropy between `logits` and `labels`
- padding positions are ignored (label `-100`)

2) **Halting loss (`q_halt_loss`)**
- Compute `seq_is_correct`: whether the entire predicted output sequence is correct (all tokens match)
- Train `q_halt_logits` with binary cross entropy to predict `seq_is_correct`

Total loss is:

\[
\text{loss} = \text{lm\_loss} + 0.5 \cdot \text{q\_halt\_loss}
\]

---

## 6) Files to read (if you want to trace it yourself)

- **Outer training loop**: `pretrain.py`
- **Dataset/batching**: `puzzle_dataset.py`
- **Sudoku dataset builder**: `dataset/build_sudoku_dataset.py`
- **TRM model + streaming/halting**: `models/recursive_reasoning/trm.py`
- **Loss + metrics**: `models/losses.py`
- **Small demo config**: `config/experiments/sudoku_small.yaml`
