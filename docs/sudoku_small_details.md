## Sudoku-small: exact counts, hyperparameters, tensor shapes, and implementation notes

This document complements `docs/sudoku_small_readme.md` with **exact numbers** and “where the code is more specific than the paper”.

---

## 1) How many puzzles/examples are in the Sudoku-small dataset?

The dataset is built by `scripts/build_sudoku_small.sh`, which runs:

- `dataset/build_sudoku_dataset.py --subsample-size 128 --num-aug 8 --test-subsample-size 128`

### Train split

In `dataset/build_sudoku_dataset.py`, **each augmented board is treated as a separate “puzzle” with exactly 1 example**.

- **Original (subsampled) train boards**: 128
- **Augments per original**: 8
- **Total train puzzles**: \(128 \times (1+8) = 1152\)
- **Total train examples**: 1152 (because each puzzle has one example)
- **Groups**: 128 (each group corresponds to one original board + its augments)

### Test split

By default Sudoku-Extreme has a fixed test set size from HuggingFace; for the demo we additionally subsample:

- **Test puzzles/examples**: 128 (due to `--test-subsample-size 128`)

---

## 2) Training hyperparameters used by the demo

From `config/cfg_sudoku_small.yaml`:

- **global_batch_size**: 64
- **epochs**: 2000
- **eval_interval**: 200 (evaluation happens every 200 “epochs” in the dataset’s grouping scheme)
- **EMA**: enabled (`ema: True`, `ema_rate: 0.999`)
- **optimizer**:
  - `AdamATan2` if available, otherwise **fallback to `torch.optim.AdamW`** (Colab/Python 3.12 commonly uses the fallback)
- **lr**: 1e-4
- **weight_decay**: 1.0
- **warmup**: 100 “steps” (internal scheduler steps, not epochs)

---

## 3) TRM refinement / recursion settings used by the demo

From `config/cfg_sudoku_small.yaml` + `config/arch/trm.yaml`:

- **halt_max_steps** (refinement steps per puzzle during evaluation): 8
- **halt_exploration_prob** (training-only exploration): 0.10

Inner recursion per refinement step:

- **H_cycles**: 3
- **L_cycles**: 6
- **L_layers**: 2
- **mlp_t**: True (uses an MLP-style sequence mixer for the “L block” instead of attention)

Important compute detail (implementation):

- The model runs **`H_cycles - 1` cycles under `torch.no_grad()`** and only the final `H` cycle with gradients (truncated backprop through the inner recurrence).

---

## 4) Tokenization / vocab / sequence length (Sudoku)

From the Sudoku builder:

- **seq_len**: 81 (9×9 flattened)
- **pad_id**: 0
- **ignore_label_id**: 0 (converted to `-100` for loss masking)
- **vocab_size**: 11 (PAD + digits “0..9” shifted by +1)

---

## 5) Key tensor shapes (Sudoku-small demo)

Let:

- \(B =\) local batch size (single GPU demo: \(B = 64\))
- \(L =\) `seq_len` = 81
- \(D =\) `hidden_size` = 512

### Puzzle embedding prefix length

The model uses a learned “puzzle embedding” prefix if `puzzle_emb_ndim > 0`.

For TRM defaults:

- `puzzle_emb_ndim = hidden_size = 512`
- `puzzle_emb_len = 16` (from `config/arch/trm.yaml`)

So the effective transformer sequence length is:

- \(L_\text{total} = L + \text{puzzle\_emb\_len} = 81 + 16 = 97\)

### Latent states

Per batch slot:

- **`z_H`**: `[B, 97, 512]`
- **`z_L`**: `[B, 97, 512]`

### Output logits

The output head predicts only the “real” token positions (the prefix is sliced off):

- **`logits`**: `[B, 81, vocab_size]` = `[64, 81, 11]`
- **`preds`** (argmax): `[B, 81]`

### Halting head

The halting head reads (by default) the first prefix position:

- **`q_halt_logits`**: `[B]`

---

## 6) Places where this repo is more specific than the TRM paper (implementation notes)

I can’t guarantee what the paper text does/doesn’t emphasize without quoting it, but these are **concrete implementation choices** in the official code that are easy to miss if you only read the high-level description:

- **Streaming batch / slot reuse during training**:
  - Training keeps a persistent `carry` across optimizer steps.
  - If a slot halts, that slot is refilled with a new example and its `z_H/z_L` are reset.
  - If a slot does not halt, it keeps refining the same puzzle on the next optimizer step (the incoming dataloader sample for that slot is ignored).

- **Evaluation uses fixed-step refinement**:
  - In `.eval()` mode, the wrapper does not early-halt based on `q_halt_logits`; it runs until `halt_max_steps` so the whole batch stays synchronized.

- **Truncated backprop through inner recursion**:
  - `H_cycles-1` internal cycles run with `torch.no_grad()`; only the final internal cycle contributes gradients.

- **Halting target is “exact correctness right now”**:
  - `q_halt_logits` is trained with BCE where the label is whether the entire output sequence is exactly correct (`seq_is_correct`), not e.g. a learned value from a separate reward model.

- **Optional Q-learning style “continue” head exists but is disabled by default**:
  - `no_ACT_continue: True` (default) means the “continue” Q-loss path is not used in the demo.

- **Optimizer differences across environments**:
  - The repo uses `AdamATan2` when available.
  - For Colab/Python 3.12 where the compiled backend often fails, `pretrain.py` falls back to `torch.optim.AdamW`.

- **Puzzle embedding mechanics**:
  - Even when the dataset provides only a single identifier (Sudoku), the model still prepends a learned prefix of length `puzzle_emb_len` (16 by default) by padding/reshaping the embedding to `[B, 16, 512]`.

