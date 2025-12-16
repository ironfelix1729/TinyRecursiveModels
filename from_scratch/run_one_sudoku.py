from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1


@dataclass(frozen=True)
class SudokuExample:
    inp: np.ndarray  # [81] tokens in 1..10 (1=blank)
    sol: np.ndarray  # [81] tokens in 1..10


def _download_one_sudoku(*, split: str = "train") -> SudokuExample:
    # Dataset used by the repo's `dataset/build_sudoku_dataset.py`
    repo_id = "sapientinc/sudoku-extreme"
    path = hf_hub_download(repo_id, f"{split}.csv", repo_type="dataset")

    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)  # header
        _source, q, a, _rating = next(r)

    # q has '.' for blanks; a is full solution. Convert to 0..9 digits then shift to 1..10 tokens.
    q_digits = np.frombuffer(q.replace(".", "0").encode(), dtype=np.uint8) - ord("0")
    a_digits = np.frombuffer(a.encode(), dtype=np.uint8) - ord("0")

    inp = (q_digits.astype(np.int32) + 1).reshape(81)
    sol = (a_digits.astype(np.int32) + 1).reshape(81)
    return SudokuExample(inp=inp, sol=sol)


def _tokens_to_grid(x: torch.Tensor) -> np.ndarray:
    # Keep *token ids* (0..10) so we can render PAD vs blank distinctly.
    return x.detach().cpu().to(torch.int32).numpy().reshape(9, 9)


def _format_grid_tokens(token_grid: np.ndarray) -> str:
    # Token mapping for Sudoku dataset in this repo:
    # - 0: PAD
    # - 1: blank cell (digit 0 in the raw CSV)
    # - 2..10: digits 1..9
    rows = []
    for r in range(9):
        out = []
        for t in token_grid[r]:
            t = int(t)
            if t == 0:
                out.append("_")   # PAD
            elif t == 1:
                out.append(".")   # blank
            else:
                out.append(str(t - 1))
        row = "".join(out)
        rows.append(row)
    return "\n".join(rows)


def _step_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float, float]:
    # logits: [1,81,V], labels: [1,81]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).to(torch.float32),
        labels.reshape(-1).to(torch.long),
        reduction="mean",
    )
    preds = torch.argmax(logits, dim=-1)
    tok_acc = (preds == labels).to(torch.float32).mean()
    exact = (preds == labels).all(dim=-1).to(torch.float32).mean()
    return float(loss.detach().cpu()), float(tok_acc.detach().cpu()), float(exact.detach().cpu())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=str, default="train", choices=["train", "test"])
    p.add_argument("--K", type=int, default=6, help="Number of recursion steps (halt_max_steps)")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--L-layers", type=int, default=2)
    p.add_argument("--H-cycles", type=int, default=3)
    p.add_argument("--L-cycles", type=int, default=6)
    p.add_argument("--mlp-t", action="store_true", help="Use MLP-T (no attention) for speed")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    ex = _download_one_sudoku(split=args.split)

    batch = {
        "inputs": torch.from_numpy(ex.inp[None, :]).to(device),
        "labels": torch.from_numpy(ex.sol[None, :]).to(device),
        "puzzle_identifiers": torch.zeros((1,), dtype=torch.int32, device=device),
    }

    # Repo TRM config dict (matching `TinyRecursiveReasoningModel_ACTV1Config`).
    cfg = dict(
        batch_size=1,
        seq_len=81,
        vocab_size=11,
        num_puzzle_identifiers=1,
        puzzle_emb_ndim=0,
        puzzle_emb_len=0,
        H_cycles=int(args.H_cycles),
        L_cycles=int(args.L_cycles),
        H_layers=0,
        L_layers=int(args.L_layers),
        hidden_size=int(args.hidden),
        expansion=4.0,
        num_heads=4,
        pos_encodings="none",
        halt_max_steps=int(args.K),
        halt_exploration_prob=0.0,
        forward_dtype="float32",
        mlp_t=bool(args.mlp_t),
        no_ACT_continue=True,
    )

    model = TinyRecursiveReasoningModel_ACTV1(cfg).to(device)

    # We run in eval mode so halting is deterministic: exactly K steps.
    model.eval()

    carry = model.initial_carry(batch)

    print("### Token mapping: 0=PAD, 1='.' blank, 2..10=digits 1..9")
    print("\n### Input")
    print(_format_grid_tokens(_tokens_to_grid(batch["inputs"][0])))
    print("\n### Solution")
    print(_format_grid_tokens(_tokens_to_grid(batch["labels"][0])))

    prev_pred = None

    for step in range(1, args.K + 1):
        carry, out = model(carry=carry, batch=batch)
        logits = out["logits"]
        preds = torch.argmax(logits, dim=-1)

        loss, tok_acc, exact = _step_metrics(logits, batch["labels"])
        g = _tokens_to_grid(preds[0])

        changed = None
        if prev_pred is not None:
            changed = int((preds != prev_pred).sum().item())
        prev_pred = preds

        q = float(out["q_halt_logits"].detach().cpu().item())
        halted = bool(carry.halted.all().detach().cpu().item())

        print(f"\n### Step {step}/{args.K}")
        print({"loss": loss, "token_acc": tok_acc, "exact": exact, "q_halt_logit": q, "changed_tokens": changed, "all_halted": halted})
        print(_format_grid_tokens(_tokens_to_grid(preds[0])))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
