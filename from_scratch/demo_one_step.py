from __future__ import annotations

import argparse
import os

import torch

from from_scratch.puzzle_batch import iter_first_real_batch, synthetic_batch
from from_scratch.trm_min import ACTLossHeadMin, TRMMin, IGNORE_LABEL_ID


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None, help="Path to built dataset dir (e.g. data/sudoku-extreme-1k-aug-1000)")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)

    if args.data and os.path.exists(args.data):
        batch, md = iter_first_real_batch(args.data, split=args.split, batch_size=args.batch_size, device=device)
    else:
        batch, md = synthetic_batch(
            batch_size=args.batch_size,
            seq_len=256,
            vocab_size=64,
            num_puzzle_identifiers=1024,
            device=device,
        )

    # Ensure IGNORE_LABEL_ID matches the training loop convention.
    # (Real datasets use a dataset-specific ignore id which is converted in the full loader.)
    batch["labels"] = batch["labels"].clone()
    batch["labels"][batch["labels"] < 0] = IGNORE_LABEL_ID

    model = TRMMin(
        batch_size=batch["inputs"].shape[0],
        seq_len=batch["inputs"].shape[1],
        vocab_size=md.vocab_size,
        num_puzzle_identifiers=md.num_puzzle_identifiers,
        hidden_size=128,
        puzzle_emb_ndim=0,
        H_cycles=3,
        L_cycles=2,
        L_layers=2,
        halt_max_steps=4,
        halt_exploration_prob=0.0,
    ).to(device)

    head = ACTLossHeadMin(model, loss_type="softmax").to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)

    carry = head.initial_carry(batch)

    # run until all halted (like eval)
    head.train()
    steps = 0
    loss_total = 0.0
    while True:
        carry, loss, metrics, _preds, all_finish = head(carry=carry, batch=batch, return_keys=("preds",))
        (loss / batch["inputs"].shape[0]).backward()
        loss_total += float(loss.detach().cpu())
        steps += 1
        if bool(all_finish):
            break
        if steps > 16:
            break

    opt.step()
    opt.zero_grad(set_to_none=True)

    # Print a small, stable summary
    metrics_cpu = {k: float(v.detach().cpu()) for k, v in metrics.items()}
    print({"loops": steps, "loss": loss_total, **metrics_cpu})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
