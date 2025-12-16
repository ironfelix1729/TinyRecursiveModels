from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class DatasetMetadata:
    seq_len: int
    vocab_size: int
    pad_id: int
    ignore_label_id: Optional[int]
    blank_identifier_id: int
    num_puzzle_identifiers: int
    sets: Tuple[str, ...]


def _load_metadata(dataset_path: str, split: str) -> DatasetMetadata:
    meta_path = os.path.join(dataset_path, split, "dataset.json")
    with open(meta_path, "r") as f:
        j = json.load(f)
    return DatasetMetadata(
        seq_len=int(j["seq_len"]),
        vocab_size=int(j["vocab_size"]),
        pad_id=int(j["pad_id"]),
        ignore_label_id=j.get("ignore_label_id", None),
        blank_identifier_id=int(j["blank_identifier_id"]),
        num_puzzle_identifiers=int(j["num_puzzle_identifiers"]),
        sets=tuple(j["sets"]),
    )


def iter_first_real_batch(
    dataset_path: str,
    *,
    split: str = "train",
    set_name: Optional[str] = None,
    batch_size: int = 8,
    device: str | torch.device = "cpu",
) -> Tuple[Dict[str, torch.Tensor], DatasetMetadata]:
    """Load the first `batch_size` examples from disk.

    This is intentionally simple (no grouping, no distributed sampling) so you can
    test model plumbing quickly.

    Expected files are those produced by `dataset/build_*_dataset.py`:
      {split}/{set}__inputs.npy, {split}/{set}__labels.npy, {split}/{set}__puzzle_identifiers.npy

    Returns a batch dict with keys: inputs, labels, puzzle_identifiers.
    """
    md = _load_metadata(dataset_path, split)

    chosen_set = set_name or md.sets[0]
    inputs = np.load(os.path.join(dataset_path, split, f"{chosen_set}__inputs.npy"), mmap_mode="r")
    labels = np.load(os.path.join(dataset_path, split, f"{chosen_set}__labels.npy"), mmap_mode="r")

    # `puzzle_identifiers` is indexed per-example in the official loader via puzzle_indices.
    # For a quick sanity batch, we accept either per-example or per-puzzle arrays.
    pid_path = os.path.join(dataset_path, split, f"{chosen_set}__puzzle_identifiers.npy")
    puzzle_ids = np.load(pid_path, mmap_mode="r")

    b = int(min(batch_size, len(inputs)))
    x = np.array(inputs[:b], dtype=np.int32)
    y = np.array(labels[:b], dtype=np.int32)

    if puzzle_ids.ndim == 1 and len(puzzle_ids) == len(inputs):
        pid = np.array(puzzle_ids[:b], dtype=np.int32)
    else:
        # Fallback: if it's per-puzzle, just take first id for all examples.
        pid = np.full((b,), int(puzzle_ids[0]), dtype=np.int32)

    batch = {
        "inputs": torch.from_numpy(x).to(device),
        "labels": torch.from_numpy(y).to(device),
        "puzzle_identifiers": torch.from_numpy(pid).to(device),
    }
    return batch, md


def synthetic_batch(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_puzzle_identifiers: int,
    device: str | torch.device = "cpu",
    pad_id: int = 0,
    blank_identifier_id: int = 0,
) -> Tuple[Dict[str, torch.Tensor], DatasetMetadata]:
    """Generate a batch shaped like the real training batches."""
    rng = np.random.default_rng(0)
    x = rng.integers(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
    y = rng.integers(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
    pid = rng.integers(0, max(num_puzzle_identifiers, 1), size=(batch_size,), dtype=np.int32)

    batch = {
        "inputs": torch.from_numpy(x).to(device),
        "labels": torch.from_numpy(y).to(device),
        "puzzle_identifiers": torch.from_numpy(pid).to(device),
    }

    md = DatasetMetadata(
        seq_len=seq_len,
        vocab_size=vocab_size,
        pad_id=pad_id,
        ignore_label_id=None,
        blank_identifier_id=blank_identifier_id,
        num_puzzle_identifiers=num_puzzle_identifiers,
        sets=("synthetic",),
    )
    return batch, md
