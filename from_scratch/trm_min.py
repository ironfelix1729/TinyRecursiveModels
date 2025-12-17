from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


IGNORE_LABEL_ID = -100


@dataclass
class InnerCarry:
    z_h: torch.Tensor  # [B, L, D]
    z_l: torch.Tensor  # [B, L, D]


@dataclass
class Carry:
    inner: InnerCarry
    steps: torch.Tensor          # [B]
    halted: torch.Tensor         # [B] bool
    current_data: Dict[str, torch.Tensor]


class TRMMin(nn.Module):
    """Minimal TRM-like model.

    Goal: match the repo's *interfaces* so you can replace internals step-by-step.

    - Batch keys: inputs, labels, puzzle_identifiers
    - `initial_carry(batch)` returns Carry
    - `forward(carry, batch)` returns (new_carry, outputs_dict)

    Outputs dict contains:
      - logits: [B, seq_len, vocab]
      - q_halt_logits: [B]
      - q_continue_logits: [B]

    Halting policy here is intentionally simple (learned q_halt only).
    """

    def __init__(
        self,
        *,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        num_puzzle_identifiers: int,
        hidden_size: int = 128,
        puzzle_emb_ndim: int = 0,
        puzzle_emb_len: int = 16,
        H_cycles: int = 3,
        L_cycles: int = 4,
        L_layers: int = 2,
        expansion: float = 4.0,
        use_attention: bool = True,
        num_heads: int = 4,
        pos_encodings: str = "learned",
        halt_max_steps: int = 6,
        halt_exploration_prob: float = 0.0,
        forward_dtype: str = "bfloat16",
    ):
        super().__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_puzzle_identifiers = num_puzzle_identifiers

        self.puzzle_emb_ndim = int(puzzle_emb_ndim)
        self.puzzle_emb_len = int(puzzle_emb_len) if self.puzzle_emb_ndim > 0 else 0

        self.H_cycles = int(H_cycles)
        self.L_cycles = int(L_cycles)
        self.L_layers = int(L_layers)
        self.use_attention = bool(use_attention)
        self.num_heads = int(num_heads)
        self.pos_encodings = str(pos_encodings)

        self.halt_max_steps = int(halt_max_steps)
        self.halt_exploration_prob = float(halt_exploration_prob)

        self.forward_dtype = getattr(torch, forward_dtype)

        self.embed = nn.Embedding(vocab_size, hidden_size)
        if self.puzzle_emb_ndim > 0:
            self.puzzle_emb = nn.Embedding(num_puzzle_identifiers, self.puzzle_emb_ndim)
            self.puzzle_proj = nn.Linear(self.puzzle_emb_ndim, self.puzzle_emb_len * hidden_size, bias=False)

        total_len = self.seq_len + self.puzzle_emb_len
        if self.pos_encodings not in {"none", "learned"}:
            raise ValueError("TRMMin only supports pos_encodings={'none','learned'} for now.")
        if self.pos_encodings == "learned":
            self.pos_emb = nn.Embedding(total_len, hidden_size)

        self.in_norm = nn.LayerNorm(hidden_size)

        class Block(nn.Module):
            def __init__(self, *, hidden: int, heads: int, expansion_: float, use_attn: bool):
                super().__init__()
                self.use_attn = use_attn
                if use_attn:
                    self.ln_attn = nn.LayerNorm(hidden)
                    self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
                self.ln_ff = nn.LayerNorm(hidden)
                inner = int(hidden * expansion_)
                self.ff = nn.Sequential(
                    nn.Linear(hidden, inner),
                    nn.GELU(),
                    nn.Linear(inner, hidden),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if self.use_attn:
                    h = self.ln_attn(x)
                    a, _ = self.attn(h, h, h, need_weights=False)
                    x = x + a
                x = x + self.ff(self.ln_ff(x))
                return x

        self.l_blocks = nn.ModuleList(
            [Block(hidden=hidden_size, heads=self.num_heads, expansion_=expansion, use_attn=self.use_attention) for _ in range(self.L_layers)]
        )

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.q_head = nn.Linear(hidden_size, 2, bias=True)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)

        self.h_init = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.l_init = nn.Parameter(torch.randn(hidden_size) * 0.02)

    def _compute_dtype(self, device: torch.device) -> torch.dtype:
        # Keep things simple/robust: float32 for this minimal sandbox.
        # (The repo TRM uses custom \"casted\" modules to make bf16 easy.)
        return torch.float32

    def _input_embed(self, inputs: torch.Tensor, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        compute_dtype = self._compute_dtype(inputs.device)
        x = self.embed(inputs.to(torch.long)).to(compute_dtype)
        if self.puzzle_emb_len:
            p = self.puzzle_emb(puzzle_identifiers.to(torch.long)).to(compute_dtype)
            p = self.puzzle_proj(p.to(torch.float32)).to(compute_dtype).view(-1, self.puzzle_emb_len, x.shape[-1])
            x = torch.cat([p, x], dim=1)
        if self.pos_encodings == "learned":
            L = x.shape[1]
            pos = torch.arange(L, device=x.device, dtype=torch.long)
            x = x + self.pos_emb(pos).to(compute_dtype).unsqueeze(0)
        return self.in_norm(x)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> Carry:
        b = batch["inputs"].shape[0]
        L = self.seq_len + self.puzzle_emb_len
        D = self.embed.embedding_dim
        compute_dtype = self._compute_dtype(batch["inputs"].device)
        empty = InnerCarry(
            z_h=torch.empty((b, L, D), device=batch["inputs"].device, dtype=compute_dtype),
            z_l=torch.empty((b, L, D), device=batch["inputs"].device, dtype=compute_dtype),
        )
        return Carry(
            inner=empty,
            steps=torch.zeros((b,), device=batch["inputs"].device, dtype=torch.int32),
            halted=torch.ones((b,), device=batch["inputs"].device, dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def _reset_inner(self, reset_flag: torch.Tensor, inner: InnerCarry) -> InnerCarry:
        # broadcast [B] -> [B,1,1]
        rf = reset_flag.view(-1, 1, 1)
        z_h = torch.where(rf, self.h_init.view(1, 1, -1).to(inner.z_h.dtype), inner.z_h)
        z_l = torch.where(rf, self.l_init.view(1, 1, -1).to(inner.z_l.dtype), inner.z_l)
        return InnerCarry(z_h=z_h, z_l=z_l)

    def forward(self, carry: Carry, batch: Dict[str, torch.Tensor]) -> Tuple[Carry, Dict[str, torch.Tensor]]:
        # update per-seq data on halting boundary
        new_current = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }
        inner = self._reset_inner(carry.halted, carry.inner)
        steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)

        inp = self._input_embed(new_current["inputs"], new_current["puzzle_identifiers"])
        z_h, z_l = inner.z_h, inner.z_l

        # recursion: (H_cycles-1) no-grad + last grad, like the repo
        with torch.no_grad():
            for _ in range(max(self.H_cycles - 1, 0)):
                for _ in range(self.L_cycles):
                    for blk in self.l_blocks:
                        z_l = z_l + blk(z_h + inp)
                for blk in self.l_blocks:
                    z_h = z_h + blk(z_l)

        for _ in range(self.L_cycles):
            for blk in self.l_blocks:
                z_l = z_l + blk(z_h + inp)
        for blk in self.l_blocks:
            z_h = z_h + blk(z_l)

        # heads
        logits_full = self.lm_head(z_h.to(torch.float32))
        logits = logits_full[:, self.puzzle_emb_len : self.puzzle_emb_len + self.seq_len]

        q_logits = self.q_head(z_h[:, 0].to(torch.float32))
        q_halt_logits, q_continue_logits = q_logits[:, 0], q_logits[:, 1]

        # halting decision (train-time only, evaluation will run fixed steps in our driver)
        steps = steps + 1
        is_last = steps >= self.halt_max_steps
        halted = is_last
        if self.training and self.halt_max_steps > 1:
            halted = halted | (q_halt_logits > 0)
            if self.halt_exploration_prob > 0:
                explore = (torch.rand_like(q_halt_logits) < self.halt_exploration_prob)
                min_steps = explore.to(steps.dtype) * torch.randint_like(steps, low=2, high=self.halt_max_steps + 1)
                halted = halted & (steps >= min_steps)

        out = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        new_carry = Carry(
            inner=InnerCarry(z_h=z_h.detach(), z_l=z_l.detach()),
            steps=steps,
            halted=halted,
            current_data=new_current,
        )
        return new_carry, out


class ACTLossHeadMin(nn.Module):
    """Matches `models.losses.ACTLossHead` return signature used by `pretrain.py`."""

    def __init__(self, model: TRMMin, *, loss_type: str = "softmax"):
        super().__init__()
        self.model = model
        self.loss_type = loss_type

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def _token_loss(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # logits [B,L,V], labels [B,L]
        if self.loss_type == "softmax":
            per_tok = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).to(torch.float32),
                labels.reshape(-1).to(torch.long),
                ignore_index=IGNORE_LABEL_ID,
                reduction="none",
            ).view_as(labels)
            return torch.where(mask, per_tok, torch.zeros_like(per_tok, dtype=per_tok.dtype))
        raise ValueError(f"Unknown loss_type={self.loss_type}")

    def forward(
        self,
        *,
        carry: Carry,
        batch: Dict[str, torch.Tensor],
        return_keys: Sequence[str] = (),
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        new_carry, outputs = self.model(carry=carry, batch=batch)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            preds = torch.argmax(outputs["logits"], dim=-1)
            outputs["preds"] = preds

            mask = labels != IGNORE_LABEL_ID
            counts = mask.sum(-1)
            divisor = counts.clamp_min(1).unsqueeze(-1)
            is_correct = mask & (preds == labels)
            seq_is_correct = is_correct.sum(-1) == counts

            valid = new_carry.halted & (counts > 0)
            metrics = {
                "count": valid.sum(),
                "accuracy": torch.where(valid, (is_correct.to(torch.float32) / divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid & seq_is_correct).sum(),
                "q_halt_accuracy": (valid & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps": torch.where(valid, new_carry.steps, 0).sum(),
            }

        # losses
        mask = labels != IGNORE_LABEL_ID
        counts = mask.sum(-1)
        divisor = counts.clamp_min(1).unsqueeze(-1)
        lm_loss = (self._token_loss(outputs["logits"], labels, mask) / divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )
        metrics.update({"lm_loss": lm_loss.detach(), "q_halt_loss": q_halt_loss.detach()})

        detached = {k: outputs[k].detach() for k in return_keys if k in outputs}
        return new_carry, lm_loss + 0.5 * q_halt_loss, metrics, detached, new_carry.halted.all()
