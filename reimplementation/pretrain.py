import os
import math
import shutil
import copy
import argparse
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import wandb
import pydantic
from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
from torch.optim import AdamW

try:
    from adam_atan2 import AdamATan2
except ImportError:
    AdamATan2 = None

from .puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from .utils.functions import load_model_class, get_model_source_path
from .models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from .models.ema import EMAHelper

class LossConfig(pydantic.BaseModel):
    name: str = "stablemax_cross_entropy"

class ArchConfig(pydantic.BaseModel):
    name: str = "TinyRecursiveReasoningModel_ACTV1@reimplementation.models.trm"
    loss: LossConfig = LossConfig()
    # TRM specific params
    hidden_size: int = 384
    expansion: float = 2.6666
    num_heads: int = 6
    pos_encodings: str = "none"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    halt_max_steps: int = 16
    halt_exploration_prob: float = 0.05
    forward_dtype: str = "bfloat16"
    mlp_t: bool = True
    puzzle_emb_len: int = 0
    no_ACT_continue: bool = True
    
    # Cycles
    H_cycles: int = 1
    L_cycles: int = 1
    H_layers: int = 0
    L_layers: int = 2
    
    puzzle_emb_ndim: int = 0 # If 0, no puzzle embeddings

class PretrainConfig(pydantic.BaseModel):
    arch: ArchConfig = ArchConfig()
    data_paths: List[str] = []
    data_paths_test: List[str] = []
    
    global_batch_size: int = 128
    epochs: int = 100
    
    lr: float = 1e-3
    lr_min_ratio: float = 0.1
    lr_warmup_steps: int = 1000
    
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    
    puzzle_emb_lr: float = 1e-3
    puzzle_emb_weight_decay: float = 1e-2
    
    project_name: str = "reimpl-trm"
    run_name: str = "test-run"
    checkpoint_path: Optional[str] = "checkpoints/test"
    
    seed: int = 42
    ema: bool = True
    ema_rate: float = 0.999
    freeze_weights: bool = False
    eval_interval: int = 10

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any
    step: int
    total_steps: int

def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test)>0 and split=="test" else config.data_paths,
        global_batch_size=config.global_batch_size,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    # Using single worker for simplicity
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)
    return dataloader, dataset.metadata

def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    model_cfg = config.arch.model_dump()
    model_cfg.update({
        "batch_size": config.global_batch_size // world_size,
        "vocab_size": train_metadata.vocab_size,
        "seq_len": train_metadata.seq_len,
        "num_puzzle_identifiers": train_metadata.num_puzzle_identifiers,
    })
    
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class("ACTLossHead@reimplementation.models.losses")
    
    with torch.device("cuda"):
        model = model_cls(model_cfg)
        model = loss_head_cls(model, loss_type=config.arch.loss.name)
        if hasattr(model, "puzzle_emb") and config.arch.puzzle_emb_ndim > 0:
             # Just a check
             pass

    # Optimizers
    optimizers = []
    optimizer_lrs = []
    
    # Parameters separation
    puzzle_params = []
    other_params = []
    
    if config.arch.puzzle_emb_ndim > 0 and hasattr(model.model, "puzzle_emb"):
        puzzle_params = list(model.model.puzzle_emb.buffers()) # In sparse_embedding, weights are buffers?
        # Wait, in sparse_embedding.py: self.weights is a Buffer (persistent=True).
        # And CastedSparseEmbeddingSignSGD_Distributed takes buffers.
        pass

    # If simple training
    if AdamATan2:
        opt_cls = AdamATan2
    else:
        opt_cls = AdamW
        print("Using AdamW instead of AdamATan2")

    optimizers.append(opt_cls(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(config.beta1, config.beta2)))
    optimizer_lrs.append(config.lr)

    return model, optimizers, optimizer_lrs

def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int):
    train_state.step += 1
    batch = {k: v.cuda() for k, v in batch.items()}
    
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)

    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])
    
    ((1.0 / global_batch_size) * loss).backward()
    
    for optim in train_state.optimizers:
        optim.step()
        optim.zero_grad()
        
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    config = PretrainConfig(data_paths=[args.data_dir], epochs=args.epochs)
    
    # Initialize
    rank = 0
    world_size = 1
    
    train_loader, train_metadata = create_dataloader(config, "train", rank, world_size, test_set_mode=False, epochs_per_iter=config.epochs)
    
    model, optimizers, lrs = create_model(config, train_metadata, rank, world_size)
    
    train_state = TrainState(
        model=model, optimizers=optimizers, optimizer_lrs=lrs, carry=None,
        step=0, total_steps=int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)
    )
    
    if config.ema:
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)
        
    pbar = tqdm.tqdm(total=train_state.total_steps)
    
    model.train()
    for set_name, batch, effective_bs in train_loader:
        metrics = train_batch(config, train_state, batch, config.global_batch_size)
        pbar.update(1)
        pbar.set_description(f"Loss: {metrics.get('lm_loss', 0):.4f} Acc: {metrics.get('accuracy', 0):.4f}")
        
        if config.ema:
            ema_helper.update(train_state.model)

    print("Training complete.")

if __name__ == "__main__":
    main()
