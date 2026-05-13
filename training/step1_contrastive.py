"""
Step 1 – Vertical Partitioning Contrastive Learning.

The source domain feature table is split into K disjoint subsets (vertical partitions).
For each mini-batch the FeatureTransformer encodes each partition separately.
Positive pairs  = different partitions of the SAME customer.
Negative pairs  = any partition of a DIFFERENT customer in the batch.
The NT-Xent loss is back-propagated only through the FeatureTransformer and
the projection head.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from models.feature_transformer import FeatureTransformer
from losses.contrastive_loss import VerticalPartitionContrastiveLoss
from data.preprocessing import vertical_partition


def _warmup_schedule(optimizer: torch.optim.Optimizer, warmup_steps: int):
    return torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )


def train_step1(
    source_X: np.ndarray,       # (N, F) – quantile-transformed
    feature_cols: list[str],
    cfg: dict,
    device: str = "cpu",
    verbose: bool = True,
) -> FeatureTransformer:
    """
    Train the FeatureTransformer with vertical-partitioning contrastive learning.

    Parameters
    ----------
    source_X     : feature matrix (already quantile-transformed, NaN-free)
    feature_cols : column names (length must match source_X.shape[1])
    cfg          : full config dict
    device       : torch device string

    Returns
    -------
    Trained FeatureTransformer (can be reused in later steps).
    """
    dev = torch.device(device)
    step_cfg = cfg["training"]["step1"]
    model_cfg = cfg["model"]

    n_partitions: int = step_cfg["n_partitions"]
    lr: float = step_cfg["lr"]
    epochs: int = step_cfg["epochs"]
    batch_size: int = step_cfg["batch_size"]
    temperature: float = step_cfg["temperature"]
    warmup_epochs: int = step_cfg.get("warmup_epochs", 5)

    # ── build partitions (column index lists) ─────────────────────────────────
    partitions = vertical_partition(
        list(range(len(feature_cols))),
        n_partitions=n_partitions,
        random_seed=cfg["data"]["random_seed"],
    )
    if verbose:
        sizes = [len(p) for p in partitions]
        print(f"[Step1] Partition sizes: {sizes}")

    # ── model & loss ──────────────────────────────────────────────────────────
    max_features = len(feature_cols)
    model = FeatureTransformer(
        max_features=max_features,
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        d_ff=model_cfg["d_ff"],
        dropout=model_cfg["dropout"],
    ).to(dev)

    contrastive_loss = VerticalPartitionContrastiveLoss(
        d_model=model_cfg["d_model"],
        proj_dim=model_cfg["bottleneck_dim"],
        temperature=temperature,
    ).to(dev)

    params = list(model.parameters()) + list(contrastive_loss.parameters())
    optimizer = AdamW(params, lr=lr, weight_decay=1e-4)

    total_steps = epochs * (len(source_X) // batch_size + 1)
    warmup_steps = warmup_epochs * (len(source_X) // batch_size + 1)
    scheduler_warmup = _warmup_schedule(optimizer, warmup_steps)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=lr * 0.01)

    # ── dataset ───────────────────────────────────────────────────────────────
    X_tensor = torch.tensor(source_X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=True, drop_last=True)

    # ── training loop ─────────────────────────────────────────────────────────
    model.train()
    contrastive_loss.train()
    global_step = 0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (batch_x,) in loader:
            batch_x = batch_x.to(dev)

            # Encode each partition
            embeddings = []
            for part_ids in partitions:
                col_ids = torch.tensor(part_ids, dtype=torch.long, device=dev)
                emb = model(batch_x[:, part_ids], col_ids)  # (B, d_model)
                embeddings.append(emb)

            loss = contrastive_loss(embeddings)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            if global_step < warmup_steps:
                scheduler_warmup.step()
            else:
                scheduler_cosine.step()
            global_step += 1

            epoch_loss += loss.item()

        if verbose and (epoch % 10 == 0 or epoch == 1):
            avg = epoch_loss / len(loader)
            print(f"  [Step1] Epoch {epoch:3d}/{epochs}  contrastive_loss={avg:.4f}")

    if verbose:
        print("[Step1] Done.")
    return model
