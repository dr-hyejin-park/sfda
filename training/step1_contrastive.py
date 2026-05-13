"""
Step 1 – Vertical Partitioning Contrastive Learning.

The full source feature space (numerical + binary + categorical) is split into
K disjoint subsets using global feature indices.  For each mini-batch the
FeatureTransformer encodes each partition separately via TransTabTokenizer.

Positive pairs  = different partitions of the SAME customer.
Negative pairs  = any partition of a DIFFERENT customer in the batch.
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
from data.preprocessing import (
    SchemaEncoding, vertical_partition, arrays_to_tensors, unpack_batch,
)


def _warmup_schedule(optimizer, warmup_steps: int):
    return torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )


def train_step1(
    source_x_num: np.ndarray | None,    # (N, F_num) quantile-transformed
    source_x_bin: np.ndarray | None,    # (N, F_bin) int {0,1}
    source_x_cat: np.ndarray | None,    # (N, F_cat) int codes
    schema_enc:   SchemaEncoding,       # pre-computed, CPU
    vocab_size:   int,
    cfg:          dict,
    device:       str  = "cpu",
    verbose:      bool = True,
) -> FeatureTransformer:
    """
    Train the FeatureTransformer with vertical-partitioning contrastive learning.

    Returns the trained FeatureTransformer for reuse in Step 2.
    """
    dev      = torch.device(device)
    step_cfg = cfg["training"]["step1"]
    model_cfg = cfg["model"]

    n_partitions: int   = step_cfg["n_partitions"]
    lr:           float = step_cfg["lr"]
    epochs:       int   = step_cfg["epochs"]
    batch_size:   int   = step_cfg["batch_size"]
    temperature:  float = step_cfg["temperature"]
    warmup_epochs: int  = step_cfg.get("warmup_epochs", 5)

    n = (source_x_num if source_x_num is not None
         else source_x_bin if source_x_bin is not None
         else source_x_cat).shape[0]

    n_total = schema_enc.n_num + schema_enc.n_bin + schema_enc.n_cat
    partitions = vertical_partition(
        n_total, n_partitions=n_partitions, random_seed=cfg["data"]["random_seed"]
    )
    if verbose:
        print(f"[Step1] Partition sizes: {[len(p) for p in partitions]}")

    # ── model & loss ──────────────────────────────────────────────────────────
    model = FeatureTransformer(
        vocab_size=vocab_size,
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

    params    = list(model.parameters()) + list(contrastive_loss.parameters())
    optimizer = AdamW(params, lr=lr, weight_decay=1e-4)

    steps_per_epoch = n // batch_size + 1
    total_steps     = epochs * steps_per_epoch
    warmup_steps    = warmup_epochs * steps_per_epoch
    sched_w = _warmup_schedule(optimizer, warmup_steps)
    sched_c = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps),
                                eta_min=lr * 0.01)

    # ── dataset (three tensors; zero-column if type absent) ───────────────────
    t_num, t_bin, t_cat = arrays_to_tensors(source_x_num, source_x_bin, source_x_cat, n)
    loader = DataLoader(TensorDataset(t_num, t_bin, t_cat),
                        batch_size=batch_size, shuffle=True, drop_last=True)

    schema_dev = schema_enc.to(dev)
    model.train()
    contrastive_loss.train()
    global_step = 0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for bt_num, bt_bin, bt_cat in loader:
            bx_num, bx_bin, bx_cat = unpack_batch(
                bt_num.to(dev), bt_bin.to(dev), bt_cat.to(dev))

            embeddings = model.encode_partitions(
                bx_num, bx_bin, bx_cat, partitions, schema_dev)

            loss = contrastive_loss(embeddings)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            if global_step < warmup_steps:
                sched_w.step()
            else:
                sched_c.step()
            global_step += 1
            epoch_loss += loss.item()

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  [Step1] Epoch {epoch:3d}/{epochs}  "
                  f"contrastive_loss={epoch_loss / len(loader):.4f}")

    if verbose:
        print("[Step1] Done.")
    return model
