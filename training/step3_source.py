"""
Step 3 – Source-Specific Feature Training.

Trains a fresh DomainAdaptationModel on source-domain features that were
excluded from Step 2 (i.e. the domain-specific features).  This model is
then transferred to the target domain in Step 4.

Because the word embedding is shared across both source and target
vocabularies (built once in main.py), transferring to the target domain
requires NO embedding-table extension – simply pass the target SchemaEncoding
to forward().
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from models.feature_transformer import DomainAdaptationModel
from losses.focal_loss import CombinedLoss
from data.preprocessing import SchemaEncoding, arrays_to_tensors, unpack_batch


def train_step3(
    x_num_specific:    np.ndarray | None,  # (N_src, F_num_spec)
    x_bin_specific:    np.ndarray | None,  # (N_src, F_bin_spec) or None
    x_cat_specific:    np.ndarray | None,  # (N_src, F_cat_spec) or None
    specific_schema_enc: SchemaEncoding,
    y_binary:          np.ndarray,         # (N_src, n_labels)
    y_continuous:      np.ndarray,         # (N_src, n_labels)
    vocab_size:        int,
    cfg:               dict,
    device:            str  = "cpu",
    verbose:           bool = True,
) -> DomainAdaptationModel:
    dev       = torch.device(device)
    step_cfg  = cfg["training"]["step3"]
    model_cfg = cfg["model"]
    n_labels  = cfg["data"]["n_indices"]

    lr:         float = step_cfg["lr"]
    epochs:     int   = step_cfg["epochs"]
    batch_size: int   = step_cfg["batch_size"]
    lambda_reg: float = step_cfg.get("lambda_reg", 0.3)

    model = DomainAdaptationModel(
        vocab_size=vocab_size,
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        d_ff=model_cfg["d_ff"],
        dropout=model_cfg["dropout"],
        bottleneck_dim=model_cfg["bottleneck_dim"],
        n_labels=n_labels,
    ).to(dev)

    criterion = CombinedLoss(
        alpha=step_cfg["focal_alpha"],
        gamma=step_cfg["focal_gamma"],
        lambda_reg=lambda_reg,
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    n = y_binary.shape[0]
    total_steps = epochs * (n // batch_size + 1)
    scheduler   = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.01)

    t_num, t_bin, t_cat = arrays_to_tensors(x_num_specific, x_bin_specific, x_cat_specific, n)
    yb_t = torch.tensor(y_binary,     dtype=torch.float32)
    yc_t = torch.tensor(y_continuous, dtype=torch.float32)
    loader = DataLoader(TensorDataset(t_num, t_bin, t_cat, yb_t, yc_t),
                        batch_size=batch_size, shuffle=True)

    schema_dev = specific_schema_enc.to(dev)
    model.train()

    for epoch in range(1, epochs + 1):
        e_total = e_focal = e_reg = 0.0
        for bt_num, bt_bin, bt_cat, byb, byc in loader:
            bx_num, bx_bin, bx_cat = unpack_batch(
                bt_num.to(dev), bt_bin.to(dev), bt_cat.to(dev))
            byb, byc = byb.to(dev), byc.to(dev)

            clf_logits, reg_out = model(bx_num, bx_bin, bx_cat, schema_dev)
            loss, fl, rl = criterion(clf_logits, reg_out, byb, byc)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            e_total += loss.item(); e_focal += fl.item(); e_reg += rl.item()

        if verbose and (epoch % 10 == 0 or epoch == 1):
            nb = len(loader)
            print(f"  [Step3] Epoch {epoch:3d}/{epochs}  "
                  f"total={e_total/nb:.4f}  focal={e_focal/nb:.4f}  reg={e_reg/nb:.4f}")

    if verbose:
        print("[Step3] Done.")
    return model
