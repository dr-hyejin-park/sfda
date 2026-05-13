"""
Step 2 – Multi-label Training on Common Features.

Uses the pretrained FeatureTransformer from Step 1.
Only common features (numerical subset shared by both domains) and common
customers are used.  A BottleneckLayer + MultiLabelHead are added on top.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from models.feature_transformer import (
    FeatureTransformer, DomainAdaptationModel,
)
from losses.focal_loss import CombinedLoss
from data.preprocessing import SchemaEncoding, arrays_to_tensors, unpack_batch


def _build_step2_model(
    pretrained_ft: FeatureTransformer,
    vocab_size:    int,
    model_cfg:     dict,
    n_labels:      int,
) -> DomainAdaptationModel:
    """Attach fresh bottleneck + head on top of a (possibly pretrained) FT."""
    model = DomainAdaptationModel(
        vocab_size=vocab_size,
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        d_ff=model_cfg["d_ff"],
        dropout=model_cfg["dropout"],
        bottleneck_dim=model_cfg["bottleneck_dim"],
        n_labels=n_labels,
    )
    model.feature_transformer.load_state_dict(pretrained_ft.state_dict())
    return model


def train_step2(
    x_num_common:     np.ndarray | None,  # (N_common, F_num_common)
    x_bin_common:     np.ndarray | None,  # (N_common, F_bin_common) or None
    x_cat_common:     np.ndarray | None,  # (N_common, F_cat_common) or None
    common_schema_enc: SchemaEncoding,
    y_binary:         np.ndarray,         # (N_common, n_labels)
    y_continuous:     np.ndarray,         # (N_common, n_labels)
    pretrained_ft:    FeatureTransformer,
    vocab_size:       int,
    cfg:              dict,
    device:           str  = "cpu",
    verbose:          bool = True,
) -> DomainAdaptationModel:
    dev      = torch.device(device)
    step_cfg = cfg["training"]["step2"]
    model_cfg = cfg["model"]
    n_labels  = cfg["data"]["n_indices"]

    lr:           float = step_cfg["lr"]
    epochs:       int   = step_cfg["epochs"]
    batch_size:   int   = step_cfg["batch_size"]
    lambda_reg:   float = step_cfg.get("lambda_reg", 0.3)
    warmup_epochs: int  = step_cfg.get("warmup_epochs", 3)

    model = _build_step2_model(pretrained_ft, vocab_size, model_cfg, n_labels).to(dev)

    criterion = CombinedLoss(
        alpha=step_cfg["focal_alpha"],
        gamma=step_cfg["focal_gamma"],
        lambda_reg=lambda_reg,
    )

    # Lower LR for pretrained backbone, higher for new layers
    ft_params  = list(model.feature_transformer.parameters())
    new_params = list(model.bottleneck.parameters()) + list(model.head.parameters())
    optimizer  = AdamW(
        [{"params": ft_params, "lr": lr * 0.1},
         {"params": new_params, "lr": lr}],
        weight_decay=1e-4,
    )

    n = y_binary.shape[0]
    steps_per_epoch = n // batch_size + 1
    total_steps     = epochs * steps_per_epoch
    warmup_steps    = warmup_epochs * steps_per_epoch
    sched_w = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    sched_c = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps),
                                eta_min=lr * 0.01)

    t_num, t_bin, t_cat = arrays_to_tensors(x_num_common, x_bin_common, x_cat_common, n)
    yb_t = torch.tensor(y_binary,     dtype=torch.float32)
    yc_t = torch.tensor(y_continuous, dtype=torch.float32)
    loader = DataLoader(TensorDataset(t_num, t_bin, t_cat, yb_t, yc_t),
                        batch_size=batch_size, shuffle=True)

    schema_dev = common_schema_enc.to(dev)
    model.train()
    global_step = 0

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

            if global_step < warmup_steps:
                sched_w.step()
            else:
                sched_c.step()
            global_step += 1

            e_total += loss.item(); e_focal += fl.item(); e_reg += rl.item()

        if verbose and (epoch % 10 == 0 or epoch == 1):
            nb = len(loader)
            print(f"  [Step2] Epoch {epoch:3d}/{epochs}  "
                  f"total={e_total/nb:.4f}  focal={e_focal/nb:.4f}  reg={e_reg/nb:.4f}")

    if verbose:
        print("[Step2] Done.")
    return model


def infer_step2(
    model:      DomainAdaptationModel,
    x_num:      np.ndarray | None,
    x_bin:      np.ndarray | None,
    x_cat:      np.ndarray | None,
    schema_enc: SchemaEncoding,
    device:     str = "cpu",
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs, reg_preds) for the given feature set."""
    dev = torch.device(device)
    model.eval(); model.to(dev)
    schema_dev = schema_enc.to(dev)

    n = (x_num if x_num is not None
         else x_bin if x_bin is not None else x_cat).shape[0]
    t_num, t_bin, t_cat = arrays_to_tensors(x_num, x_bin, x_cat, n)

    all_probs, all_reg = [], []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            bn, bb, bc = unpack_batch(
                t_num[i:i+batch_size].to(dev),
                t_bin[i:i+batch_size].to(dev),
                t_cat[i:i+batch_size].to(dev),
            )
            clf, reg = model(bn, bb, bc, schema_dev)
            all_probs.append(torch.sigmoid(clf).cpu().numpy())
            all_reg.append(reg.cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_reg)
