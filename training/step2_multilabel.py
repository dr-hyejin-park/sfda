"""
Step 2 – Multi-label Training on Common Features.

Uses the pretrained FeatureTransformer from Step 1.
Only the common features (shared vocabulary between domains) and common customers
(identifiable in both source and target) are used.
A bottleneck layer and multi-label classification/regression head are added.
Training uses Focal + Huber (CombinedLoss).
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from models.feature_transformer import (
    FeatureTransformer,
    BottleneckLayer,
    MultiLabelHead,
    DomainAdaptationModel,
)
from losses.focal_loss import CombinedLoss


def _build_model_from_pretrained(
    pretrained_ft: FeatureTransformer,
    max_features: int,
    model_cfg: dict,
    n_labels: int,
) -> DomainAdaptationModel:
    """Attach a fresh bottleneck+head on top of a (possibly frozen) FeatureTransformer."""
    model = DomainAdaptationModel(
        max_features=max_features,
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        d_ff=model_cfg["d_ff"],
        dropout=model_cfg["dropout"],
        bottleneck_dim=model_cfg["bottleneck_dim"],
        n_labels=n_labels,
    )
    # Replace only the feature_transformer with pretrained weights
    model.feature_transformer.load_state_dict(pretrained_ft.state_dict())
    return model


def train_step2(
    source_X_common: np.ndarray,         # (N_common, F_common)
    source_y_binary: np.ndarray,         # (N_common, n_labels)
    source_y_continuous: np.ndarray,     # (N_common, n_labels)
    common_feature_indices: list[int],   # indices into original feature column order
    pretrained_ft: FeatureTransformer,
    cfg: dict,
    device: str = "cpu",
    verbose: bool = True,
) -> DomainAdaptationModel:
    """
    Fine-tune the FeatureTransformer (from Step 1) with a bottleneck + multi-label head
    on the common features and common customers only.

    Returns the fully assembled DomainAdaptationModel (step-2 variant).
    """
    dev = torch.device(device)
    step_cfg = cfg["training"]["step2"]
    model_cfg = cfg["model"]
    n_labels = cfg["data"]["n_indices"]

    lr: float = step_cfg["lr"]
    epochs: int = step_cfg["epochs"]
    batch_size: int = step_cfg["batch_size"]
    lambda_reg: float = step_cfg.get("lambda_reg", 0.3)
    warmup_epochs: int = step_cfg.get("warmup_epochs", 3)

    # ── model ─────────────────────────────────────────────────────────────────
    max_features = max(common_feature_indices) + 1  # respect original col-id vocab
    model = _build_model_from_pretrained(pretrained_ft, max_features, model_cfg, n_labels)
    model = model.to(dev)

    col_ids = torch.tensor(common_feature_indices, dtype=torch.long, device=dev)

    # ── loss ──────────────────────────────────────────────────────────────────
    criterion = CombinedLoss(
        alpha=step_cfg["focal_alpha"],
        gamma=step_cfg["focal_gamma"],
        lambda_reg=lambda_reg,
    )

    # ── optimiser: separate LR for pretrained vs new layers ───────────────────
    ft_params = list(model.feature_transformer.parameters())
    new_params = list(model.bottleneck.parameters()) + list(model.head.parameters())
    optimizer = AdamW(
        [{"params": ft_params, "lr": lr * 0.1},
         {"params": new_params, "lr": lr}],
        weight_decay=1e-4,
    )

    total_steps = epochs * (len(source_X_common) // batch_size + 1)
    warmup_steps = warmup_epochs * (len(source_X_common) // batch_size + 1)
    sched_w = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    sched_c = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=lr * 0.01)

    # ── dataset ───────────────────────────────────────────────────────────────
    X_t = torch.tensor(source_X_common, dtype=torch.float32)
    yb_t = torch.tensor(source_y_binary, dtype=torch.float32)
    yc_t = torch.tensor(source_y_continuous, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, yb_t, yc_t),
                        batch_size=batch_size, shuffle=True, drop_last=False)

    # ── training ──────────────────────────────────────────────────────────────
    model.train()
    global_step = 0

    for epoch in range(1, epochs + 1):
        epoch_total, epoch_focal, epoch_reg = 0.0, 0.0, 0.0
        for bx, byb, byc in loader:
            bx, byb, byc = bx.to(dev), byb.to(dev), byc.to(dev)

            clf_logits, reg_out = model(bx, col_ids)
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

            epoch_total += loss.item()
            epoch_focal += fl.item()
            epoch_reg += rl.item()

        if verbose and (epoch % 10 == 0 or epoch == 1):
            n = len(loader)
            print(f"  [Step2] Epoch {epoch:3d}/{epochs}  "
                  f"total={epoch_total/n:.4f}  "
                  f"focal={epoch_focal/n:.4f}  "
                  f"reg={epoch_reg/n:.4f}")

    if verbose:
        print("[Step2] Done.")
    return model


def infer_step2(
    model: DomainAdaptationModel,
    X: np.ndarray,
    col_ids_tensor: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference with the Step-2 model.

    Returns
    -------
    probs       : (N, n_labels) sigmoid probabilities
    reg_preds   : (N, n_labels) continuous 1–10 predictions
    """
    dev = torch.device(device)
    model.eval()
    model.to(dev)
    col_ids_tensor = col_ids_tensor.to(dev)

    all_probs, all_reg = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            bx = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=dev)
            clf_logits, reg_out = model(bx, col_ids_tensor)
            all_probs.append(torch.sigmoid(clf_logits).cpu().numpy())
            all_reg.append(reg_out.cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_reg)
