"""
Step 3 – Source-Only Feature Multi-label Training.

Trains a new DomainAdaptationModel on the source-specific features
(all source features EXCLUDING those used in Step 2).
This model will be transferred to the target domain in Step 4.
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


def train_step3(
    source_X_specific: np.ndarray,      # (N_source, F_specific)
    source_y_binary: np.ndarray,        # (N_source, n_labels)
    source_y_continuous: np.ndarray,    # (N_source, n_labels)
    specific_feature_indices: list[int],
    cfg: dict,
    n_total_source_features: int | None = None,
    device: str = "cpu",
    verbose: bool = True,
) -> DomainAdaptationModel:
    """
    Train a fresh DomainAdaptationModel on source-specific features.

    The column IDs used here correspond to the positions of source-specific
    features within the *original* source feature vocabulary, so the
    FeatureTransformer's column embedding table is sized to cover all source
    features (important so that target domain can reuse these column embeddings
    for features that map to the same semantic position).

    Returns the trained model (to be adapted in Step 4).
    """
    dev = torch.device(device)
    step_cfg = cfg["training"]["step3"]
    model_cfg = cfg["model"]
    n_labels = cfg["data"]["n_indices"]

    lr: float = step_cfg["lr"]
    epochs: int = step_cfg["epochs"]
    batch_size: int = step_cfg["batch_size"]
    lambda_reg: float = step_cfg.get("lambda_reg", 0.3)

    # ── model ─────────────────────────────────────────────────────────────────
    # max_features must accommodate the FULL source vocabulary so that when
    # this model is extended in Step 4, target col IDs (starting at n_src_vocab)
    # are correctly offset from the original embedding table.
    max_features = n_total_source_features if n_total_source_features is not None \
        else max(specific_feature_indices) + 1
    model = DomainAdaptationModel(
        max_features=max_features,
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        d_ff=model_cfg["d_ff"],
        dropout=model_cfg["dropout"],
        bottleneck_dim=model_cfg["bottleneck_dim"],
        n_labels=n_labels,
    ).to(dev)

    col_ids = torch.tensor(specific_feature_indices, dtype=torch.long, device=dev)

    # ── loss & optimiser ──────────────────────────────────────────────────────
    criterion = CombinedLoss(
        alpha=step_cfg["focal_alpha"],
        gamma=step_cfg["focal_gamma"],
        lambda_reg=lambda_reg,
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    total_steps = epochs * (len(source_X_specific) // batch_size + 1)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.01)

    # ── dataset ───────────────────────────────────────────────────────────────
    X_t = torch.tensor(source_X_specific, dtype=torch.float32)
    yb_t = torch.tensor(source_y_binary, dtype=torch.float32)
    yc_t = torch.tensor(source_y_continuous, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, yb_t, yc_t),
                        batch_size=batch_size, shuffle=True, drop_last=False)

    # ── training ──────────────────────────────────────────────────────────────
    model.train()
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
            scheduler.step()

            epoch_total += loss.item()
            epoch_focal += fl.item()
            epoch_reg += rl.item()

        if verbose and (epoch % 10 == 0 or epoch == 1):
            n = len(loader)
            print(f"  [Step3] Epoch {epoch:3d}/{epochs}  "
                  f"total={epoch_total/n:.4f}  "
                  f"focal={epoch_focal/n:.4f}  "
                  f"reg={epoch_reg/n:.4f}")

    if verbose:
        print("[Step3] Done.")
    return model
