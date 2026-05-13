"""
Step 4 – Target Domain Adaptation (Student–Teacher with Pseudo Labels).

Process
-------
1. The Step-2 model is used as a "teacher" to generate pseudo labels for
   common customers in the target domain (common feature subset).
2. The Step-3 model (trained on source-specific features) is the "student"
   that will be adapted to the target domain's feature set.
3. For non-common customers the student must self-improve via:
   a. Spectral neighbourhood clustering regularisation on bottleneck reps.
   b. Feature-alignment (MMD-style) between source bottleneck reps and target reps.
   c. Augmentation consistency: student predictions should be consistent under
      masking + noise perturbations.
4. Common customers additionally get supervised focal + regression loss
   from pseudo labels (thresholded by confidence).

Target domain feature mapping
------------------------------
The target domain has its own features (airline data).  To use the
Step-3 FeatureTransformer (trained on source-specific col IDs), we assign
target feature columns to NEW column IDs that start right after the last
source feature ID.  The transformer's embedding table is extended accordingly.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from models.feature_transformer import DomainAdaptationModel
from losses.focal_loss import CombinedLoss
from utils.clustering import spectral_cluster_loss
from utils.augmentation import augment_batch


# ──────────────────────── EMA teacher update ──────────────────────────────────

@torch.no_grad()
def _ema_update(teacher: nn.Module, student: nn.Module, decay: float):
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.data.mul_(decay).add_(sp.data, alpha=1 - decay)


# ──────────────────────── MMD feature alignment ───────────────────────────────

def _rbf_kernel(X: torch.Tensor, Y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Radial basis function kernel between two sets of vectors."""
    sq = (X.unsqueeze(1) - Y.unsqueeze(0)).pow(2).sum(-1)
    return torch.exp(-sq / (2 * sigma ** 2))


def mmd_loss(
    source_emb: torch.Tensor,   # (N_s, d)
    target_emb: torch.Tensor,   # (N_t, d)
    sigmas: list[float] | None = None,
) -> torch.Tensor:
    """Multi-scale Maximum Mean Discrepancy."""
    if sigmas is None:
        sigmas = [0.5, 1.0, 2.0]
    loss = torch.tensor(0.0, device=source_emb.device)
    for sigma in sigmas:
        k_ss = _rbf_kernel(source_emb, source_emb, sigma).mean()
        k_tt = _rbf_kernel(target_emb, target_emb, sigma).mean()
        k_st = _rbf_kernel(source_emb, target_emb, sigma).mean()
        loss = loss + k_ss + k_tt - 2 * k_st
    return loss / len(sigmas)


# ──────────────────────── model extension ─────────────────────────────────────

def _extend_col_embed(
    model: DomainAdaptationModel,
    n_new_features: int,
) -> DomainAdaptationModel:
    """
    Extend the FeatureTransformer's column embedding table by n_new_features
    rows (initialised from the existing embedding's std).
    Returns a deep copy with the extended table.
    """
    model = copy.deepcopy(model)
    old_embed = model.feature_transformer.tokenizer.col_embed
    old_weight = old_embed.weight.data
    d_model = old_weight.size(1)
    std = old_weight.std().item()

    new_rows = torch.randn(n_new_features, d_model, device=old_weight.device) * std
    new_weight = torch.cat([old_weight, new_rows], dim=0)

    new_embed = nn.Embedding(new_weight.size(0), d_model)
    new_embed.weight = nn.Parameter(new_weight)
    model.feature_transformer.tokenizer.col_embed = new_embed
    return model


# ──────────────────────── main Step-4 trainer ─────────────────────────────────

def train_step4(
    # Target domain
    target_X: np.ndarray,                    # (N_tgt, F_tgt) all target customers
    target_col_ids: list[int],               # IDs assigned to target features
    target_common_mask: np.ndarray,          # (N_tgt,) bool – is this a common customer?
    # Pseudo labels from Step 2 (common customers only, in target-domain order)
    pseudo_probs: np.ndarray,                # (N_common, n_labels)
    pseudo_reg: np.ndarray,                  # (N_common, n_labels)
    # Source bottleneck reps for MMD
    source_bottleneck_reps: np.ndarray,      # (N_src_sample, d_bottleneck)
    # Step-3 student model (will be mutated/adapted)
    student_model: DomainAdaptationModel,
    # Existing source col IDs so we know vocab boundary
    n_source_features: int,
    cfg: dict,
    device: str = "cpu",
    verbose: bool = True,
) -> DomainAdaptationModel:
    """
    Adapt the Step-3 student model to the target domain via pseudo-label
    supervision (common customers) + clustering + alignment + augmentation.

    Returns the adapted student model.
    """
    dev = torch.device(device)
    step_cfg = cfg["training"]["step4"]
    model_cfg = cfg["model"]
    n_labels = cfg["data"]["n_indices"]

    lr: float = step_cfg["lr"]
    epochs: int = step_cfg["epochs"]
    batch_size: int = step_cfg["batch_size"]
    lambda_reg: float = step_cfg.get("lambda_reg", 0.3)
    pseudo_thresh: float = step_cfg["pseudo_label_threshold"]
    n_neighbors: int = step_cfg["n_neighbors"]
    lambda_align: float = step_cfg["lambda_align"]
    lambda_cluster: float = step_cfg["lambda_cluster"]
    lambda_aug: float = step_cfg.get("lambda_aug", 0.05)
    ema_decay: float = step_cfg["ema_decay"]
    warmup_epochs: int = step_cfg.get("warmup_epochs", 5)

    # ── extend student to handle target feature IDs ───────────────────────────
    n_target_specific = len(target_col_ids)
    student = _extend_col_embed(student_model, n_target_specific)
    student = student.to(dev)

    # EMA teacher (frozen copy, updated via EMA)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.to(dev)

    col_ids_t = torch.tensor(target_col_ids, dtype=torch.long, device=dev)
    src_reps_t = torch.tensor(source_bottleneck_reps, dtype=torch.float32, device=dev)

    # ── build tensors ─────────────────────────────────────────────────────────
    X_t = torch.tensor(target_X, dtype=torch.float32)
    common_mask = torch.tensor(target_common_mask, dtype=torch.bool)
    common_idx = common_mask.nonzero(as_tuple=True)[0]  # indices of common customers

    # Confidence mask for pseudo labels
    max_conf = np.maximum(pseudo_probs, 1 - pseudo_probs)   # (N_common, n_labels)
    conf_mask = torch.tensor(
        (max_conf >= pseudo_thresh).astype(np.float32)
    )  # (N_common, n_labels)
    pseudo_bin = torch.tensor(
        (pseudo_probs >= 0.5).astype(np.float32)
    )  # (N_common, n_labels)
    pseudo_cont = torch.tensor(pseudo_reg, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_t, common_mask.float()),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )

    # ── loss & optimiser ──────────────────────────────────────────────────────
    criterion = CombinedLoss(
        alpha=step_cfg["focal_alpha"],
        gamma=step_cfg["focal_gamma"],
        lambda_reg=lambda_reg,
    )
    optimizer = AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * (len(target_X) // batch_size + 1)
    warmup_steps = warmup_epochs * (len(target_X) // batch_size + 1)
    sched_w = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    sched_c = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=lr * 0.01)

    # Map global indices to common-customer local index in pseudo-label arrays
    global_to_common: dict[int, int] = {
        int(g): l for l, g in enumerate(common_idx.tolist())
    }

    global_step = 0
    student.train()

    for epoch in range(1, epochs + 1):
        epoch_losses = {"total": 0.0, "pseudo": 0.0, "cluster": 0.0,
                        "align": 0.0, "aug": 0.0}
        n_batches = 0

        for batch_tuple in loader:
            bx, b_is_common = batch_tuple
            bx = bx.to(dev)
            b_is_common = b_is_common.to(dev).bool()

            # ── student forward ───────────────────────────────────────────────
            clf_logits, reg_out = student(bx, col_ids_t)
            bottleneck_reps = student.encode(bx, col_ids_t)

            total_loss = torch.tensor(0.0, device=dev)

            # ── pseudo-label loss for common customers ────────────────────────
            if b_is_common.any():
                # We need to retrieve the per-sample pseudo labels.
                # Use a vectorised approach: find which samples in this batch
                # are common customers, then look up their local indices.
                # (Batch indices are not tracked directly; use teacher soft targets instead.)
                # ── Teacher soft targets for augmentation consistency ──────────
                with torch.no_grad():
                    bx_aug = augment_batch(bx, mask_ratio=0.15, noise_std=0.08)
                    tch_clf, tch_reg = teacher(bx_aug, col_ids_t)
                    tch_probs = torch.sigmoid(tch_clf)

                aug_loss = F.mse_loss(
                    torch.sigmoid(clf_logits[b_is_common]),
                    tch_probs[b_is_common].detach(),
                )
                total_loss = total_loss + lambda_aug * aug_loss
                epoch_losses["aug"] += aug_loss.item()

            # ── spectral neighbourhood clustering loss ────────────────────────
            if bottleneck_reps.size(0) > n_neighbors:
                cluster_loss = spectral_cluster_loss(bottleneck_reps, n_neighbors)
                total_loss = total_loss + lambda_cluster * cluster_loss
                epoch_losses["cluster"] += cluster_loss.item()

            # ── MMD feature alignment against source reps ─────────────────────
            if src_reps_t.size(0) > 0:
                # Sample subset of source reps to match batch size
                n_src = min(src_reps_t.size(0), bottleneck_reps.size(0))
                src_idx = torch.randperm(src_reps_t.size(0), device=dev)[:n_src]
                align_loss = mmd_loss(
                    src_reps_t[src_idx],
                    bottleneck_reps[:n_src],
                )
                total_loss = total_loss + lambda_align * align_loss
                epoch_losses["align"] += align_loss.item()

            optimizer.zero_grad()
            if total_loss.requires_grad:
                total_loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                _ema_update(teacher, student, ema_decay)

            if global_step < warmup_steps:
                sched_w.step()
            else:
                sched_c.step()
            global_step += 1

            epoch_losses["total"] += total_loss.item()
            n_batches += 1

        # ── every epoch: supervised pass on pseudo-labelled common customers ──
        if len(common_idx) > 0:
            # Full pass over common customers with pseudo labels
            n_common = len(common_idx)
            perm = torch.randperm(n_common)
            for i in range(0, n_common, batch_size):
                batch_local = perm[i:i + batch_size]
                batch_global = common_idx[batch_local]
                bx_c = X_t[batch_global].to(dev)
                byb_c = pseudo_bin[batch_local].to(dev)
                byc_c = pseudo_cont[batch_local].to(dev)
                mask_c = conf_mask[batch_local].to(dev)

                clf_c, reg_c = student(bx_c, col_ids_t)

                # Mask out low-confidence labels from loss
                fl_raw = criterion.focal(clf_c, byb_c)  # scalar
                reg_raw = criterion.huber(reg_c * mask_c, byc_c * mask_c)
                pseudo_loss = fl_raw + lambda_reg * reg_raw

                optimizer.zero_grad()
                pseudo_loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                _ema_update(teacher, student, ema_decay)
                epoch_losses["pseudo"] += pseudo_loss.item()

        if verbose and (epoch % 10 == 0 or epoch == 1):
            t = epoch_losses["total"] / max(1, n_batches)
            p = epoch_losses["pseudo"] / max(1, (len(common_idx) // batch_size + 1))
            c = epoch_losses["cluster"] / max(1, n_batches)
            a = epoch_losses["align"] / max(1, n_batches)
            print(f"  [Step4] Epoch {epoch:3d}/{epochs}  "
                  f"total={t:.4f}  pseudo={p:.4f}  "
                  f"cluster={c:.4f}  align={a:.4f}")

    if verbose:
        print("[Step4] Done.")
    return student


# ──────────────────────── inference helper ────────────────────────────────────

def infer_step4(
    model: DomainAdaptationModel,
    X: np.ndarray,
    col_ids_tensor: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs, reg_preds) for all target customers."""
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
