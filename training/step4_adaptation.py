"""
Step 4 – Target Domain Adaptation (Student–Teacher with Pseudo Labels).

Key difference from the old implementation
-------------------------------------------
Because the word embedding in TransTabTokenizer is shared across the full
vocabulary (source + target column names + all categorical values), NO
embedding-table extension is needed when moving to the target domain.
We simply pass the target-domain SchemaEncoding to forward().

Losses
------
1. Pseudo-label focal + regression: common customers in target domain,
   supervised by Step-2 predictions filtered by confidence.
2. Spectral neighbourhood clustering: smoothness over bottleneck reps.
3. MMD feature alignment: match source and target bottleneck distributions.
4. Augmentation consistency: student ≈ EMA teacher under masked/noisy input.
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
from data.preprocessing import SchemaEncoding, arrays_to_tensors, unpack_batch
from utils.clustering import spectral_cluster_loss
from utils.augmentation import augment_batch


# ──────────────────────── EMA update ─────────────────────────────────────────

@torch.no_grad()
def _ema_update(teacher: nn.Module, student: nn.Module, decay: float):
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.data.mul_(decay).add_(sp.data, alpha=1 - decay)


# ──────────────────────── MMD alignment ──────────────────────────────────────

def _rbf_kernel(X: torch.Tensor, Y: torch.Tensor, sigma: float) -> torch.Tensor:
    sq = (X.unsqueeze(1) - Y.unsqueeze(0)).pow(2).sum(-1)
    return torch.exp(-sq / (2 * sigma ** 2))


def mmd_loss(src: torch.Tensor, tgt: torch.Tensor,
             sigmas: list[float] | None = None) -> torch.Tensor:
    """Multi-scale Maximum Mean Discrepancy."""
    sigmas = sigmas or [0.5, 1.0, 2.0]
    loss = torch.zeros(1, device=src.device)
    for s in sigmas:
        loss = loss + _rbf_kernel(src, src, s).mean() \
                    + _rbf_kernel(tgt, tgt, s).mean() \
                    - 2 * _rbf_kernel(src, tgt, s).mean()
    return loss.squeeze() / len(sigmas)


# ──────────────────────── main Step-4 trainer ─────────────────────────────────

def train_step4(
    # Target domain – full set of customers
    tgt_x_num:           np.ndarray | None,  # (N_tgt, F_num_tgt)
    tgt_x_bin:           np.ndarray | None,  # (N_tgt, F_bin_tgt)
    tgt_x_cat:           np.ndarray | None,  # (N_tgt, F_cat_tgt)
    tgt_schema_enc:      SchemaEncoding,
    target_common_mask:  np.ndarray,         # (N_tgt,) bool
    # Pseudo labels from Step 2 (aligned to target common-customer row order)
    pseudo_probs:        np.ndarray,         # (N_common, n_labels)
    pseudo_reg:          np.ndarray,         # (N_common, n_labels)
    # Source bottleneck reps for MMD alignment
    source_bottleneck_reps: np.ndarray,      # (N_src_sample, d_bottleneck)
    # Step-3 student model
    student_model:       DomainAdaptationModel,
    cfg:                 dict,
    device:              str  = "cpu",
    verbose:             bool = True,
) -> DomainAdaptationModel:
    dev      = torch.device(device)
    step_cfg = cfg["training"]["step4"]

    lr:            float = step_cfg["lr"]
    epochs:        int   = step_cfg["epochs"]
    batch_size:    int   = step_cfg["batch_size"]
    lambda_reg:    float = step_cfg.get("lambda_reg", 0.3)
    pseudo_thresh: float = step_cfg["pseudo_label_threshold"]
    n_neighbors:   int   = step_cfg["n_neighbors"]
    lambda_align:  float = step_cfg["lambda_align"]
    lambda_cluster: float = step_cfg["lambda_cluster"]
    lambda_aug:    float = step_cfg.get("lambda_aug", 0.05)
    ema_decay:     float = step_cfg["ema_decay"]
    warmup_epochs: int   = step_cfg.get("warmup_epochs", 5)

    # ── student & EMA teacher ─────────────────────────────────────────────────
    # Deep copy so source model is not modified in place
    student = copy.deepcopy(student_model).to(dev)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)

    schema_dev = tgt_schema_enc.to(dev)
    src_reps_t = torch.tensor(source_bottleneck_reps, dtype=torch.float32, device=dev)

    # ── full dataset loader ───────────────────────────────────────────────────
    n_tgt = (tgt_x_num if tgt_x_num is not None
             else tgt_x_bin if tgt_x_bin is not None else tgt_x_cat).shape[0]
    t_num, t_bin, t_cat = arrays_to_tensors(tgt_x_num, tgt_x_bin, tgt_x_cat, n_tgt)
    common_mask_t = torch.tensor(target_common_mask, dtype=torch.bool)
    loader = DataLoader(
        TensorDataset(t_num, t_bin, t_cat, common_mask_t.float()),
        batch_size=batch_size, shuffle=True,
    )

    # ── common customer pseudo-label tensors ──────────────────────────────────
    common_idx = common_mask_t.nonzero(as_tuple=True)[0]  # global row indices
    n_common   = len(common_idx)

    max_conf   = np.maximum(pseudo_probs, 1 - pseudo_probs)
    conf_mask  = torch.tensor((max_conf >= pseudo_thresh).astype(np.float32))
    pseudo_bin = torch.tensor((pseudo_probs >= 0.5).astype(np.float32))
    pseudo_cont = torch.tensor(pseudo_reg, dtype=torch.float32)

    # ── optimiser ────────────────────────────────────────────────────────────
    criterion = CombinedLoss(
        alpha=step_cfg["focal_alpha"],
        gamma=step_cfg["focal_gamma"],
        lambda_reg=lambda_reg,
    )
    optimizer = AdamW(student.parameters(), lr=lr, weight_decay=1e-4)

    steps_per_epoch = n_tgt // batch_size + 1
    total_steps     = epochs * steps_per_epoch
    warmup_steps    = warmup_epochs * steps_per_epoch
    sched_w = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    sched_c = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps),
                                eta_min=lr * 0.01)

    global_step = 0
    student.train()

    for epoch in range(1, epochs + 1):
        losses = {"total": 0.0, "cluster": 0.0, "align": 0.0, "aug": 0.0}
        n_batches = 0

        for bt_num, bt_bin, bt_cat, b_is_common in loader:
            bx_num, bx_bin, bx_cat = unpack_batch(
                bt_num.to(dev), bt_bin.to(dev), bt_cat.to(dev))
            b_is_common = b_is_common.to(dev).bool()

            reps = student.encode(bx_num, bx_bin, bx_cat, schema_dev)
            total_loss = torch.zeros(1, device=dev)

            # ── augmentation consistency (common customers) ───────────────────
            if b_is_common.any() and bx_num is not None:
                bx_num_aug = augment_batch(bx_num, mask_ratio=0.15, noise_std=0.08)
                with torch.no_grad():
                    tch_clf, _ = teacher(bx_num_aug, bx_bin, bx_cat, schema_dev)
                    tch_probs  = torch.sigmoid(tch_clf)
                stu_clf, _ = student(bx_num, bx_bin, bx_cat, schema_dev)
                aug_loss = F.mse_loss(
                    torch.sigmoid(stu_clf[b_is_common]),
                    tch_probs[b_is_common].detach(),
                )
                total_loss = total_loss + lambda_aug * aug_loss
                losses["aug"] += aug_loss.item()

            # ── spectral neighbourhood clustering ─────────────────────────────
            if reps.size(0) > n_neighbors:
                cl = spectral_cluster_loss(reps, n_neighbors)
                total_loss = total_loss + lambda_cluster * cl
                losses["cluster"] += cl.item()

            # ── MMD feature alignment ─────────────────────────────────────────
            if src_reps_t.size(0) > 0:
                k = min(src_reps_t.size(0), reps.size(0))
                src_idx = torch.randperm(src_reps_t.size(0), device=dev)[:k]
                al = mmd_loss(src_reps_t[src_idx], reps[:k])
                total_loss = total_loss + lambda_align * al
                losses["align"] += al.item()

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
            losses["total"] += total_loss.item()
            n_batches += 1

        # ── per-epoch: pseudo-label supervised pass on common customers ───────
        pseudo_total = 0.0
        if n_common > 0:
            perm = torch.randperm(n_common)
            for i in range(0, n_common, batch_size):
                bl = perm[i:i + batch_size]
                bg = common_idx[bl]
                bx_n, bx_b, bx_c = unpack_batch(
                    t_num[bg].to(dev), t_bin[bg].to(dev), t_cat[bg].to(dev))
                byb  = pseudo_bin[bl].to(dev)
                byc  = pseudo_cont[bl].to(dev)
                mask = conf_mask[bl].to(dev)

                clf, reg = student(bx_n, bx_b, bx_c, schema_dev)
                fl  = criterion.focal(clf, byb)
                rl  = criterion.huber(reg * mask, byc * mask)
                pl  = fl + lambda_reg * rl

                optimizer.zero_grad()
                pl.backward()
                nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                _ema_update(teacher, student, ema_decay)
                pseudo_total += pl.item()

        if verbose and (epoch % 10 == 0 or epoch == 1):
            nb = max(1, n_batches)
            np_ = max(1, n_common // batch_size + 1)
            print(f"  [Step4] Epoch {epoch:3d}/{epochs}  "
                  f"total={losses['total']/nb:.4f}  pseudo={pseudo_total/np_:.4f}  "
                  f"cluster={losses['cluster']/nb:.4f}  align={losses['align']/nb:.4f}")

    if verbose:
        print("[Step4] Done.")
    return student


# ──────────────────────── inference helper ────────────────────────────────────

def infer_step4(
    model:      DomainAdaptationModel,
    x_num:      np.ndarray | None,
    x_bin:      np.ndarray | None,
    x_cat:      np.ndarray | None,
    schema_enc: SchemaEncoding,
    device:     str = "cpu",
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
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
