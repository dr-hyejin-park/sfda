"""
Data augmentation for tabular features in the target domain.

Strategies
----------
1. Feature masking  – zero out a random subset of features.
2. Gaussian noise   – add small Gaussian perturbation (appropriate after quantile-transform).
3. Feature mixup    – convex combination of two samples.
"""
from __future__ import annotations

import numpy as np
import torch


def feature_masking(
    x: torch.Tensor,
    mask_ratio: float = 0.15,
) -> torch.Tensor:
    """Randomly zero out mask_ratio of features per sample."""
    mask = torch.bernoulli(
        torch.full_like(x, 1 - mask_ratio)
    )
    return x * mask


def gaussian_noise(
    x: torch.Tensor,
    noise_std: float = 0.1,
) -> torch.Tensor:
    """Add Gaussian noise (features are already N(0,1) after quantile transform)."""
    return x + torch.randn_like(x) * noise_std


def mixup(
    x: torch.Tensor,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return two views created by mixing pairs in the batch.
    Returns (mixed_x, lam) where lam is the mixing coefficient.
    """
    lam = float(np.random.beta(alpha, alpha))
    B = x.size(0)
    idx = torch.randperm(B, device=x.device)
    return lam * x + (1 - lam) * x[idx], lam


def augment_batch(
    x: torch.Tensor,
    mask_ratio: float = 0.15,
    noise_std: float = 0.08,
) -> torch.Tensor:
    """Apply masking + noise augmentation to produce a second view."""
    x_aug = feature_masking(x, mask_ratio)
    x_aug = gaussian_noise(x_aug, noise_std)
    return x_aug
