"""
Focal loss for multi-label binary classification + auxiliary regression loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss averaged over all labels and batch elements.

    L = -α(1-p)^γ log(p)  for positive class
    L = -(1-α) p^γ log(1-p)  for negative class

    Parameters
    ----------
    alpha : float  – weighting for the positive class
    gamma : float  – focusing exponent (0 → cross-entropy)
    reduction : 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,   # (B, n_labels)
        targets: torch.Tensor,  # (B, n_labels) – binary {0, 1}
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """
    Focal (classification) + Huber (regression) composite loss.

    Parameters
    ----------
    lambda_reg : float – weight on the regression term
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        lambda_reg: float = 0.3,
    ):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.huber = nn.SmoothL1Loss()
        self.lambda_reg = lambda_reg

    def forward(
        self,
        clf_logits: torch.Tensor,   # (B, n_labels)
        reg_out: torch.Tensor,      # (B, n_labels) – predicted 1–10
        y_binary: torch.Tensor,     # (B, n_labels) – binary targets
        y_continuous: torch.Tensor, # (B, n_labels) – continuous 1–10 targets
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        focal_l = self.focal(clf_logits, y_binary)
        reg_l = self.huber(reg_out, y_continuous)
        total = focal_l + self.lambda_reg * reg_l
        return total, focal_l, reg_l
