"""
NT-Xent (SimCLR-style) contrastive loss for vertical partitioning.

Positive pair: two different feature partitions of the *same* customer.
Negative pairs: partitions of *different* customers within the same batch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalised Temperature-scaled Cross Entropy Loss.

    Supports K ≥ 2 views per sample (one CLS embedding per partition).
    For each anchor view, all views from the same sample are positives and
    all views from other samples are negatives.

    Parameters
    ----------
    temperature : float – softmax temperature τ
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings : list of (B, d) tensors, one per partition view.

        Returns
        -------
        Scalar contrastive loss.
        """
        K = len(embeddings)          # number of views
        B = embeddings[0].size(0)

        # L2-normalise all views
        z = [F.normalize(e, dim=-1) for e in embeddings]   # list of (B, d)

        # Stack to (K*B, d)
        z_cat = torch.cat(z, dim=0)

        # Similarity matrix (K*B, K*B)
        sim = torch.matmul(z_cat, z_cat.T) / self.tau

        # Mask out self-similarity
        mask_self = torch.eye(K * B, dtype=torch.bool, device=sim.device)
        sim.masked_fill_(mask_self, float("-inf"))

        # Build positive mask: view i of sample b vs all other views of sample b
        # Sample b has indices {b, B+b, 2B+b, …, (K-1)B+b}
        pos_mask = torch.zeros(K * B, K * B, dtype=torch.bool, device=sim.device)
        for k in range(K):
            for l in range(K):
                if k != l:
                    idx_k = torch.arange(B, device=sim.device) + k * B
                    idx_l = torch.arange(B, device=sim.device) + l * B
                    pos_mask[idx_k, idx_l] = True

        # For each row, the number of positives is K-1
        # log-sum-exp over all non-self columns → denominator
        log_denom = torch.logsumexp(sim, dim=1)  # (K*B,)

        # Average log-probability of positive pairs
        pos_sim = sim.masked_fill(~pos_mask, float("-inf"))
        log_num = torch.logsumexp(pos_sim, dim=1)  # (K*B,)

        loss = -(log_num - log_denom).mean()
        return loss


class VerticalPartitionContrastiveLoss(nn.Module):
    """Wrapper that manages the projection head on top of NT-Xent."""

    def __init__(
        self,
        d_model: int,
        proj_dim: int = 64,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )
        self.ntxent = NTXentLoss(temperature)

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        projected = [self.proj(e) for e in embeddings]
        return self.ntxent(projected)
