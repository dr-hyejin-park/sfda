"""
Spectral Neighbourhood Clustering utilities for Step 4.

Builds a soft neighbourhood affinity graph in the bottleneck representation
space and applies a spectral consistency regularisation term that pushes
representations of neighbouring customers to be similar.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _pairwise_sq_dist(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Efficient squared Euclidean distance (B_a, d) × (B_b, d) → (B_a, B_b)."""
    return (
        (A ** 2).sum(1, keepdim=True)
        + (B ** 2).sum(1)
        - 2 * torch.mm(A, B.T)
    ).clamp(min=0)


def build_affinity_matrix(
    embeddings: torch.Tensor,   # (B, d) – L2-normalised bottleneck reps
    n_neighbors: int = 15,
    sigma: float | None = None,
) -> torch.Tensor:              # (B, B) sparse-soft affinity
    """
    Gaussian affinity matrix using k-NN distances.
    For each row, only the k nearest neighbours get nonzero weight.
    """
    B = embeddings.size(0)
    sq_dist = _pairwise_sq_dist(embeddings, embeddings)  # (B, B)

    # Adaptive sigma: median k-NN distance
    k = min(n_neighbors, B - 1)
    topk_vals, _ = torch.topk(sq_dist, k + 1, dim=1, largest=False)  # includes self
    knn_dists = topk_vals[:, 1:]  # exclude self-distance (B, k)

    if sigma is None:
        sigma = knn_dists.median().item() + 1e-8

    # Zero out non-neighbours
    kth_dist = knn_dists[:, -1:].clamp(min=1e-8)          # (B, 1)
    mask = (sq_dist <= kth_dist).float()
    W = torch.exp(-sq_dist / (2 * sigma ** 2)) * mask
    W.fill_diagonal_(0)

    # Symmetrize & row-normalize → transition matrix
    W = (W + W.T) / 2
    row_sum = W.sum(1, keepdim=True).clamp(min=1e-8)
    return W / row_sum                                     # (B, B)


def spectral_cluster_loss(
    embeddings: torch.Tensor,   # (B, d)
    n_neighbors: int = 15,
) -> torch.Tensor:
    """
    Spectral smoothness loss: encourage neighbours (by affinity) to have
    similar L2-normalised embeddings.

    L = Σ_{i,j} W_ij ||z_i − z_j||^2  (graph Laplacian trace)
    """
    z = F.normalize(embeddings, dim=-1)
    W = build_affinity_matrix(z.detach(), n_neighbors).to(embeddings.device)
    # ||z_i - z_j||^2 = 2 - 2 z_i·z_j  (unit vectors)
    sim_mat = torch.mm(z, z.T)
    loss = (W * (1 - sim_mat)).sum() / embeddings.size(0)
    return loss
