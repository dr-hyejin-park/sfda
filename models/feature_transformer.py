"""
TransTab-inspired Feature Transformer.

Key design choices
------------------
* Each feature is tokenised as:  value_projection(x_i) + col_name_embedding(i)
  – enables the transformer to handle arbitrary feature subsets (vertical partition).
* A learnable [CLS] token is prepended; its output is the row representation.
* Works with quantile-transformed (continuous) features only.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────── tokeniser ───────────────────────────────────────

class FeatureTokenizer(nn.Module):
    """
    Map a (batch × n_features) tensor to (batch × n_features × d_model) tokens.

    Parameters
    ----------
    max_features : int   – upper bound on number of features (vocabulary size for col IDs)
    d_model      : int   – token dimensionality
    """

    def __init__(self, max_features: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        # Column-name embedding (feature identity)
        self.col_embed = nn.Embedding(max_features, d_model)
        # Value projection: scalar → d_model
        self.val_proj = nn.Linear(1, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,        # (B, F) – feature values
        col_ids: torch.Tensor,  # (F,)   – column indices 0..max_features-1
    ) -> torch.Tensor:          # (B, F, d_model)
        B, F = x.shape
        val_emb = self.val_proj(x.unsqueeze(-1))                  # (B, F, d)
        col_emb = self.col_embed(col_ids).unsqueeze(0).expand(B, -1, -1)  # (B, F, d)
        tokens = self.norm(val_emb + col_emb)
        return self.drop(tokens)


# ──────────────────────────── transformer ─────────────────────────────────────

class FeatureTransformer(nn.Module):
    """
    Transformer encoder operating on per-feature tokens plus a [CLS] token.

    The CLS token output is used as the row-level representation.
    """

    def __init__(
        self,
        max_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.tokenizer = FeatureTokenizer(max_features, d_model, dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,        # (B, F)
        col_ids: torch.Tensor,  # (F,)
    ) -> torch.Tensor:          # (B, d_model)
        B = x.size(0)
        feat_tokens = self.tokenizer(x, col_ids)          # (B, F, d)
        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, d)
        seq = torch.cat([cls, feat_tokens], dim=1)        # (B, F+1, d)
        out = self.transformer(seq)                       # (B, F+1, d)
        cls_out = self.out_norm(out[:, 0])               # (B, d)
        return cls_out

    def encode_partitions(
        self,
        x: torch.Tensor,                    # (B, F)
        partitions: list[list[int]],        # list of column-index lists
    ) -> list[torch.Tensor]:                # list of (B, d_model)
        """Encode each partition separately; used in contrastive step."""
        device = x.device
        reps = []
        for part_ids in partitions:
            col_ids = torch.tensor(part_ids, dtype=torch.long, device=device)
            reps.append(self.forward(x[:, part_ids], col_ids))
        return reps


# ──────────────────────────── bottleneck ──────────────────────────────────────

class BottleneckLayer(nn.Module):
    """Compress CLS representation before the classifier."""

    def __init__(self, d_model: int, bottleneck_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────── classifier head ─────────────────────────────────

class MultiLabelHead(nn.Module):
    """
    Dual-head output:
    * clf_logits  – binary logit per index (high / low split at threshold 5.5)
    * reg_out     – continuous prediction in [1, 10]
    """

    def __init__(self, input_dim: int, n_labels: int = 10, dropout: float = 0.1):
        super().__init__()
        hidden = input_dim * 2
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.clf_head = nn.Linear(hidden, n_labels)
        self.reg_head = nn.Linear(hidden, n_labels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        clf_logits = self.clf_head(h)                          # (B, n_labels)
        reg_out = torch.sigmoid(self.reg_head(h)) * 9.0 + 1.0  # [1, 10]
        return clf_logits, reg_out


# ──────────────────────────── full model ──────────────────────────────────────

class DomainAdaptationModel(nn.Module):
    """
    Full model: FeatureTransformer → BottleneckLayer → MultiLabelHead.
    Used in steps 2, 3, 4.
    """

    def __init__(
        self,
        max_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        bottleneck_dim: int = 64,
        n_labels: int = 10,
    ):
        super().__init__()
        self.feature_transformer = FeatureTransformer(
            max_features, d_model, n_heads, n_layers, d_ff, dropout
        )
        self.bottleneck = BottleneckLayer(d_model, bottleneck_dim, dropout)
        self.head = MultiLabelHead(bottleneck_dim, n_labels, dropout)

    def forward(
        self,
        x: torch.Tensor,        # (B, F)
        col_ids: torch.Tensor,  # (F,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.feature_transformer(x, col_ids)
        b = self.bottleneck(h)
        return self.head(b)

    def encode(
        self,
        x: torch.Tensor,
        col_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return bottleneck representation without prediction heads."""
        h = self.feature_transformer(x, col_ids)
        return self.bottleneck(h)

    @classmethod
    def from_config(cls, cfg: dict) -> "DomainAdaptationModel":
        m = cfg["model"]
        return cls(
            max_features=cfg["max_features"],
            d_model=m["d_model"],
            n_heads=m["n_heads"],
            n_layers=m["n_layers"],
            d_ff=m["d_ff"],
            dropout=m["dropout"],
            bottleneck_dim=m["bottleneck_dim"],
            n_labels=cfg["data"]["n_indices"],
        )
