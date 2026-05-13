"""
Feature partitioning utilities for vertical contrastive learning and
common/source-specific feature selection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ────────────────────────── feature groups ────────────────────────────────────

# Semantic names of "common" features – conceptually shared by both domains
# (spending amount proxy, geographic mobility, temporal activity, value tier).
COMMON_FEATURE_PATTERNS = [
    "total_monthly_spend",
    "avg_transaction_amount",
    "n_transactions_monthly",
    "merchant_diversity_cnt",
    "visited_city_cnt",
    "weekend_spend_ratio",
    "evening_spend_ratio",
    "premium_merchant_ratio",
    "luxury",
    "travel",
    "seasonal_ratio_",
    "customer_tenure_months",
    "recency_days",
    "local_merchant_ratio",
    "cashback_usage_rate",
    "installment_ratio",
    "refund_rate",
    "chain_store_ratio",
]


def identify_common_features(source_cols: list[str]) -> tuple[list[str], list[str]]:
    """
    Split source columns into common_features (semantically shared) and
    source_specific_features (credit-card specific).
    """
    common, specific = [], []
    for col in source_cols:
        matched = any(pat in col for pat in COMMON_FEATURE_PATTERNS)
        (common if matched else specific).append(col)
    return common, specific


def vertical_partition(
    feature_cols: list[str],
    n_partitions: int = 4,
    random_seed: int = 0,
) -> list[list[str]]:
    """
    Randomly split feature columns into ``n_partitions`` disjoint subsets.
    Each partition retains at least 1 feature.
    """
    rng = np.random.RandomState(random_seed)
    shuffled = feature_cols.copy()
    rng.shuffle(shuffled)
    return [list(part) for part in np.array_split(shuffled, n_partitions)]


class DomainDataset:
    """Thin wrapper around feature/label DataFrames for training."""

    def __init__(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame | None = None,
        feature_cols: list[str] | None = None,
        label_cols: list[str] | None = None,
    ):
        self.features = features
        self.labels = labels
        self.feature_cols = feature_cols or list(features.columns)
        self.label_cols = label_cols or (list(labels.columns) if labels is not None else [])

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    @property
    def n_labels(self) -> int:
        return len(self.label_cols)

    def get_X(self) -> np.ndarray:
        return self.features[self.feature_cols].values.astype(np.float32)

    def get_y_binary(self, threshold: float = 5.5) -> np.ndarray:
        """Return binary labels (1 if index > threshold)."""
        if self.labels is None:
            raise ValueError("No labels available.")
        return (self.labels[self.label_cols].values > threshold).astype(np.float32)

    def get_y_continuous(self) -> np.ndarray:
        if self.labels is None:
            raise ValueError("No labels available.")
        return self.labels[self.label_cols].values.astype(np.float32)

    def subset_by_ids(self, ids: list[str]) -> "DomainDataset":
        feat = self.features.loc[ids]
        lab = self.labels.loc[ids] if self.labels is not None else None
        return DomainDataset(feat, lab, self.feature_cols, self.label_cols)

    def subset_by_features(self, cols: list[str]) -> "DomainDataset":
        return DomainDataset(
            self.features[cols], self.labels, cols, self.label_cols
        )
