"""
Evaluation metrics:
  – MAE between source and target domain index predictions for common customers
  – AUROC and AUPRC for binary (1–5 vs 6–10) classification
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    roc_auc_score,
    average_precision_score,
)


INDEX_NAMES = [
    "shopping_preference",
    "dining_preference",
    "travel_preference",
    "entertainment_preference",
    "health_wellness",
    "luxury_lifestyle",
    "budget_consciousness",
    "social_activity",
    "work_life_balance",
    "family_stage",
]


def compute_metrics(
    y_true_continuous: np.ndarray,   # (N, n_labels) ground-truth 1–10
    y_pred_continuous: np.ndarray,   # (N, n_labels) predicted 1–10
    y_pred_proba: np.ndarray,        # (N, n_labels) P(high)
    binary_threshold: float = 5.5,
    label_names: list[str] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Compute per-label and macro-averaged MAE, AUROC, AUPRC.

    Parameters
    ----------
    y_true_continuous : ground-truth index values (1–10)
    y_pred_continuous : predicted index values (1–10)
    y_pred_proba      : predicted probability that the index is > threshold
    binary_threshold  : split point for high / low class
    """
    n_labels = y_true_continuous.shape[1]
    label_names = label_names or [f"label_{i}" for i in range(n_labels)]

    y_true_binary = (y_true_continuous > binary_threshold).astype(int)

    per_label: list[dict] = []
    for i, name in enumerate(label_names):
        mae = mean_absolute_error(y_true_continuous[:, i], y_pred_continuous[:, i])

        # Handle single-class edge case
        n_pos = y_true_binary[:, i].sum()
        if n_pos == 0 or n_pos == len(y_true_binary):
            auroc = auprc = float("nan")
        else:
            auroc = roc_auc_score(y_true_binary[:, i], y_pred_proba[:, i])
            auprc = average_precision_score(y_true_binary[:, i], y_pred_proba[:, i])

        per_label.append({"name": name, "mae": mae, "auroc": auroc, "auprc": auprc})

    valid = [d for d in per_label if not np.isnan(d["auroc"])]
    macro_mae = np.mean([d["mae"] for d in per_label])
    macro_auroc = np.mean([d["auroc"] for d in valid]) if valid else float("nan")
    macro_auprc = np.mean([d["auprc"] for d in valid]) if valid else float("nan")

    if verbose:
        print("\n── Evaluation Results ─────────────────────────────────────────")
        print(f"{'Label':<30}  {'MAE':>7}  {'AUROC':>7}  {'AUPRC':>7}")
        print("─" * 60)
        for d in per_label:
            auroc_str = f"{d['auroc']:.4f}" if not np.isnan(d["auroc"]) else "  N/A "
            auprc_str = f"{d['auprc']:.4f}" if not np.isnan(d["auprc"]) else "  N/A "
            print(f"{d['name']:<30}  {d['mae']:>7.4f}  {auroc_str:>7}  {auprc_str:>7}")
        print("─" * 60)
        print(f"{'MACRO':.<30}  {macro_mae:>7.4f}  {macro_auroc:>7.4f}  {macro_auprc:>7.4f}")
        print("─" * 60)

    return {
        "per_label": per_label,
        "macro_mae": macro_mae,
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
    }
