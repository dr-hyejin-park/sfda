"""
Evaluation entry point.

Compares source-domain ground-truth indices against the adapted model's
predictions for common customers in the target domain.
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import torch

from utils.metrics import compute_metrics, INDEX_NAMES


def run_evaluation(
    source_indices: np.ndarray,       # (N_common, n_labels) ground truth from source
    pred_probs: np.ndarray,           # (N_common, n_labels) model probabilities
    pred_reg: np.ndarray,             # (N_common, n_labels) model 1–10 regression
    binary_threshold: float = 5.5,
    save_path: str | None = None,
) -> dict:
    """
    Compute and display evaluation metrics.

    For common customers the source-domain ground truth is compared against
    the target domain model's predictions.
    """
    results = compute_metrics(
        y_true_continuous=source_indices,
        y_pred_continuous=pred_reg,
        y_pred_proba=pred_probs,
        binary_threshold=binary_threshold,
        label_names=INDEX_NAMES,
        verbose=True,
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(
            save_path,
            source_indices=source_indices,
            pred_probs=pred_probs,
            pred_reg=pred_reg,
            macro_mae=results["macro_mae"],
            macro_auroc=results["macro_auroc"],
            macro_auprc=results["macro_auprc"],
        )
        print(f"Results saved to {save_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_npz", required=True,
                        help="Path to results .npz file saved by main.py")
    parser.add_argument("--threshold", type=float, default=5.5)
    args = parser.parse_args()

    data = np.load(args.results_npz)
    run_evaluation(
        source_indices=data["source_indices"],
        pred_probs=data["pred_probs"],
        pred_reg=data["pred_reg"],
        binary_threshold=args.threshold,
    )
