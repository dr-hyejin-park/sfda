"""
Main pipeline orchestrator.

Runs Steps 1–4 end-to-end and evaluates on common customers.

Usage
-----
    python main.py                            # use config/config.yaml defaults
    python main.py --device cuda --epochs1 30
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import torch
import yaml

from data.generate_data import generate_source_domain, generate_target_domain
from data.preprocessing import identify_common_features, vertical_partition
from training.step1_contrastive import train_step1
from training.step2_multilabel import train_step2, infer_step2
from training.step3_source import train_step3
from training.step4_adaptation import train_step4, infer_step4
from evaluate import run_evaluation
from utils.metrics import INDEX_NAMES


# ──────────────────────── helpers ─────────────────────────────────────────────

def _load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _source_bottleneck_reps(
    model,
    X: np.ndarray,
    col_ids: torch.Tensor,
    device: str,
    batch_size: int = 512,
    max_samples: int = 2000,
) -> np.ndarray:
    """Extract bottleneck representations from a subset of source data for MMD."""
    idx = np.random.choice(len(X), min(max_samples, len(X)), replace=False)
    dev = torch.device(device)
    model.eval()
    model.to(dev)
    col_ids = col_ids.to(dev)
    reps = []
    with torch.no_grad():
        for i in range(0, len(idx), batch_size):
            bx = torch.tensor(X[idx[i:i + batch_size]], dtype=torch.float32, device=dev)
            reps.append(model.encode(bx, col_ids).cpu().numpy())
    return np.concatenate(reps)


# ──────────────────────── main ────────────────────────────────────────────────

def main(cfg: dict, device: str = "cpu"):
    rng = np.random.RandomState(cfg["data"]["random_seed"])
    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # DATA GENERATION
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("DATA GENERATION")
    print("=" * 70)

    src_X, src_indices, src_latent, src_qt = generate_source_domain(
        n_customers=cfg["data"]["n_source_customers"],
        random_seed=cfg["data"]["random_seed"],
    )
    print(f"Source: {src_X.shape[0]} customers × {src_X.shape[1]} features")
    print(f"Source indices shape: {src_indices.shape}")

    tgt_X, common_id_map, tgt_latent, tgt_qt = generate_target_domain(
        n_customers=cfg["data"]["n_target_customers"],
        n_common=cfg["data"]["n_common_customers"],
        source_latent=src_latent,
        source_ids=list(src_X.index),
        random_seed=cfg["data"]["random_seed"],
    )
    print(f"Target: {tgt_X.shape[0]} customers × {tgt_X.shape[1]} features")
    print(f"Common customers: {len(common_id_map)}")

    # Identify feature splits
    src_cols = list(src_X.columns)
    common_feat_cols, specific_feat_cols = identify_common_features(src_cols)
    print(f"Common features  : {len(common_feat_cols)}")
    print(f"Specific features: {len(specific_feat_cols)}")

    # Map column names → integer IDs in source vocabulary
    src_col2id = {col: i for i, col in enumerate(src_cols)}
    common_feat_ids = [src_col2id[c] for c in common_feat_cols]
    specific_feat_ids = [src_col2id[c] for c in specific_feat_cols]

    # Common customer arrays (source side)
    src_common_ids = [sid for sid in src_X.index if sid in common_id_map]
    tgt_common_ids = [common_id_map[sid] for sid in src_common_ids]

    src_common_indices = src_indices.loc[src_common_ids].values  # ground truth
    src_common_X_common = src_X.loc[src_common_ids, common_feat_cols].values
    src_common_X_specific = src_X.loc[src_common_ids, specific_feat_cols].values

    # Source binary labels (all customers)
    bin_thresh = cfg["evaluation"]["binary_threshold"]
    src_y_binary = (src_indices.values > bin_thresh).astype(np.float32)
    src_y_cont = src_indices.values.astype(np.float32)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 – VERTICAL PARTITIONING CONTRASTIVE LEARNING
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 1 – Vertical Partitioning Contrastive Learning")
    print("=" * 70)

    pretrained_ft = train_step1(
        source_X=src_X.values.astype(np.float32),
        feature_cols=src_cols,
        cfg=cfg,
        device=device,
        verbose=True,
    )
    ckpt_s1 = os.path.join(cfg["paths"]["checkpoint_dir"], "step1_ft.pt")
    torch.save(pretrained_ft.state_dict(), ckpt_s1)
    print(f"Step1 checkpoint saved → {ckpt_s1}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 – COMMON-FEATURE MULTI-LABEL TRAINING
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 2 – Common-Feature Multi-label Training")
    print("=" * 70)

    src_common_yb = (src_indices.loc[src_common_ids].values > bin_thresh).astype(np.float32)
    src_common_yc = src_indices.loc[src_common_ids].values.astype(np.float32)

    model_step2 = train_step2(
        source_X_common=src_common_X_common.astype(np.float32),
        source_y_binary=src_common_yb,
        source_y_continuous=src_common_yc,
        common_feature_indices=common_feat_ids,
        pretrained_ft=pretrained_ft,
        cfg=cfg,
        device=device,
        verbose=True,
    )
    ckpt_s2 = os.path.join(cfg["paths"]["checkpoint_dir"], "step2_model.pt")
    torch.save(model_step2.state_dict(), ckpt_s2)
    print(f"Step2 checkpoint saved → {ckpt_s2}")

    # ══════════════════════════════════════════════════════════════════════════
    # GENERATE PSEUDO LABELS for target domain common customers
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── Generating pseudo labels for common target customers …")

    # Common customers in target domain, common feature subset
    # (target domain doesn't have source features; we use the target's own
    #  representation of "common" features – for simulation we approximate
    #  by using the source common-feature values with slight noise, since in
    #  practice the step-2 model would be applied via the common feature bridge.)
    # In a real system this would use the actual target-side common features.
    noise = np.random.RandomState(99).randn(*src_common_X_common.shape) * 0.1
    tgt_common_X_common = src_common_X_common + noise  # simulated target-side common features

    col_ids_common_t = torch.tensor(common_feat_ids, dtype=torch.long)
    pseudo_probs, pseudo_reg = infer_step2(
        model=model_step2,
        X=tgt_common_X_common.astype(np.float32),
        col_ids_tensor=col_ids_common_t,
        device=device,
        batch_size=512,
    )
    print(f"  Pseudo labels generated for {len(pseudo_probs)} common customers.")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 – SOURCE-SPECIFIC FEATURE TRAINING
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 3 – Source-Specific Feature Training")
    print("=" * 70)

    model_step3 = train_step3(
        source_X_specific=src_X[specific_feat_cols].values.astype(np.float32),
        source_y_binary=src_y_binary,
        source_y_continuous=src_y_cont,
        specific_feature_indices=specific_feat_ids,
        cfg=cfg,
        n_total_source_features=len(src_cols),  # ensures target IDs are correctly offset
        device=device,
        verbose=True,
    )
    ckpt_s3 = os.path.join(cfg["paths"]["checkpoint_dir"], "step3_model.pt")
    torch.save(model_step3.state_dict(), ckpt_s3)
    print(f"Step3 checkpoint saved → {ckpt_s3}")

    # Source bottleneck reps for MMD alignment in Step 4
    src_specific_col_ids_t = torch.tensor(specific_feat_ids, dtype=torch.long)
    src_bottleneck = _source_bottleneck_reps(
        model=model_step3,
        X=src_X[specific_feat_cols].values.astype(np.float32),
        col_ids=src_specific_col_ids_t,
        device=device,
    )
    print(f"  Source bottleneck reps: {src_bottleneck.shape}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 – TARGET DOMAIN ADAPTATION
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 4 – Target Domain Adaptation")
    print("=" * 70)

    # Assign target feature col IDs as an extension of the source vocabulary
    n_src_vocab = len(src_cols)
    tgt_cols = list(tgt_X.columns)
    tgt_col_ids = list(range(n_src_vocab, n_src_vocab + len(tgt_cols)))

    # Build common-customer mask for target domain (in target row order)
    tgt_ids_ordered = list(tgt_X.index)
    common_target_set = set(tgt_common_ids)
    target_common_mask = np.array([tid in common_target_set for tid in tgt_ids_ordered])

    # Reorder pseudo labels to match target row order
    tgt_common_id_to_local = {tid: i for i, tid in enumerate(tgt_common_ids)}
    # pseudo labels are already in tgt_common_ids order; keep as-is

    model_step4 = train_step4(
        target_X=tgt_X.values.astype(np.float32),
        target_col_ids=tgt_col_ids,
        target_common_mask=target_common_mask,
        pseudo_probs=pseudo_probs,
        pseudo_reg=pseudo_reg,
        source_bottleneck_reps=src_bottleneck,
        student_model=model_step3,
        n_source_features=n_src_vocab,
        cfg=cfg,
        device=device,
        verbose=True,
    )
    ckpt_s4 = os.path.join(cfg["paths"]["checkpoint_dir"], "step4_adapted.pt")
    torch.save(model_step4.state_dict(), ckpt_s4)
    print(f"Step4 checkpoint saved → {ckpt_s4}")

    # ══════════════════════════════════════════════════════════════════════════
    # INFERENCE ON ALL TARGET CUSTOMERS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── Running inference on all target customers …")
    col_ids_tgt_t = torch.tensor(tgt_col_ids, dtype=torch.long)
    all_probs, all_reg = infer_step4(
        model=model_step4,
        X=tgt_X.values.astype(np.float32),
        col_ids_tensor=col_ids_tgt_t,
        device=device,
        batch_size=512,
    )
    print(f"  Inference done. Output shape: {all_probs.shape}")

    # Save full prediction table
    pred_df_path = os.path.join(cfg["paths"]["results_dir"], "target_predictions.npz")
    np.savez(
        pred_df_path,
        customer_ids=np.array(tgt_ids_ordered),
        pred_probs=all_probs,
        pred_reg=all_reg,
        index_names=np.array(INDEX_NAMES),
    )
    print(f"Full target predictions saved → {pred_df_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # EVALUATION on common customers
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EVALUATION (common customers)")
    print("=" * 70)

    # Collect predictions for common customers (in target-domain row order)
    common_row_idx = [i for i, tid in enumerate(tgt_ids_ordered) if tid in common_target_set]
    common_probs = all_probs[common_row_idx]
    common_reg = all_reg[common_row_idx]

    # Ground-truth indices from source domain (same order as tgt_common_ids)
    src_common_gt = src_indices.loc[src_common_ids].values.astype(np.float32)

    results = run_evaluation(
        source_indices=src_common_gt,
        pred_probs=common_probs,
        pred_reg=common_reg,
        binary_threshold=bin_thresh,
        save_path=os.path.join(cfg["paths"]["results_dir"], "eval_results.npz"),
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"  Macro MAE   : {results['macro_mae']:.4f}")
    print(f"  Macro AUROC : {results['macro_auroc']:.4f}")
    print(f"  Macro AUPRC : {results['macro_auprc']:.4f}")
    print("=" * 70)
    return results


# ──────────────────────── CLI ──────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFDA Pipeline")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--device", default="cpu",
                        help="torch device: cpu | cuda | cuda:0")
    parser.add_argument("--fast", action="store_true",
                        help="Cut epochs for quick smoke-test")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    if args.fast:
        cfg["training"]["step1"]["epochs"] = 3
        cfg["training"]["step2"]["epochs"] = 3
        cfg["training"]["step3"]["epochs"] = 3
        cfg["training"]["step4"]["epochs"] = 5
        cfg["data"]["n_source_customers"] = 1000
        cfg["data"]["n_target_customers"] = 500
        cfg["data"]["n_common_customers"] = 100
        print("[FAST MODE] Reduced epochs and dataset sizes for smoke-test.")

    main(cfg, device=args.device)
