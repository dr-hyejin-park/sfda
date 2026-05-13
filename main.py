"""
Main pipeline orchestrator – TransTab tokenization version.

Runs Steps 1–4 end-to-end and evaluates on common customers.

Usage
-----
    python main.py                   # use config/config.yaml defaults
    python main.py --device cuda
    python main.py --fast            # smoke-test (tiny data, few epochs)
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import torch
import yaml

from data.generate_data import generate_source_domain, generate_target_domain
from data.preprocessing import (
    build_vocab, encode_schema,
    identify_common_features,
    FeatureSchema,
)
from training.step1_contrastive import train_step1
from training.step2_multilabel import train_step2, infer_step2
from training.step3_source import train_step3
from training.step4_adaptation import train_step4, infer_step4
from evaluate import run_evaluation
from utils.metrics import INDEX_NAMES


# ─────────────────────────────── helpers ──────────────────────────────────────

def _load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _source_bottleneck_reps(
    model, x_num, x_bin, x_cat, schema_enc, device: str,
    batch_size: int = 512, max_samples: int = 2000,
) -> np.ndarray:
    from data.preprocessing import arrays_to_tensors, unpack_batch
    n = (x_num if x_num is not None
         else x_bin if x_bin is not None else x_cat).shape[0]
    idx = np.random.choice(n, min(max_samples, n), replace=False)
    dev = torch.device(device)
    model.eval(); model.to(dev)
    schema_dev = schema_enc.to(dev)
    x_num_s  = x_num[idx]  if x_num  is not None else None
    x_bin_s  = x_bin[idx]  if x_bin  is not None else None
    x_cat_s  = x_cat[idx]  if x_cat  is not None else None
    t_num, t_bin, t_cat = arrays_to_tensors(x_num_s, x_bin_s, x_cat_s, len(idx))
    reps = []
    with torch.no_grad():
        for i in range(0, len(idx), batch_size):
            bn, bb, bc = unpack_batch(
                t_num[i:i+batch_size].to(dev),
                t_bin[i:i+batch_size].to(dev),
                t_cat[i:i+batch_size].to(dev),
            )
            reps.append(model.encode(bn, bb, bc, schema_dev).cpu().numpy())
    return np.concatenate(reps)


# ─────────────────────────────── main ─────────────────────────────────────────

def main(cfg: dict, device: str = "cpu"):
    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["results_dir"],    exist_ok=True)
    bin_thresh = cfg["evaluation"]["binary_threshold"]

    # ══════════════════════════════════════════════════════════════════════════
    # DATA GENERATION
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("DATA GENERATION")
    print("=" * 70)

    (src_num, src_bin, src_cat,
     src_indices, src_latent, src_qt, src_schema) = generate_source_domain(
        n_customers=cfg["data"]["n_source_customers"],
        random_seed=cfg["data"]["random_seed"],
    )
    print(f"Source: {len(src_num)} customers  "
          f"num={src_schema.n_num}  bin={src_schema.n_bin}  cat={src_schema.n_cat}")

    (tgt_num, tgt_bin, tgt_cat,
     common_id_map, tgt_latent, tgt_qt, tgt_schema) = generate_target_domain(
        n_customers=cfg["data"]["n_target_customers"],
        n_common=cfg["data"]["n_common_customers"],
        source_latent=src_latent,
        source_ids=list(src_num.index),
        random_seed=cfg["data"]["random_seed"],
    )
    print(f"Target: {len(tgt_num)} customers  "
          f"num={tgt_schema.n_num}  bin={tgt_schema.n_bin}  cat={tgt_schema.n_cat}")
    print(f"Common customers: {len(common_id_map)}")

    # ── build shared vocabulary & schema encodings ────────────────────────────
    vocab = build_vocab(src_schema, tgt_schema)
    print(f"Shared vocabulary size: {vocab.size} words")

    src_schema_enc  = encode_schema(src_schema, vocab)
    tgt_schema_enc  = encode_schema(tgt_schema, vocab)

    # ── identify common vs source-specific NUMERICAL features ─────────────────
    common_num_cols, specific_num_cols = identify_common_features(src_schema)
    print(f"Common num features: {len(common_num_cols)}  "
          f"Specific num features: {len(specific_num_cols)}")

    # Common-feature schema (Step 2): only common numerical, no bin/cat
    common_schema = FeatureSchema(
        numerical_cols=common_num_cols,
        binary_cols=[],
        categorical_cols=[],
        cat_value_strings={},
    )
    common_schema_enc = encode_schema(common_schema, vocab)

    # Specific-feature schema (Step 3): specific-num + ALL src bin + cat
    specific_schema = FeatureSchema(
        numerical_cols=specific_num_cols,
        binary_cols=src_schema.binary_cols,
        categorical_cols=src_schema.categorical_cols,
        cat_value_strings=src_schema.cat_value_strings,
    )
    specific_schema_enc = encode_schema(specific_schema, vocab)

    # ── identify common customers ─────────────────────────────────────────────
    src_ids      = list(src_num.index)
    src_common_ids = [sid for sid in src_ids if sid in common_id_map]
    tgt_common_ids = [common_id_map[sid] for sid in src_common_ids]
    n_common = len(src_common_ids)

    # ── source numpy arrays ───────────────────────────────────────────────────
    src_X_num = src_num.values.astype(np.float32)
    src_X_bin = src_bin.values.astype(np.int64)
    src_X_cat = src_cat.values.astype(np.int64)

    # common-customer arrays (common features only)
    src_common_X_num_common = src_num.loc[src_common_ids, common_num_cols].values.astype(np.float32)
    src_common_yb = (src_indices.loc[src_common_ids].values > bin_thresh).astype(np.float32)
    src_common_yc = src_indices.loc[src_common_ids].values.astype(np.float32)

    # source-specific feature arrays (all source customers)
    src_X_num_specific = src_num[specific_num_cols].values.astype(np.float32)
    src_X_bin_specific = src_X_bin          # all binary features are source-specific
    src_X_cat_specific = src_X_cat          # all categorical features are source-specific
    src_yb = (src_indices.values > bin_thresh).astype(np.float32)
    src_yc =  src_indices.values.astype(np.float32)

    # ── target numpy arrays ───────────────────────────────────────────────────
    tgt_X_num  = tgt_num.values.astype(np.float32)
    tgt_X_bin  = tgt_bin.values.astype(np.int64)
    tgt_X_cat  = tgt_cat.values.astype(np.int64)
    tgt_ids    = list(tgt_num.index)
    common_tgt_set = set(tgt_common_ids)
    target_common_mask = np.array([tid in common_tgt_set for tid in tgt_ids])

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 – VERTICAL PARTITIONING CONTRASTIVE LEARNING
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 1 – Vertical Partitioning Contrastive Learning")
    print("=" * 70)

    pretrained_ft = train_step1(
        source_x_num = src_X_num,
        source_x_bin = src_X_bin,
        source_x_cat = src_X_cat,
        schema_enc   = src_schema_enc,
        vocab_size   = vocab.size,
        cfg=cfg, device=device, verbose=True,
    )
    ckpt = os.path.join(cfg["paths"]["checkpoint_dir"], "step1_ft.pt")
    torch.save(pretrained_ft.state_dict(), ckpt)
    print(f"Step1 checkpoint → {ckpt}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 – COMMON-FEATURE MULTI-LABEL TRAINING
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 2 – Common-Feature Multi-label Training")
    print("=" * 70)

    model_step2 = train_step2(
        x_num_common      = src_common_X_num_common,
        x_bin_common      = None,   # no binary common features
        x_cat_common      = None,   # no categorical common features
        common_schema_enc = common_schema_enc,
        y_binary          = src_common_yb,
        y_continuous      = src_common_yc,
        pretrained_ft     = pretrained_ft,
        vocab_size        = vocab.size,
        cfg=cfg, device=device, verbose=True,
    )
    ckpt = os.path.join(cfg["paths"]["checkpoint_dir"], "step2_model.pt")
    torch.save(model_step2.state_dict(), ckpt)
    print(f"Step2 checkpoint → {ckpt}")

    # ── pseudo labels for target common customers ─────────────────────────────
    print("\n── Generating pseudo labels for common target customers …")
    # Simulate target-side common features (source features + slight noise)
    rng = np.random.RandomState(99)
    tgt_common_X_num_common = src_common_X_num_common + rng.randn(*src_common_X_num_common.shape) * 0.1
    pseudo_probs, pseudo_reg = infer_step2(
        model       = model_step2,
        x_num       = tgt_common_X_num_common.astype(np.float32),
        x_bin       = None,
        x_cat       = None,
        schema_enc  = common_schema_enc,
        device=device, batch_size=512,
    )
    print(f"  Pseudo labels generated for {len(pseudo_probs)} common customers.")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 – SOURCE-SPECIFIC FEATURE TRAINING
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 3 – Source-Specific Feature Training  "
          f"(num={specific_schema.n_num}  bin={specific_schema.n_bin}  "
          f"cat={specific_schema.n_cat})")
    print("=" * 70)

    model_step3 = train_step3(
        x_num_specific     = src_X_num_specific,
        x_bin_specific     = src_X_bin_specific,
        x_cat_specific     = src_X_cat_specific,
        specific_schema_enc = specific_schema_enc,
        y_binary           = src_yb,
        y_continuous       = src_yc,
        vocab_size         = vocab.size,
        cfg=cfg, device=device, verbose=True,
    )
    ckpt = os.path.join(cfg["paths"]["checkpoint_dir"], "step3_model.pt")
    torch.save(model_step3.state_dict(), ckpt)
    print(f"Step3 checkpoint → {ckpt}")

    # Source bottleneck reps for MMD alignment in Step 4
    src_bottleneck = _source_bottleneck_reps(
        model_step3, src_X_num_specific, src_X_bin_specific, src_X_cat_specific,
        specific_schema_enc, device,
    )
    print(f"  Source bottleneck reps: {src_bottleneck.shape}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 – TARGET DOMAIN ADAPTATION
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 4 – Target Domain Adaptation")
    print("=" * 70)

    model_step4 = train_step4(
        tgt_x_num            = tgt_X_num,
        tgt_x_bin            = tgt_X_bin,
        tgt_x_cat            = tgt_X_cat,
        tgt_schema_enc       = tgt_schema_enc,
        target_common_mask   = target_common_mask,
        pseudo_probs         = pseudo_probs,
        pseudo_reg           = pseudo_reg,
        source_bottleneck_reps = src_bottleneck,
        student_model        = model_step3,
        cfg=cfg, device=device, verbose=True,
    )
    ckpt = os.path.join(cfg["paths"]["checkpoint_dir"], "step4_adapted.pt")
    torch.save(model_step4.state_dict(), ckpt)
    print(f"Step4 checkpoint → {ckpt}")

    # ══════════════════════════════════════════════════════════════════════════
    # INFERENCE ON ALL TARGET CUSTOMERS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── Running inference on all target customers …")
    all_probs, all_reg = infer_step4(
        model=model_step4, x_num=tgt_X_num, x_bin=tgt_X_bin, x_cat=tgt_X_cat,
        schema_enc=tgt_schema_enc, device=device, batch_size=512,
    )
    print(f"  Inference done. Output shape: {all_probs.shape}")

    np.savez(
        os.path.join(cfg["paths"]["results_dir"], "target_predictions.npz"),
        customer_ids=np.array(tgt_ids),
        pred_probs=all_probs,
        pred_reg=all_reg,
        index_names=np.array(INDEX_NAMES),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # EVALUATION on common customers
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EVALUATION (common customers)")
    print("=" * 70)

    common_row_idx = [i for i, tid in enumerate(tgt_ids) if tid in common_tgt_set]
    common_probs   = all_probs[common_row_idx]
    common_reg     = all_reg[common_row_idx]
    src_gt         = src_indices.loc[src_common_ids].values.astype(np.float32)

    results = run_evaluation(
        source_indices=src_gt,
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


# ─────────────────────────────── CLI ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFDA Pipeline – TransTab tokenization")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fast",   action="store_true",
                        help="Tiny data + minimal epochs for smoke-test")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    if args.fast:
        cfg["training"]["step1"]["epochs"] = 3
        cfg["training"]["step2"]["epochs"] = 3
        cfg["training"]["step3"]["epochs"] = 3
        cfg["training"]["step4"]["epochs"] = 5
        cfg["data"]["n_source_customers"]  = 1000
        cfg["data"]["n_target_customers"]  = 500
        cfg["data"]["n_common_customers"]  = 100
        print("[FAST MODE] Reduced epochs and dataset sizes for smoke-test.")

    main(cfg, device=args.device)
