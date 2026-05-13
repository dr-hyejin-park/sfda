"""
Synthetic data generation for source (credit card) and target (airline) domains.

Feature types follow the TransTab paper convention:
  - Numerical  : quantile-transformed to N(0,1)
  - Binary     : raw {0, 1} integer – NOT quantile-transformed
  - Categorical: raw integer category codes – NOT quantile-transformed

Each domain exposes a FeatureSchema (defined in preprocessing.py) that
records which columns belong to which type, plus the string names of
categorical values used to build the shared word vocabulary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import QuantileTransformer

from data.preprocessing import FeatureSchema


# ─────────────────────────────── helpers ──────────────────────────────────────

def _make_index(latent_combo: np.ndarray, rng: np.random.RandomState,
                noise_std: float = 0.6) -> np.ndarray:
    """Map latent score to 1–10 index via normal CDF."""
    raw = latent_combo + rng.randn(len(latent_combo)) * noise_std
    return np.clip(stats.norm.cdf(raw) * 9 + 1, 1.0, 10.0)


# ────────────────────────── SOURCE DOMAIN ─────────────────────────────────────

# Categorical value string maps (shared vocabulary for TransTab word embedding)
SOURCE_CAT_VALUE_STRINGS: dict[str, list[str]] = {
    "card_tier":         ["standard", "gold", "platinum"],
    "customer_segment":  ["new", "growth", "loyal", "vip"],
}


def generate_source_domain(
    n_customers: int = 10_000,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
           np.ndarray, QuantileTransformer, FeatureSchema]:
    """
    Credit-card offline payment history for source domain customers.

    Returns
    -------
    x_num_qt : pd.DataFrame  – quantile-transformed numerical features
    x_bin    : pd.DataFrame  – binary {0,1} integer features
    x_cat    : pd.DataFrame  – categorical integer-code features
    indices  : pd.DataFrame  – ground-truth 1–10 preference indices
    latent   : np.ndarray    – latent profiles (N × 10)
    qt       : QuantileTransformer fitted on raw numerical features
    schema   : FeatureSchema
    """
    rng = np.random.RandomState(random_seed)
    n = n_customers
    cids = [f"SRC_{i:05d}" for i in range(n)]

    L = rng.randn(n, 10)  # latent profiles

    num: dict[str, np.ndarray] = {}
    bin_: dict[str, np.ndarray] = {}
    cat: dict[str, np.ndarray] = {}

    # ── numerical: spending amount per merchant category ──────────────────────
    spend_cats = [
        "grocery", "restaurant", "fashion", "beauty", "travel",
        "entertainment", "fuel", "pharmacy", "electronics", "sports",
        "home_deco", "luxury", "education", "kids", "pet",
    ]
    for i, c in enumerate(spend_cats):
        base = np.exp(L[:, i % 10] * 0.5 + rng.randn(n) * 0.4 + 4.0)
        num[f"spend_{c}_monthly"] = base

    # ── numerical: transaction frequency per category ─────────────────────────
    for i, c in enumerate(spend_cats):
        freq = np.maximum(0, L[:, (i + 2) % 10] * 2 + rng.randn(n) + 5)
        num[f"freq_{c}_monthly"] = freq

    # ── numerical: temporal patterns ──────────────────────────────────────────
    num["weekend_spend_ratio"] = np.clip(L[:, 0] * 0.1 + 0.42 + rng.randn(n) * 0.08, 0, 1)
    num["evening_spend_ratio"] = np.clip(L[:, 1] * 0.08 + 0.35 + rng.randn(n) * 0.07, 0, 1)
    num["midnight_spend_ratio"] = np.clip(rng.beta(1, 10, n), 0, 0.15)

    # ── numerical: merchant diversity & geography ─────────────────────────────
    num["merchant_diversity_cnt"] = np.maximum(1, L[:, 3] * 5 + rng.randn(n) * 3 + 22)
    num["local_merchant_ratio"] = np.clip(L[:, 4] * 0.1 + 0.58 + rng.randn(n) * 0.08, 0, 1)
    num["chain_store_ratio"] = np.clip(1 - num["local_merchant_ratio"] + rng.randn(n) * 0.04, 0, 1)
    num["visited_city_cnt"] = np.maximum(1, L[:, 4] * 2 + rng.randn(n) * 1.5 + 4)

    # ── numerical: payment behaviour ──────────────────────────────────────────
    total_spend = sum(num[f"spend_{c}_monthly"] for c in spend_cats)
    total_freq = sum(num[f"freq_{c}_monthly"] for c in spend_cats)
    num["total_monthly_spend"] = total_spend
    num["avg_transaction_amount"] = total_spend / np.maximum(1, total_freq)
    num["n_transactions_monthly"] = total_freq
    num["premium_merchant_ratio"] = np.clip(L[:, 5] * 0.1 + 0.18 + rng.randn(n) * 0.05, 0, 1)
    num["installment_ratio"] = np.clip(L[:, 6] * 0.08 + 0.28 + rng.randn(n) * 0.05, 0, 1)
    num["cashback_usage_rate"] = np.clip(L[:, 7] * 0.08 + 0.38 + rng.randn(n) * 0.05, 0, 1)
    num["refund_rate"] = np.clip(rng.beta(1, 20, n), 0, 0.15)

    # ── numerical: seasonal spend ratios ──────────────────────────────────────
    season_bases = rng.dirichlet(np.ones(4) * 3, n)
    for k, s in enumerate(["spring", "summer", "fall", "winter"]):
        num[f"seasonal_ratio_{s}"] = season_bases[:, k]

    # ── numerical: RFM ────────────────────────────────────────────────────────
    num["recency_days"] = np.maximum(0, -L[:, 9] * 5 + rng.randn(n) * 2 + 7)
    num["customer_tenure_months"] = np.maximum(1, rng.gamma(3, 12, n))

    # ── binary features ───────────────────────────────────────────────────────
    # is_premium_member: 1 if premium merchant ratio above threshold
    bin_["is_premium_member"] = (
        (num["premium_merchant_ratio"] > 0.2).astype(np.int64)
    )
    # has_cashback_benefit: driven by latent + noise
    bin_["has_cashback_benefit"] = (
        (stats.norm.cdf(L[:, 7] + rng.randn(n) * 0.5) > 0.5).astype(np.int64)
    )

    # ── categorical features ──────────────────────────────────────────────────
    # card_tier: 0=standard, 1=gold, 2=platinum
    tier_logit = L[:, 7] * 0.6 + L[:, 5] * 0.3 + rng.randn(n) * 0.3
    cat["card_tier"] = np.searchsorted([-0.5, 0.5], tier_logit).astype(np.int64)

    # customer_segment: 0=new, 1=growth, 2=loyal, 3=vip
    seg_logit = L[:, 8] * 0.5 + num["customer_tenure_months"] / 30 + rng.randn(n) * 0.3
    cat["customer_segment"] = np.searchsorted([-0.5, 0.3, 1.0], seg_logit).astype(np.int64)

    # ── build DataFrames ──────────────────────────────────────────────────────
    num_df = pd.DataFrame(num, index=cids)
    bin_df = pd.DataFrame(bin_, index=cids)
    cat_df = pd.DataFrame(cat, index=cids)

    # ── ground-truth indices ──────────────────────────────────────────────────
    indices = {
        "shopping_preference":      _make_index(L[:, 0] + L[:, 2] * 0.5, rng),
        "dining_preference":        _make_index(L[:, 1] + L[:, 3] * 0.5, rng),
        "travel_preference":        _make_index(L[:, 4] * 1.4, rng),
        "entertainment_preference": _make_index(L[:, 5] + L[:, 1] * 0.4, rng),
        "health_wellness":          _make_index(L[:, 6] - L[:, 0] * 0.3, rng),
        "luxury_lifestyle":         _make_index(L[:, 7] + L[:, 5] * 0.5, rng),
        "budget_consciousness":     _make_index(-L[:, 7] + L[:, 8] * 0.6, rng),
        "social_activity":          _make_index(L[:, 1] + L[:, 3] + L[:, 0] * 0.3, rng),
        "work_life_balance":        _make_index(L[:, 9], rng),
        "family_stage":             _make_index(L[:, 8] + L[:, 6] * 0.3, rng),
    }
    indices_df = pd.DataFrame(indices, index=cids)

    # ── quantile-transform numerical features only ────────────────────────────
    qt = QuantileTransformer(
        n_quantiles=min(1000, n), output_distribution="normal", random_state=random_seed
    )
    num_qt = qt.fit_transform(num_df.values)
    num_qt_df = pd.DataFrame(num_qt, columns=num_df.columns, index=cids)

    schema = FeatureSchema(
        numerical_cols=list(num_df.columns),
        binary_cols=list(bin_df.columns),
        categorical_cols=list(cat_df.columns),
        cat_value_strings=SOURCE_CAT_VALUE_STRINGS,
    )
    return num_qt_df, bin_df, cat_df, indices_df, L, qt, schema


# ────────────────────────── TARGET DOMAIN ─────────────────────────────────────

TARGET_CAT_VALUE_STRINGS: dict[str, list[str]] = {
    "loyalty_tier":   ["bronze", "silver", "gold", "platinum"],
    "preferred_cabin": ["economy", "business", "first_class"],
}


def generate_target_domain(
    n_customers: int = 5_000,
    n_common: int = 1_000,
    source_latent: np.ndarray | None = None,
    source_ids: list[str] | None = None,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           dict[str, str], np.ndarray, QuantileTransformer, FeatureSchema]:
    """
    Airline mileage / boarding-pass payment history for target domain customers.

    Returns
    -------
    x_num_qt      : pd.DataFrame
    x_bin         : pd.DataFrame
    x_cat         : pd.DataFrame
    common_id_map : dict  src_id → tgt_id
    latent        : np.ndarray
    qt            : QuantileTransformer
    schema        : FeatureSchema
    """
    rng = np.random.RandomState(random_seed + 1)
    n = n_customers

    common_ids = [f"CMN_{i:05d}" for i in range(n_common)]
    unique_ids = [f"TGT_{i:05d}" for i in range(n - n_common)]
    cids = common_ids + unique_ids

    if source_latent is not None and n_common > 0:
        common_L = source_latent[:n_common] + rng.randn(n_common, source_latent.shape[1]) * 0.25
        unique_L = rng.randn(n - n_common, source_latent.shape[1])
        L = np.vstack([common_L, unique_L])
    else:
        L = rng.randn(n, 10)

    num: dict[str, np.ndarray] = {}
    bin_: dict[str, np.ndarray] = {}
    cat: dict[str, np.ndarray] = {}

    # ── numerical: mileage ────────────────────────────────────────────────────
    num["miles_earned_total"] = np.maximum(0, np.exp(L[:, 4] * 0.8 + rng.randn(n) * 0.5 + 8))
    num["miles_redeemed_total"] = num["miles_earned_total"] * np.clip(rng.beta(2, 3, n), 0.1, 0.9)
    num["miles_balance"] = num["miles_earned_total"] - num["miles_redeemed_total"]
    num["miles_expiry_risk_ratio"] = np.clip(rng.beta(1, 6, n), 0, 1)
    num["miles_earn_rate_per_flight"] = np.maximum(100, L[:, 7] * 300 + rng.randn(n) * 100 + 1200)

    # ── numerical: flight frequency ───────────────────────────────────────────
    num["domestic_flights_yearly"] = np.maximum(0, L[:, 4] * 3 + rng.randn(n) * 2 + 6)
    num["intl_flights_yearly"] = np.maximum(0, L[:, 4] * 1.5 + rng.randn(n) * 1 + 2)
    num["total_flights_yearly"] = num["domestic_flights_yearly"] + num["intl_flights_yearly"]
    num["flight_yoy_growth"] = rng.randn(n) * 0.2

    # ── numerical: booking class mix ──────────────────────────────────────────
    alphas = np.stack([
        np.clip(5 - L[:, 7], 0.5, 10),
        np.clip(1 + L[:, 7], 0.1, 5),
        np.clip(0.3 + L[:, 7] * 0.2 + 0.2, 0.05, 2),
    ], axis=1)
    class_mix = np.array([rng.dirichlet(alphas[i]) for i in range(n)])
    num["economy_ratio"] = class_mix[:, 0]
    num["business_ratio"] = class_mix[:, 1]
    num["first_class_ratio"] = class_mix[:, 2]

    # ── numerical: booking behaviour ──────────────────────────────────────────
    num["avg_advance_booking_days"] = np.maximum(1, L[:, 9] * 10 + rng.randn(n) * 5 + 28)
    num["last_minute_ratio"] = np.clip(-L[:, 9] * 0.05 + 0.18 + rng.randn(n) * 0.05, 0, 0.5)
    num["direct_booking_ratio"] = np.clip(L[:, 0] * 0.05 + 0.52 + rng.randn(n) * 0.08, 0, 1)
    num["mobile_booking_ratio"] = np.clip(rng.beta(4, 2, n), 0.2, 1)

    # ── numerical: in-flight spending ─────────────────────────────────────────
    num["inflight_meal_spend_per_flight"] = np.maximum(0, L[:, 1] * 5 + rng.randn(n) * 3 + 14)
    num["inflight_duty_free_spend_yearly"] = np.maximum(0, L[:, 7] * 20 + rng.randn(n) * 10 + 30)
    num["inflight_entertainment_spend_per_flight"] = np.maximum(0, L[:, 5] * 3 + rng.randn(n) * 2 + 5)
    num["inflight_wifi_usage_ratio"] = np.clip(L[:, 9] * 0.1 + 0.4 + rng.randn(n) * 0.1, 0, 1)

    # ── numerical: lounge ─────────────────────────────────────────────────────
    num["lounge_visits_yearly"] = np.maximum(0, L[:, 7] * 4 + rng.randn(n) * 2 + 3)
    num["lounge_spend_yearly"] = num["lounge_visits_yearly"] * np.maximum(
        0, L[:, 7] * 20 + rng.randn(n) * 10 + 50)

    # ── numerical: destination / trip type ───────────────────────────────────
    num["destination_diversity"] = np.maximum(1, L[:, 3] * 3 + rng.randn(n) * 2 + 7)
    num["leisure_flight_ratio"] = np.clip(L[:, 4] * 0.1 + 0.62 + rng.randn(n) * 0.1, 0.2, 1)
    num["business_trip_ratio"] = np.clip(1 - num["leisure_flight_ratio"] + rng.randn(n) * 0.02, 0, 1)
    num["avg_trip_duration_days"] = np.maximum(1, L[:, 4] + rng.randn(n) * 1.5 + 4)

    # ── numerical: ticket spend ───────────────────────────────────────────────
    num["avg_ticket_price"] = np.maximum(80, np.exp(L[:, 7] * 0.3 + rng.randn(n) * 0.3 + 5.5))
    num["total_ticket_spend_yearly"] = num["avg_ticket_price"] * num["total_flights_yearly"]

    # ── numerical: seasonal ───────────────────────────────────────────────────
    season_bases = rng.dirichlet(np.ones(4) * 3, n)
    for k, s in enumerate(["spring", "summer", "fall", "winter"]):
        num[f"flights_{s}_ratio"] = season_bases[:, k]

    # ── binary features ───────────────────────────────────────────────────────
    tier_logit = L[:, 4] * 0.5 + L[:, 7] * 0.3 + rng.randn(n) * 0.3
    # has_lounge_access: tier ≥ gold (≥2)
    raw_tier = np.searchsorted([-0.43, 0.43, 1.07], tier_logit).astype(np.int64)
    bin_["has_lounge_access"] = (raw_tier >= 2).astype(np.int64)
    # has_elite_status: tier ≥ platinum (=3)
    bin_["has_elite_status"] = (raw_tier >= 3).astype(np.int64)

    # ── categorical features ──────────────────────────────────────────────────
    # loyalty_tier: 0=bronze, 1=silver, 2=gold, 3=platinum
    cat["loyalty_tier"] = raw_tier

    # preferred_cabin: 0=economy, 1=business, 2=first_class
    cabin_logit = L[:, 7] * 0.8 + rng.randn(n) * 0.5
    cat["preferred_cabin"] = np.searchsorted([-0.3, 0.7], cabin_logit).astype(np.int64)

    # ── build DataFrames ──────────────────────────────────────────────────────
    num_df = pd.DataFrame(num, index=cids)
    bin_df = pd.DataFrame(bin_, index=cids)
    cat_df = pd.DataFrame(cat, index=cids)

    # ── quantile-transform numerical features only ────────────────────────────
    qt = QuantileTransformer(
        n_quantiles=min(1000, n), output_distribution="normal", random_state=random_seed
    )
    num_qt = qt.fit_transform(num_df.values)
    num_qt_df = pd.DataFrame(num_qt, columns=num_df.columns, index=cids)

    # ── common customer ID mapping ────────────────────────────────────────────
    common_id_map: dict[str, str] = {}
    if source_ids is not None:
        for sid, tid in zip(source_ids[:n_common], common_ids):
            common_id_map[sid] = tid

    schema = FeatureSchema(
        numerical_cols=list(num_df.columns),
        binary_cols=list(bin_df.columns),
        categorical_cols=list(cat_df.columns),
        cat_value_strings=TARGET_CAT_VALUE_STRINGS,
    )
    return num_qt_df, bin_df, cat_df, common_id_map, L, qt, schema
