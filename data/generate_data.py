"""
Synthetic data generation for source (credit card) and target (airline) domains.
Source domain features are quantile-transformed to N(0,1).
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import QuantileTransformer


# ─────────────────────────────── helpers ──────────────────────────────────────

def _make_index(latent_combo: np.ndarray, rng: np.random.RandomState,
                noise_std: float = 0.6) -> np.ndarray:
    """Map latent score to 1–10 index via normal CDF."""
    raw = latent_combo + rng.randn(len(latent_combo)) * noise_std
    return np.clip(stats.norm.cdf(raw) * 9 + 1, 1.0, 10.0)


# ────────────────────────── SOURCE DOMAIN ─────────────────────────────────────

def generate_source_domain(
    n_customers: int = 10_000,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, QuantileTransformer]:
    """
    Credit-card offline payment history for source domain customers.

    Returns
    -------
    features_qt : pd.DataFrame  – quantile-transformed feature matrix
    indices_df  : pd.DataFrame  – ground-truth 1–10 preference indices
    latent      : np.ndarray    – underlying latent profiles (n_customers × 10)
    qt          : QuantileTransformer fitted on raw features
    """
    rng = np.random.RandomState(random_seed)
    n = n_customers
    cids = [f"SRC_{i:05d}" for i in range(n)]

    L = rng.randn(n, 10)  # latent profiles

    feats: dict[str, np.ndarray] = {}

    # ── spending amount per merchant category ──────────────────────────────────
    spend_cats = [
        "grocery", "restaurant", "fashion", "beauty", "travel",
        "entertainment", "fuel", "pharmacy", "electronics", "sports",
        "home_deco", "luxury", "education", "kids", "pet",
    ]
    for i, cat in enumerate(spend_cats):
        base = np.exp(L[:, i % 10] * 0.5 + rng.randn(n) * 0.4 + 4.0)
        feats[f"spend_{cat}_monthly"] = base

    # ── transaction frequency per category ────────────────────────────────────
    for i, cat in enumerate(spend_cats):
        freq = np.maximum(0, L[:, (i + 2) % 10] * 2 + rng.randn(n) + 5)
        feats[f"freq_{cat}_monthly"] = freq

    # ── temporal patterns ─────────────────────────────────────────────────────
    feats["weekend_spend_ratio"] = np.clip(L[:, 0] * 0.1 + 0.42 + rng.randn(n) * 0.08, 0, 1)
    feats["evening_spend_ratio"] = np.clip(L[:, 1] * 0.08 + 0.35 + rng.randn(n) * 0.07, 0, 1)
    feats["midnight_spend_ratio"] = np.clip(rng.beta(1, 10, n), 0, 0.15)

    # ── merchant diversity & geography ────────────────────────────────────────
    feats["merchant_diversity_cnt"] = np.maximum(1, L[:, 3] * 5 + rng.randn(n) * 3 + 22)
    feats["local_merchant_ratio"] = np.clip(L[:, 4] * 0.1 + 0.58 + rng.randn(n) * 0.08, 0, 1)
    feats["chain_store_ratio"] = np.clip(1 - feats["local_merchant_ratio"] + rng.randn(n) * 0.04, 0, 1)
    feats["visited_city_cnt"] = np.maximum(1, L[:, 4] * 2 + rng.randn(n) * 1.5 + 4)

    # ── payment behaviour ─────────────────────────────────────────────────────
    total_spend = sum(feats[f"spend_{c}_monthly"] for c in spend_cats)
    total_freq = sum(feats[f"freq_{c}_monthly"] for c in spend_cats)
    feats["total_monthly_spend"] = total_spend
    feats["avg_transaction_amount"] = total_spend / np.maximum(1, total_freq)
    feats["n_transactions_monthly"] = total_freq
    feats["premium_merchant_ratio"] = np.clip(L[:, 5] * 0.1 + 0.18 + rng.randn(n) * 0.05, 0, 1)
    feats["installment_ratio"] = np.clip(L[:, 6] * 0.08 + 0.28 + rng.randn(n) * 0.05, 0, 1)
    feats["cashback_usage_rate"] = np.clip(L[:, 7] * 0.08 + 0.38 + rng.randn(n) * 0.05, 0, 1)
    feats["refund_rate"] = np.clip(rng.beta(1, 20, n), 0, 0.15)

    # ── seasonal spend ratios ─────────────────────────────────────────────────
    season_bases = rng.dirichlet(np.ones(4) * 3, n)
    for k, season in enumerate(["spring", "summer", "fall", "winter"]):
        feats[f"seasonal_ratio_{season}"] = season_bases[:, k]

    # ── RFM-style ─────────────────────────────────────────────────────────────
    feats["recency_days"] = np.maximum(0, -L[:, 9] * 5 + rng.randn(n) * 2 + 7)
    feats["customer_tenure_months"] = np.maximum(1, rng.gamma(3, 12, n))

    feats_df = pd.DataFrame(feats, index=cids)

    # ── ground-truth indices from latent profiles ─────────────────────────────
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

    # ── quantile-transform features ───────────────────────────────────────────
    qt = QuantileTransformer(
        n_quantiles=min(1000, n),
        output_distribution="normal",
        random_state=random_seed,
    )
    feat_qt = qt.fit_transform(feats_df.values)
    feats_qt_df = pd.DataFrame(feat_qt, columns=feats_df.columns, index=cids)

    return feats_qt_df, indices_df, L, qt


# ────────────────────────── TARGET DOMAIN ─────────────────────────────────────

def generate_target_domain(
    n_customers: int = 5_000,
    n_common: int = 1_000,
    source_latent: np.ndarray | None = None,
    source_ids: list[str] | None = None,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, str], np.ndarray, QuantileTransformer]:
    """
    Airline mileage / boarding-pass payment history for target domain customers.

    The first ``n_common`` customers are shared with source domain
    (their latent profiles are taken from source with small noise).

    Returns
    -------
    features_qt          : pd.DataFrame
    common_id_map        : dict  src_id → tgt_id
    latent               : np.ndarray
    qt                   : QuantileTransformer
    """
    rng = np.random.RandomState(random_seed + 1)
    n = n_customers

    common_ids = [f"CMN_{i:05d}" for i in range(n_common)]
    unique_ids = [f"TGT_{i:05d}" for i in range(n - n_common)]
    cids = common_ids + unique_ids

    # latent
    if source_latent is not None and n_common > 0:
        common_L = source_latent[:n_common] + rng.randn(n_common, source_latent.shape[1]) * 0.25
        unique_L = rng.randn(n - n_common, source_latent.shape[1])
        L = np.vstack([common_L, unique_L])
    else:
        L = rng.randn(n, 10)

    feats: dict[str, np.ndarray] = {}

    # ── mileage ───────────────────────────────────────────────────────────────
    feats["miles_earned_total"] = np.maximum(0, np.exp(L[:, 4] * 0.8 + rng.randn(n) * 0.5 + 8))
    feats["miles_redeemed_total"] = feats["miles_earned_total"] * np.clip(rng.beta(2, 3, n), 0.1, 0.9)
    feats["miles_balance"] = feats["miles_earned_total"] - feats["miles_redeemed_total"]
    feats["miles_expiry_risk_ratio"] = np.clip(rng.beta(1, 6, n), 0, 1)
    feats["miles_earn_rate_per_flight"] = np.maximum(100, L[:, 7] * 300 + rng.randn(n) * 100 + 1200)

    # ── flight frequency ──────────────────────────────────────────────────────
    feats["domestic_flights_yearly"] = np.maximum(0, L[:, 4] * 3 + rng.randn(n) * 2 + 6)
    feats["intl_flights_yearly"] = np.maximum(0, L[:, 4] * 1.5 + rng.randn(n) * 1 + 2)
    feats["total_flights_yearly"] = feats["domestic_flights_yearly"] + feats["intl_flights_yearly"]
    feats["flight_yoy_growth"] = rng.randn(n) * 0.2

    # ── booking class ─────────────────────────────────────────────────────────
    # Use per-customer Dirichlet concentration parameters
    alphas = np.stack([
        np.clip(5 - L[:, 7], 0.5, 10),
        np.clip(1 + L[:, 7], 0.1, 5),
        np.clip(0.3 + L[:, 7] * 0.2 + 0.2, 0.05, 2),
    ], axis=1)  # (n, 3)
    class_mix = np.array([rng.dirichlet(alphas[i]) for i in range(n)])  # (n, 3)
    feats["economy_ratio"] = class_mix[:, 0]
    feats["business_ratio"] = class_mix[:, 1]
    feats["first_class_ratio"] = class_mix[:, 2]

    # ── booking behaviour ─────────────────────────────────────────────────────
    feats["avg_advance_booking_days"] = np.maximum(1, L[:, 9] * 10 + rng.randn(n) * 5 + 28)
    feats["last_minute_ratio"] = np.clip(-L[:, 9] * 0.05 + 0.18 + rng.randn(n) * 0.05, 0, 0.5)
    feats["direct_booking_ratio"] = np.clip(L[:, 0] * 0.05 + 0.52 + rng.randn(n) * 0.08, 0, 1)
    feats["mobile_booking_ratio"] = np.clip(rng.beta(4, 2, n), 0.2, 1)

    # ── in-flight spending ────────────────────────────────────────────────────
    feats["inflight_meal_spend_per_flight"] = np.maximum(0, L[:, 1] * 5 + rng.randn(n) * 3 + 14)
    feats["inflight_duty_free_spend_yearly"] = np.maximum(0, L[:, 7] * 20 + rng.randn(n) * 10 + 30)
    feats["inflight_entertainment_spend_per_flight"] = np.maximum(0, L[:, 5] * 3 + rng.randn(n) * 2 + 5)
    feats["inflight_wifi_usage_ratio"] = np.clip(L[:, 9] * 0.1 + 0.4 + rng.randn(n) * 0.1, 0, 1)

    # ── lounge ────────────────────────────────────────────────────────────────
    feats["lounge_visits_yearly"] = np.maximum(0, L[:, 7] * 4 + rng.randn(n) * 2 + 3)
    feats["lounge_spend_yearly"] = feats["lounge_visits_yearly"] * np.maximum(0, L[:, 7] * 20 + rng.randn(n) * 10 + 50)

    # ── destination / trip type ───────────────────────────────────────────────
    feats["destination_diversity"] = np.maximum(1, L[:, 3] * 3 + rng.randn(n) * 2 + 7)
    feats["leisure_flight_ratio"] = np.clip(L[:, 4] * 0.1 + 0.62 + rng.randn(n) * 0.1, 0.2, 1)
    feats["business_trip_ratio"] = np.clip(1 - feats["leisure_flight_ratio"] + rng.randn(n) * 0.02, 0, 1)
    feats["avg_trip_duration_days"] = np.maximum(1, L[:, 4] * 1 + rng.randn(n) * 1.5 + 4)

    # ── ticket spend ──────────────────────────────────────────────────────────
    feats["avg_ticket_price"] = np.maximum(80, np.exp(L[:, 7] * 0.3 + rng.randn(n) * 0.3 + 5.5))
    feats["total_ticket_spend_yearly"] = feats["avg_ticket_price"] * feats["total_flights_yearly"]

    # ── loyalty tier (0–3 encoded) ────────────────────────────────────────────
    tier_logit = L[:, 4] * 0.5 + L[:, 7] * 0.3
    feats["loyalty_tier"] = np.searchsorted([-0.43, 0.43, 1.07], tier_logit).astype(float)

    # ── seasonal ──────────────────────────────────────────────────────────────
    season_bases = rng.dirichlet(np.ones(4) * 3, n)
    for k, season in enumerate(["spring", "summer", "fall", "winter"]):
        feats[f"flights_{season}_ratio"] = season_bases[:, k]

    feats_df = pd.DataFrame(feats, index=cids)

    # ── quantile-transform ────────────────────────────────────────────────────
    qt = QuantileTransformer(
        n_quantiles=min(1000, n),
        output_distribution="normal",
        random_state=random_seed,
    )
    feat_qt = qt.fit_transform(feats_df.values)
    feats_qt_df = pd.DataFrame(feat_qt, columns=feats_df.columns, index=cids)

    # ── common customer ID mapping ────────────────────────────────────────────
    common_id_map: dict[str, str] = {}
    if source_ids is not None:
        for src_id, tgt_id in zip(source_ids[:n_common], common_ids):
            common_id_map[src_id] = tgt_id

    return feats_qt_df, common_id_map, L, qt
