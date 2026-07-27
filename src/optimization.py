"""
optimization.py
-----------------
Reverse mix-design optimization: given a *target* compressive strength, find
candidate SCC mix proportions the trained model predicts will achieve it.

Approach: constrained Monte-Carlo search.
  1. Sample a large number of candidate mixes within realistic SCC domain
     constraints (water/cement ratio, total aggregate content, fine/coarse
     split, superplasticizer dosage as % of cement) rather than sampling
     each raw feature independently — this keeps generated mixes physically
     sensible instead of e.g. pairing very high cement with very high water.
  2. Predict strength for every candidate in one vectorized batch call.
  3. Keep the candidates closest to the target strength.
  4. Select `n_designs` diverse-but-accurate mixes by bucketing the closest
     matches by cement content (a good proxy for both cost and mix
     "character") and taking the best match in each bucket. This naturally
     yields an economical / balanced / high-performance style spread rather
     than three near-identical mixes.

This is a black-box optimizer (it never needs the model's internals), so it
works identically whether the underlying model is XGBoost or the sklearn
fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_BOUNDS,
    FEATURE_COLUMNS,
    WC_RATIO_BOUNDS,
    SP_DOSAGE_PCT_BOUNDS,
    TOTAL_AGGREGATE_BOUNDS,
    FINE_RATIO_BOUNDS,
)
from src.model_utils import predict_batch

DESIGN_LABELS_3 = ["Economical", "Balanced", "High-Performance"]


def _sample_candidates(n_candidates: int, age: int, rng: np.random.Generator) -> pd.DataFrame:
    """Vectorized sampling of physically-plausible SCC mix candidates."""
    cement = rng.uniform(*FEATURE_BOUNDS["Cement"], n_candidates)

    wc_ratio = rng.uniform(*WC_RATIO_BOUNDS, n_candidates)
    water = np.clip(cement * wc_ratio, *FEATURE_BOUNDS["Water"])

    total_aggregate = rng.uniform(*TOTAL_AGGREGATE_BOUNDS, n_candidates)
    fine_ratio = rng.uniform(*FINE_RATIO_BOUNDS, n_candidates)
    fine_aggregate = np.clip(total_aggregate * fine_ratio, *FEATURE_BOUNDS["Fine_Aggregate"])
    coarse_aggregate = np.clip(total_aggregate * (1 - fine_ratio), *FEATURE_BOUNDS["Coarse_Aggregate"])

    sp_pct = rng.uniform(*SP_DOSAGE_PCT_BOUNDS, n_candidates)
    superplasticizer = np.clip(cement * sp_pct / 100.0, *FEATURE_BOUNDS["Superplasticizer"])

    age_arr = np.full(n_candidates, age, dtype=float)

    return pd.DataFrame({
        "Cement": cement,
        "Water": water,
        "Fine_Aggregate": fine_aggregate,
        "Coarse_Aggregate": coarse_aggregate,
        "Superplasticizer": superplasticizer,
        "Age": age_arr,
    })


def _estimate_relative_cost(df: pd.DataFrame) -> np.ndarray:
    """
    Illustrative relative-cost index (0-100, higher = more expensive), driven
    mainly by cement and superplasticizer content — the dominant cost
    drivers in an SCC mix. This is NOT a real cost estimate (no live
    material pricing is used) — it's only useful for comparing the 3
    generated options against each other.
    """
    raw = df["Cement"] * 1.0 + df["Superplasticizer"] * 25.0
    lo, hi = raw.min(), raw.max()
    if hi - lo < 1e-9:
        return np.full(len(df), 50.0)
    return ((raw - lo) / (hi - lo) * 100).round(1)


def generate_mix_designs(
    model,
    target_strength: float,
    age: int = 28,
    n_designs: int = 3,
    n_candidates: int = 40000,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate `n_designs` candidate SCC mixes predicted to achieve
    `target_strength` MPa at the given `age`.

    Returns a DataFrame with the mix design columns plus:
      - Predicted_Strength
      - Error (MPa, predicted - target)
      - Percent_Error
      - Achievable (bool — False if even the closest match misses the
        target by more than 15%, signalling the target may be outside what
        the constrained search space / trained model can reach)
      - Design_Type (Economical / Balanced / High-Performance for n=3)
      - Relative_Cost_Index (0-100, illustrative only)
    """
    rng = np.random.default_rng(seed)
    candidates = _sample_candidates(n_candidates, age=age, rng=rng)

    predicted = predict_batch(model, candidates)
    candidates["Predicted_Strength"] = predicted
    candidates["Error"] = candidates["Predicted_Strength"] - target_strength
    candidates["Abs_Error"] = candidates["Error"].abs()

    # Keep a pool of the closest matches to select diverse designs from.
    pool_size = min(600, len(candidates))
    pool = candidates.nsmallest(pool_size, "Abs_Error").copy()

    # Bucket the pool by cement content (proxy for mix "character": lean /
    # balanced / rich) and take the closest-to-target match within each
    # bucket, so the final designs are both accurate AND diverse.
    # NOTE: we split the row-index array (a plain ndarray), not the
    # DataFrame itself — np.array_split on a DataFrame silently degrades it
    # to a bare ndarray on some pandas versions, losing column labels.
    pool_sorted = pool.sort_values("Cement").reset_index(drop=True)
    index_chunks = np.array_split(np.arange(len(pool_sorted)), n_designs)

    selected_rows = []
    for idx_chunk in index_chunks:
        if len(idx_chunk) == 0:
            continue
        bucket = pool_sorted.iloc[idx_chunk]
        best = bucket.loc[bucket["Abs_Error"].idxmin()]
        selected_rows.append(best)

    # In the unlikely event a bucket was empty, backfill from the overall pool.
    while len(selected_rows) < n_designs and len(pool_sorted) > len(selected_rows):
        remaining = pool_sorted.drop(index=[r.name for r in selected_rows], errors="ignore")
        if remaining.empty:
            break
        selected_rows.append(remaining.loc[remaining["Abs_Error"].idxmin()])

    result = pd.DataFrame(selected_rows).reset_index(drop=True)
    result = result.sort_values("Cement").reset_index(drop=True)

    result["Percent_Error"] = (result["Error"] / target_strength * 100).round(2)
    result["Achievable"] = result["Abs_Error"] <= (0.15 * target_strength)
    result["Relative_Cost_Index"] = _estimate_relative_cost(result)

    if n_designs == 3:
        result["Design_Type"] = DESIGN_LABELS_3
    else:
        result["Design_Type"] = [f"Option {i + 1}" for i in range(len(result))]

    # Round the mix-design columns for clean display (Age as a clean int).
    for col in FEATURE_COLUMNS:
        if col == "Age":
            result[col] = result[col].round(0).astype(int)
        else:
            result[col] = result[col].round(1)
    result["Predicted_Strength"] = result["Predicted_Strength"].round(2)

    ordered_cols = ["Design_Type"] + FEATURE_COLUMNS + [
        "Predicted_Strength", "Percent_Error", "Achievable", "Relative_Cost_Index"
    ]
    return result[ordered_cols]
