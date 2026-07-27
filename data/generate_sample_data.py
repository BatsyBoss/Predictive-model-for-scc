"""
generate_sample_data.py
------------------------
Generates a SYNTHETIC Self-Compacting Concrete (SCC) mix dataset for demo /
development purposes, since the app needs *some* dataset to train on out of
the box.

The relationships baked into the data are loosely based on well-established
concrete-technology principles so the resulting model behaves sensibly:

  * Strength falls as the water/cement ratio rises (Abrams' law style curve).
  * Strength rises with curing age, with diminishing returns (a
    hyperbolic age-strength gain curve similar in shape to ACI 209 /
    CEB-FIP maturity models).
  * Superplasticizer improves dispersion of cement particles, giving a
    small positive strength contribution up to a saturation dosage.
  * Aggregate proportioning has a smaller, secondary effect on strength
    (its bigger real-world impact is on fresh-state SCC properties such as
    slump-flow and passing ability, which are outside this dataset's scope).
  * Realistic random noise (lab variability) is added on top.

IMPORTANT: This is NOT real laboratory data. It exists so the app is usable
immediately after download. Replace it with your own experimental / mix
design database via the Admin > Upload Dataset feature for real use.

Run directly to regenerate data/sample_scc_data.csv:
    python generate_sample_data.py
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 1200


def generate_scc_dataset(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---- Sample mix components within realistic SCC ranges (kg/m3) -------
    cement = rng.uniform(320, 550, n_samples)
    water = rng.uniform(150, 210, n_samples)

    # Total aggregate content typically ~1550-1750 kg/m3 for SCC (lower than
    # normal concrete because of the higher paste volume). Split into fine
    # and coarse using a fine-aggregate ratio typical for SCC (0.45-0.60,
    # SCC uses more fine aggregate than conventional concrete for cohesion).
    total_aggregate = rng.uniform(1550, 1750, n_samples)
    fine_ratio = rng.uniform(0.45, 0.60, n_samples)
    fine_aggregate = total_aggregate * fine_ratio
    coarse_aggregate = total_aggregate * (1 - fine_ratio)

    # Superplasticizer dosage: commonly 0.5-2.5% of cement weight for
    # polycarboxylate-ether based admixtures used in SCC.
    sp_dosage_pct = rng.uniform(0.5, 2.5, n_samples)
    superplasticizer = cement * sp_dosage_pct / 100.0

    # Age in days - sampled from typical testing ages.
    age = rng.choice([1, 3, 7, 14, 28, 56, 90, 180], size=n_samples,
                      p=[0.05, 0.10, 0.20, 0.10, 0.35, 0.10, 0.07, 0.03])

    # ---- Strength model ----------------------------------------------
    wc_ratio = water / cement

    # Abrams'-law-style base strength at 28 days (MPa), decreasing with w/c.
    strength_28 = 105 * np.exp(-2.1 * wc_ratio)
    strength_28 = np.clip(strength_28, 18, 85)

    # Age-gain factor (normalized so factor(28) ~= 1.0), hyperbolic maturity
    # curve: strength(t) ~ t / (a + b*t)
    a, b = 2.8, 0.869  # calibrated so age=28 -> factor ~1.0
    age_factor = age / (a + b * age)
    age_factor = age_factor / (28 / (a + b * 28))

    # Superplasticizer effect: better dispersion -> modest strength boost,
    # saturating around ~2% dosage.
    sp_factor = 1 + 0.06 * np.tanh(sp_dosage_pct / 1.2)

    # Small secondary effect from aggregate proportioning (mild optimum
    # around fine_ratio ~0.5 for packing density / paste efficiency).
    aggregate_factor = 1 - 0.15 * (fine_ratio - 0.50) ** 2

    strength = strength_28 * age_factor * sp_factor * aggregate_factor

    # Lab variability / measurement noise.
    noise = rng.normal(0, 1.8, n_samples)
    strength = np.clip(strength + noise, 8, None)

    df = pd.DataFrame({
        "Cement": cement.round(1),
        "Water": water.round(1),
        "Fine_Aggregate": fine_aggregate.round(1),
        "Coarse_Aggregate": coarse_aggregate.round(1),
        "Superplasticizer": superplasticizer.round(2),
        "Age": age.astype(int),
        "Compressive_Strength": strength.round(2),
    })

    # Inject a handful of missing values so the app's missing-value
    # handling path is exercised by default (bonus requirement).
    mv_idx = rng.choice(df.index, size=max(1, n_samples // 200), replace=False)
    mv_cols = rng.choice(["Superplasticizer", "Fine_Aggregate"], size=len(mv_idx))
    for idx, col in zip(mv_idx, mv_cols):
        df.loc[idx, col] = np.nan

    return df


if __name__ == "__main__":
    dataset = generate_scc_dataset()
    dataset.to_csv("sample_scc_data.csv", index=False)
    print(f"Wrote sample_scc_data.csv with {len(dataset)} rows")
    print(dataset.describe())
