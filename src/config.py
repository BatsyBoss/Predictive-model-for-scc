"""
config.py
---------
Central place for constants shared across the app: feature names, default
file paths, and realistic SCC parameter bounds used by the reverse-design
optimizer. Keeping these in one module means the rest of the codebase never
hardcodes "magic" column names or ranges.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

DEFAULT_DATA_PATH = DATA_DIR / "sample_scc_data.csv"
MODEL_PATH = MODEL_DIR / "scc_model.joblib"
METADATA_PATH = MODEL_DIR / "scc_model_metadata.joblib"

# ---------------------------------------------------------------------------
# Feature / target schema
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "Cement",
    "Water",
    "Fine_Aggregate",
    "Coarse_Aggregate",
    "Superplasticizer",
    "Age",
]
TARGET_COLUMN = "Compressive_Strength"
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

# Friendly labels + units for display purposes
FEATURE_LABELS = {
    "Cement": "Cement (kg/m³)",
    "Water": "Water (kg/m³)",
    "Fine_Aggregate": "Fine Aggregate (kg/m³)",
    "Coarse_Aggregate": "Coarse Aggregate (kg/m³)",
    "Superplasticizer": "Superplasticizer (kg/m³)",
    "Age": "Age (days)",
}

# ---------------------------------------------------------------------------
# Realistic bounds used to (a) validate/clip user inputs and (b) constrain
# the Monte-Carlo search space during reverse mix-design optimization.
# Values are broadly representative of published SCC mix design ranges
# (EFNARC guidelines / typical literature mixes). Admins retraining on their
# own dataset should sanity-check these still make sense for their data.
# ---------------------------------------------------------------------------
FEATURE_BOUNDS = {
    # Continuous quantities as float bounds and Age as an int bound — kept
    # internally consistent because st.number_input requires min_value,
    # max_value, value, and step to all share the same numeric type.
    "Cement": (300.0, 600.0),
    "Water": (140.0, 220.0),
    "Fine_Aggregate": (650.0, 1000.0),
    "Coarse_Aggregate": (650.0, 1000.0),
    "Superplasticizer": (1.5, 15.0),
    "Age": (1, 180),
}

# Domain constraints applied during Monte-Carlo mix generation
WC_RATIO_BOUNDS = (0.28, 0.55)          # water/cement ratio typical for SCC
SP_DOSAGE_PCT_BOUNDS = (0.4, 2.8)       # superplasticizer as % of cement mass
TOTAL_AGGREGATE_BOUNDS = (1500, 1800)   # fine + coarse, kg/m3
FINE_RATIO_BOUNDS = (0.42, 0.62)        # fine / (fine+coarse)

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
DEFAULT_ADMIN_PASSWORD = "admin123"  # override via st.secrets["ADMIN_PASSWORD"]
APP_TITLE = "AI-Powered SCC Mix Design Studio"
