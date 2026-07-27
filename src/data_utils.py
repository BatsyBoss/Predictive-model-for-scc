"""
data_utils.py
-------------
Loading, validating, and cleaning the SCC mix design dataset.
"""

from __future__ import annotations

import io
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st

from src.config import DEFAULT_DATA_PATH, REQUIRED_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN


class DataValidationError(Exception):
    """Raised when an uploaded dataset doesn't match the expected schema."""


@st.cache_data(show_spinner=False)
def load_default_dataset() -> pd.DataFrame:
    """Load the bundled sample SCC dataset shipped with the app."""
    return pd.read_csv(DEFAULT_DATA_PATH)


def load_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    """Load a user-uploaded CSV (Admin > Upload Dataset)."""
    raw_bytes = uploaded_file.read()
    df = pd.read_csv(io.BytesIO(raw_bytes))
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Confirm the dataframe has the columns the model expects.
    Raises DataValidationError with a human-readable message if not.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(
            "Dataset is missing required column(s): "
            f"{', '.join(missing)}. Expected columns: {', '.join(REQUIRED_COLUMNS)}"
        )
    if len(df) < 30:
        raise DataValidationError(
            f"Dataset only has {len(df)} rows — at least 30 are recommended for a "
            "meaningful train/test split."
        )


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Handle missing values and obviously invalid rows.

    Returns the cleaned dataframe plus a small report dict describing what
    was done, so the UI can show a transparent success/warning message.
    """
    df = df.copy()
    report = {"rows_in": len(df), "missing_values_filled": {}, "rows_dropped_negative": 0}

    # Keep only the columns we actually use (extra columns are ignored).
    df = df[REQUIRED_COLUMNS]

    # Coerce to numeric; anything unparsable becomes NaN so it's handled
    # by the imputation step below rather than silently corrupting the model.
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Median imputation per column (robust to outliers, simple & transparent).
    for col in REQUIRED_COLUMNS:
        n_missing = int(df[col].isna().sum())
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            report["missing_values_filled"][col] = n_missing

    # Drop physically impossible rows (negative quantities, non-positive age).
    before = len(df)
    mask = (df[FEATURE_COLUMNS] >= 0).all(axis=1) & (df["Age"] > 0)
    df = df[mask]
    report["rows_dropped_negative"] = before - len(df)

    df = df.reset_index(drop=True)
    report["rows_out"] = len(df)
    return df, report


def get_feature_target_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y
