"""FR-03: Cleaning.

Standardizes column names, parses dates, coerces numerics, fills
missing values sensibly, and sorts chronologically. Assumes
validate_dataframe() has already been run (this does not re-validate).
"""
from __future__ import annotations

import pandas as pd

from src.config_loader import load_config


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config()
    metric_cols = list(cfg["metrics"].values())

    clean = df.copy()

    # Standardize column names (strip whitespace)
    clean.columns = [str(c).strip() for c in clean.columns]

    # Parse and sort by date
    clean["Date"] = pd.to_datetime(clean["Date"], errors="coerce")
    clean = clean.dropna(subset=["Date"])
    clean = clean.sort_values("Date").reset_index(drop=True)

    # Coerce numeric columns, interpolate missing values, then forward/back fill any edges
    for col in metric_cols:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
        clean[col] = clean[col].interpolate(method="linear", limit_direction="both")

    # Drop exact duplicate dates, keeping the first occurrence
    clean = clean.drop_duplicates(subset=["Date"], keep="first").reset_index(drop=True)

    return clean
