"""FR-02: Validation.

Checks missing/duplicate dates, missing values, invalid numerics,
invalid percentages, negative values where inappropriate, and
missing required columns. Returns a structured result rather than
raising, so the pipeline can report *all* problems at once.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config_loader import load_config

REQUIRED_DATE_COL = "Date"
NON_NEGATIVE_COLS = ["Revenue", "Orders", "Traffic", "Cost", "Refunds"]
PERCENT_COL = "Conversion Rate"


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_dataframe(df: pd.DataFrame) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    cfg = load_config()
    required_cols = [REQUIRED_DATE_COL] + list(cfg["metrics"].values())

    # 1. Required columns present
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required column(s): {', '.join(missing_cols)}")
        # Can't safely check anything else without the columns
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    # 2. Date column: parseable, no duplicates, no missing
    dates = pd.to_datetime(df[REQUIRED_DATE_COL], errors="coerce")
    if dates.isna().any():
        bad_rows = df.index[dates.isna()].tolist()
        errors.append(f"Unparseable or missing dates at row(s): {bad_rows}")

    dup_dates = dates[dates.duplicated(keep=False) & dates.notna()]
    if not dup_dates.empty:
        errors.append(f"Duplicate dates found: {sorted(set(dup_dates.dt.strftime('%Y-%m-%d')))}")

    # 3. Missing numeric values
    numeric_cols = list(cfg["metrics"].values())
    for col in numeric_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            warnings.append(f"Column '{col}' has {n_missing} missing value(s); will be interpolated.")

    # 4. Invalid numerics (non-numeric strings in a numeric column)
    for col in numeric_cols:
        non_numeric = pd.to_numeric(df[col], errors="coerce").isna() & df[col].notna()
        if non_numeric.any():
            bad_rows = df.index[non_numeric].tolist()
            errors.append(f"Non-numeric values in '{col}' at row(s): {bad_rows}")

    # 5. Negative values where inappropriate
    for col in NON_NEGATIVE_COLS:
        numeric_vals = pd.to_numeric(df[col], errors="coerce")
        negative = numeric_vals < 0
        if negative.any():
            bad_rows = df.index[negative.fillna(False)].tolist()
            errors.append(f"Negative values in '{col}' at row(s): {bad_rows}")

    # 6. Invalid percentages (Conversion Rate expected within 0-100)
    if PERCENT_COL in df.columns:
        pct_vals = pd.to_numeric(df[PERCENT_COL], errors="coerce")
        out_of_range = (pct_vals < 0) | (pct_vals > 100)
        if out_of_range.any():
            bad_rows = df.index[out_of_range.fillna(False)].tolist()
            errors.append(f"'{PERCENT_COL}' out of expected 0-100 range at row(s): {bad_rows}")

    # 7. Minimum history for a meaningful baseline
    min_rows = cfg["baseline"]["min_history_rows"]
    if len(df) < min_rows:
        warnings.append(
            f"Only {len(df)} row(s) provided; at least {min_rows} are recommended for a reliable baseline."
        )

    return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)
