"""Metric Engine (TRD 27): percentage change, moving-average baseline, Z-score.

Guards against division by zero and insufficient history, as required
by the TRD. Operates on the cleaned DataFrame and returns a new
DataFrame with one set of {metric}_pct_change / {metric}_baseline /
{metric}_zscore columns per monitored metric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config_loader import load_config


def _pct_change(current: pd.Series) -> pd.Series:
    previous = current.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (current - previous) / previous.replace(0, np.nan) * 100
    return pct


def _moving_average(current: pd.Series, window: int) -> pd.Series:
    # shift(1) so "today" is compared against a baseline that excludes today
    return current.shift(1).rolling(window=window, min_periods=1).mean()


def _rolling_zscore(current: pd.Series, window: int) -> pd.Series:
    history = current.shift(1)
    mean = history.rolling(window=window, min_periods=2).mean()
    std = history.rolling(window=window, min_periods=2).std(ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (current - mean) / std.replace(0, np.nan)
    return z


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config()
    window = cfg["baseline"]["moving_average_window"]
    metric_cols = list(cfg["metrics"].values())

    out = df.copy()
    for col in metric_cols:
        out[f"{col}__baseline"] = _moving_average(out[col], window)
        out[f"{col}__pct_change"] = _pct_change(out[col])
        out[f"{col}__zscore"] = _rolling_zscore(out[col], window)

    return out
