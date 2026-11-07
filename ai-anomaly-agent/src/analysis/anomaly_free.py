"""Build an anomaly-free copy of a dataset.

The input DataFrame is NEVER modified - callers always receive a new
frame. Two strategies are supported:

* ``interpolate`` (default): flagged cells are replaced by an estimate
  (time-aware linear interpolation between the nearest unflagged
  neighbours; edge values take the nearest unflagged neighbour). Every
  date is kept, which preserves the time-series structure.
* ``drop_rows``: any row containing a flagged cell is removed.

Imputed numbers are estimates, not observations. :func:`build_anomaly_free_dataset`
therefore also returns a boolean frame marking exactly which cells were
replaced, so reports can be explicit about it.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.analysis.anomaly_flagger import AnomalyFlags

logger = logging.getLogger(__name__)

STRATEGIES = ("interpolate", "drop_rows")


class CleaningError(ValueError):
    """Raised for invalid strategies or inputs that don't match the flags."""


def _interpolate_column(values: pd.Series, dates: pd.Series) -> pd.Series:
    """Fill NaNs using time-aware interpolation when dates are usable, else positional."""
    try:
        by_time = values.copy()
        by_time.index = pd.DatetimeIndex(dates)
        if by_time.index.hasnans or not by_time.index.is_monotonic_increasing:
            raise ValueError("dates not usable as a monotonic index")
        filled = by_time.interpolate(method="time", limit_direction="both")
        return pd.Series(filled.to_numpy(), index=values.index)
    except (ValueError, TypeError):
        return values.interpolate(method="linear", limit_direction="both")


def build_anomaly_free_dataset(
    df: pd.DataFrame,
    flags: AnomalyFlags,
    strategy: str = "interpolate",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(anomaly_free_df, replaced_mask)``.

    ``replaced_mask`` has the same shape as ``flags.mask`` and is True where
    a value was substituted (all False for ``drop_rows``, where whole rows
    disappear instead).
    """
    if strategy not in STRATEGIES:
        raise CleaningError(f"Unknown cleaning strategy '{strategy}'. Choose from {STRATEGIES}.")
    if len(df) != len(flags.mask):
        raise CleaningError("The flags do not belong to this DataFrame (different number of rows).")

    clean = df.reset_index(drop=True).copy()  # new object; caller's frame untouched

    if strategy == "drop_rows":
        keep = ~flags.row_mask.to_numpy()
        removed = int((~keep).sum())
        clean = clean.loc[keep].reset_index(drop=True)
        logger.info("drop_rows: removed %d row(s) containing anomalies", removed)
        return clean, pd.DataFrame(False, index=clean.index, columns=flags.metrics)

    dates = pd.to_datetime(clean["Date"], errors="coerce")
    for col in flags.metrics:
        if not flags.mask[col].any():
            continue
        values = pd.to_numeric(clean[col], errors="coerce").astype(float)
        values[flags.mask[col].to_numpy()] = float("nan")
        if values.notna().sum() == 0:
            raise CleaningError(f"Every value in '{col}' was flagged; nothing to interpolate from.")
        clean[col] = _interpolate_column(values, dates)

    return clean, flags.mask.copy()
