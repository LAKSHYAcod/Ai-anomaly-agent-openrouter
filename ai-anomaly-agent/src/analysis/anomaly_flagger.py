"""Historical anomaly flagging across the WHOLE dataset.

``src/anomaly/detector.py`` answers "is the latest row unusual?" using a
trailing average, which is what a live monitor needs. Auditing a dataset
is a different question: "which rows, anywhere in the history, are
unusual?". A trailing baseline is a poor fit for that, because one early
spike inflates the baseline of the following days and produces false
alarms on the rebound.

This module therefore uses a Hampel filter: each value is compared with
the *centred* rolling median of its neighbours, scaled by the rolling
median absolute deviation (MAD). Median/MAD are robust, so the outliers
being hunted do not distort their own yardstick.

    robust_z = (x - rolling_median) / (1.4826 * rolling_MAD)

Thresholds come from ``config.yaml`` (``anomaly_analysis``).

This function is pure: it never mutates its input.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config_loader import load_config

logger = logging.getLogger(__name__)

MAD_TO_SIGMA = 1.4826  # makes MAD consistent with the std-dev for normal data
MEANAD_TO_SIGMA = 1.253314  # same, for the mean absolute deviation fallback

DETAIL_COLUMNS = [
    "Date", "Metric", "Value", "Expected", "Deviation", "Deviation_Pct",
    "Robust_Z", "Severity", "Direction",
]


class FlaggingError(ValueError):
    """Raised when the input cannot be analysed (missing columns, no numeric data...)."""


@dataclass
class AnomalyFlags:
    """Result of :func:`flag_anomalies`.

    Attributes:
        mask: boolean frame, same index/columns as the analysed metrics;
            True where a value is anomalous.
        details: one row per flagged cell (see ``DETAIL_COLUMNS``),
            sorted by date then metric.
        metrics: the metric columns that were analysed.
    """

    mask: pd.DataFrame
    details: pd.DataFrame
    metrics: list[str]

    @property
    def row_mask(self) -> pd.Series:
        """True for rows with at least one anomalous metric."""
        return self.mask.any(axis=1)

    @property
    def total(self) -> int:
        return int(self.mask.to_numpy().sum())


def _robust_scale(abs_dev: pd.Series, local_mad: pd.Series, floor_ratio: float) -> pd.Series:
    """Per-row scale estimate that is stable on short windows.

    * A 7-point window can have a tiny MAD purely by chance (e.g. several
      identical values), which would turn ordinary noise into a huge
      z-score. The local scale is therefore floored at ``floor_ratio`` x
      the series-wide robust scale.
    * If the series-wide MAD is itself 0 (mostly flat data) we fall back
      to the mean absolute deviation, and for a truly constant series the
      scale is NaN, since a constant series cannot contain anomalies.
    """
    global_mad = float(abs_dev.median()) if abs_dev.notna().any() else 0.0
    if global_mad > 0:
        global_scale = global_mad * MAD_TO_SIGMA
    else:
        mean_ad = float(abs_dev.mean()) if abs_dev.notna().any() else 0.0
        global_scale = mean_ad * MEANAD_TO_SIGMA if mean_ad > 0 else np.nan
    local_scale = local_mad * MAD_TO_SIGMA
    return local_scale.clip(lower=global_scale * floor_ratio).where(lambda s: s > 0, global_scale)


def _flag_one_metric(series: pd.Series, window: int, min_periods: int, floor_ratio: float) -> pd.DataFrame:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    roll = lambda x, f: getattr(x.rolling(window, center=True, min_periods=min_periods), f)()  # noqa: E731

    expected = roll(s, "median")
    abs_dev = (s - expected).abs()
    scale = _robust_scale(abs_dev, roll(abs_dev, "median"), floor_ratio)

    deviation = s - expected
    with np.errstate(divide="ignore", invalid="ignore"):
        robust_z = deviation / scale
        deviation_pct = deviation / expected.abs().replace(0, np.nan) * 100
    return pd.DataFrame(
        {"Value": s, "Expected": expected, "Deviation": deviation,
         "Deviation_Pct": deviation_pct, "Robust_Z": robust_z}
    )


def flag_anomalies(
    df: pd.DataFrame,
    metric_cols: list[str] | None = None,
    *,
    window: int | None = None,
    min_periods: int | None = None,
    z_warning: float | None = None,
    z_critical: float | None = None,
    scale_floor_ratio: float | None = None,
) -> AnomalyFlags:
    """Flag anomalous values in every metric column of ``df``.

    Args:
        df: dataset with a ``Date`` column and the configured metric columns.
        metric_cols: columns to analyse (default: ``metrics`` from config).
        window / min_periods / z_warning / z_critical / scale_floor_ratio:
            override the ``anomaly_analysis`` values from config.

    Raises:
        FlaggingError: ``df`` is empty or lacks the required columns.
    """
    cfg = load_config()
    acfg = cfg["anomaly_analysis"]
    metric_cols = metric_cols or list(cfg["metrics"].values())
    window = window or acfg["window"]
    min_periods = min_periods or acfg["min_periods"]
    z_warning = z_warning if z_warning is not None else acfg["robust_z_warning"]
    z_critical = z_critical if z_critical is not None else acfg["robust_z_critical"]
    floor_ratio = scale_floor_ratio if scale_floor_ratio is not None else acfg["scale_floor_ratio"]
    pct_crit = cfg["detection"]["pct_change_critical"]

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise FlaggingError("Input must be a non-empty pandas DataFrame.")
    missing = [c for c in ["Date", *metric_cols] if c not in df.columns]
    if missing:
        raise FlaggingError(f"Missing required column(s): {', '.join(missing)}")
    if len(df) < min_periods:
        raise FlaggingError(
            f"At least {min_periods} rows are required for anomaly flagging; got {len(df)}."
        )

    work = df.reset_index(drop=True)
    dates = pd.to_datetime(work["Date"], errors="coerce")
    mask = pd.DataFrame(False, index=work.index, columns=metric_cols)
    pieces: list[pd.DataFrame] = []

    for col in metric_cols:
        res = _flag_one_metric(work[col], window, min_periods, floor_ratio)
        flagged = res["Robust_Z"].abs() >= z_warning
        mask[col] = flagged.fillna(False)
        if not flagged.any():
            continue
        hit = res[flagged].copy()
        hit["Date"] = dates[flagged]
        hit["Metric"] = col
        # Detection is gated by the robust z-score; severity is the worse of the
        # z-score and %-deviation signals (same "worst case wins" rule as src/anomaly/severity.py).
        critical = (hit["Robust_Z"].abs() >= z_critical) | (hit["Deviation_Pct"].abs() >= pct_crit)
        hit["Severity"] = np.where(critical, "Critical", "Warning")
        hit["Direction"] = np.where(hit["Deviation"] > 0, "High", "Low")
        pieces.append(hit)

    if pieces:
        details = pd.concat(pieces)[DETAIL_COLUMNS].sort_values(["Date", "Metric"]).reset_index(drop=True)
    else:
        details = pd.DataFrame(columns=DETAIL_COLUMNS)

    logger.info("Flagged %d anomalous value(s) across %d metric(s)", int(mask.to_numpy().sum()), len(metric_cols))
    return AnomalyFlags(mask=mask, details=details, metrics=list(metric_cols))
