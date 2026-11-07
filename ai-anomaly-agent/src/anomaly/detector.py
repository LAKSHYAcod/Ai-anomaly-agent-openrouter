"""FR-05: Detection.

Compares current values with recent baselines and flags unusual
changes using the metric columns produced by metric_engine. This
module ONLY determines *what changed* - severity classification is
handled separately in severity.py, and it never asks an LLM whether
something is anomalous (see PRD Sec. 35, Key Engineering Principle).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.config_loader import load_config


@dataclass
class AnomalyCandidate:
    metric: str
    date: str
    current_value: float
    baseline_value: float | None
    change_percent: float | None
    z_score: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_anomalies(metrics_df: pd.DataFrame) -> list[AnomalyCandidate]:
    """Return one AnomalyCandidate per (metric, row) where either the
    percentage-change or Z-score threshold is exceeded on the *latest*
    row of the DataFrame (the row we're currently monitoring).
    """
    cfg = load_config()
    metric_cols = list(cfg["metrics"].values())
    pct_warn = cfg["detection"]["pct_change_warning"]
    z_warn = cfg["detection"]["zscore_warning"]

    if metrics_df.empty:
        return []

    latest = metrics_df.iloc[-1]
    candidates: list[AnomalyCandidate] = []

    for col in metric_cols:
        pct = latest.get(f"{col}__pct_change")
        z = latest.get(f"{col}__zscore")
        baseline = latest.get(f"{col}__baseline")

        pct_flag = pd.notna(pct) and abs(pct) >= pct_warn
        z_flag = pd.notna(z) and abs(z) >= z_warn

        if pct_flag or z_flag:
            date_val = latest.get("Date")
            date_str = (
                date_val.strftime("%Y-%m-%d") if isinstance(date_val, (pd.Timestamp, datetime)) else str(date_val)
            )
            candidates.append(
                AnomalyCandidate(
                    metric=col,
                    date=date_str,
                    current_value=float(latest[col]),
                    baseline_value=float(baseline) if pd.notna(baseline) else None,
                    change_percent=float(pct) if pd.notna(pct) else None,
                    z_score=float(z) if pd.notna(z) else None,
                )
            )

    return candidates
