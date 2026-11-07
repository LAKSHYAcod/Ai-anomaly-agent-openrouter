"""FR-06 / Sec. 10: Severity Model.

Normal = no significant anomaly
Low = minor unusual movement
Warning = significant deviation requiring review
Critical = extreme deviation requiring prompt attention

Worst-case wins when percentage-change and Z-score disagree.
"""
from __future__ import annotations

from src.anomaly.detector import AnomalyCandidate
from src.config_loader import load_config

_RANK = {"Normal": 0, "Low": 1, "Warning": 2, "Critical": 3}


def _severity_from_pct(pct: float | None, cfg: dict) -> str:
    if pct is None:
        return "Normal"
    abs_pct = abs(pct)
    if abs_pct >= cfg["detection"]["pct_change_critical"]:
        return "Critical"
    if abs_pct >= cfg["detection"]["pct_change_warning"]:
        return "Warning"
    if abs_pct >= cfg["detection"]["pct_change_warning"] / 2:
        return "Low"
    return "Normal"


def _severity_from_zscore(z: float | None, cfg: dict) -> str:
    if z is None:
        return "Normal"
    abs_z = abs(z)
    if abs_z >= cfg["detection"]["zscore_critical"]:
        return "Critical"
    if abs_z >= cfg["detection"]["zscore_warning"]:
        return "Warning"
    return "Normal"


def classify_severity(candidate: AnomalyCandidate) -> str:
    cfg = load_config()
    pct_severity = _severity_from_pct(candidate.change_percent, cfg)
    z_severity = _severity_from_zscore(candidate.z_score, cfg)
    return max(pct_severity, z_severity, key=lambda s: _RANK[s])


def annotate_severity(candidates: list[AnomalyCandidate]) -> list[dict]:
    """Attach a severity label to each candidate and return plain dicts,
    ready to be handed to the AI explanation layer / database.
    """
    annotated = []
    for c in candidates:
        d = c.to_dict()
        d["severity"] = classify_severity(c)
        annotated.append(d)
    return annotated
