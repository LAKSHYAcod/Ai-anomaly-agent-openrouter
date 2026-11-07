"""FR-10 support: builds the final structured report the graph returns,
which both the CLI and the Streamlit dashboard consume."""
from __future__ import annotations


def build_report(validation, anomalies: list[dict]) -> dict:
    severities = [a["severity"] for a in anomalies]
    return {
        "validation_errors": validation.errors,
        "validation_warnings": validation.warnings,
        "total_anomalies": len(anomalies),
        "critical_count": severities.count("Critical"),
        "warning_count": severities.count("Warning"),
        "low_count": severities.count("Low"),
        "anomalies": anomalies,
    }
