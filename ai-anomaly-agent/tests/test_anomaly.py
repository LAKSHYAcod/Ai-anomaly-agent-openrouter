import pandas as pd

from src.anomaly.detector import AnomalyCandidate, detect_anomalies
from src.anomaly.severity import classify_severity
from src.metrics.metric_engine import calculate_metrics


def _sample_df():
    return pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=10),
            "Revenue": [100, 102, 98, 101, 99, 103, 100, 101, 100, 68],
            "Orders": [10] * 9 + [7],
            "Conversion Rate": [3.0] * 10,
            "Traffic": [1000] * 10,
            "Cost": [40] * 10,
            "Refunds": [2] * 10,
        }
    )


def test_detect_flags_revenue_drop():
    metrics_df = calculate_metrics(_sample_df())
    anomalies = detect_anomalies(metrics_df)
    metrics_flagged = {a.metric for a in anomalies}
    assert "Revenue" in metrics_flagged


def test_no_anomalies_on_stable_data():
    stable = pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=10),
            "Revenue": [100] * 10,
            "Orders": [10] * 10,
            "Conversion Rate": [3.0] * 10,
            "Traffic": [1000] * 10,
            "Cost": [40] * 10,
            "Refunds": [2] * 10,
        }
    )
    metrics_df = calculate_metrics(stable)
    anomalies = detect_anomalies(metrics_df)
    assert anomalies == []


def test_severity_critical_for_large_drop():
    candidate = AnomalyCandidate(
        metric="Revenue", date="2026-01-10", current_value=68, baseline_value=100,
        change_percent=-32.0, z_score=-3.5,
    )
    assert classify_severity(candidate) == "Critical"


def test_severity_low_for_small_move():
    candidate = AnomalyCandidate(
        metric="Revenue", date="2026-01-10", current_value=108, baseline_value=100,
        change_percent=8.0, z_score=0.5,
    )
    assert classify_severity(candidate) == "Low"


def test_severity_normal_for_no_move():
    candidate = AnomalyCandidate(
        metric="Revenue", date="2026-01-10", current_value=100, baseline_value=100,
        change_percent=0.0, z_score=0.0,
    )
    assert classify_severity(candidate) == "Normal"
