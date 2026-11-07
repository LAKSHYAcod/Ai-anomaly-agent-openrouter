import pandas as pd

from src.metrics.metric_engine import calculate_metrics


def _sample_df():
    return pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=10),
            "Revenue": [100, 102, 98, 101, 99, 103, 100, 101, 100, 68],
            "Orders": [10] * 10,
            "Conversion Rate": [3.0] * 10,
            "Traffic": [1000] * 10,
            "Cost": [40] * 10,
            "Refunds": [2] * 10,
        }
    )


def test_pct_change_computed():
    df = calculate_metrics(_sample_df())
    # last row: (68 - 100) / 100 * 100 = -32%
    assert round(df["Revenue__pct_change"].iloc[-1], 1) == -32.0


def test_baseline_is_prior_rolling_mean():
    df = calculate_metrics(_sample_df())
    # baseline excludes "today" - first row should have no baseline
    assert pd.isna(df["Revenue__baseline"].iloc[0])
    assert not pd.isna(df["Revenue__baseline"].iloc[-1])


def test_zscore_flags_final_drop():
    df = calculate_metrics(_sample_df())
    z_last = df["Revenue__zscore"].iloc[-1]
    assert z_last < -2  # sharp drop should register as a strong negative z-score


def test_division_by_zero_guarded():
    df = _sample_df()
    df.loc[0, "Revenue"] = 0
    out = calculate_metrics(df)
    # Should not raise, and pct_change on the following row should be NaN or finite, never inf
    import numpy as np

    assert not np.isinf(out["Revenue__pct_change"].iloc[1])
