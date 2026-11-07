import pandas as pd

from src.validation.validator import validate_dataframe


def _good_df():
    return pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=5),
            "Revenue": [100, 101, 99, 102, 100],
            "Orders": [10, 10, 9, 11, 10],
            "Conversion Rate": [3.0, 3.1, 2.9, 3.2, 3.0],
            "Traffic": [1000, 1010, 990, 1020, 1000],
            "Cost": [40, 41, 39, 42, 40],
            "Refunds": [2, 2, 1, 3, 2],
        }
    )


def test_valid_dataframe_passes():
    result = validate_dataframe(_good_df())
    assert result.is_valid
    assert result.errors == []


def test_missing_column_fails():
    df = _good_df().drop(columns=["Traffic"])
    result = validate_dataframe(df)
    assert not result.is_valid
    assert any("Traffic" in e for e in result.errors)


def test_duplicate_dates_fail():
    df = _good_df()
    df.loc[1, "Date"] = df.loc[0, "Date"]
    result = validate_dataframe(df)
    assert not result.is_valid
    assert any("Duplicate dates" in e for e in result.errors)


def test_negative_revenue_fails():
    df = _good_df()
    df.loc[0, "Revenue"] = -50
    result = validate_dataframe(df)
    assert not result.is_valid
    assert any("Negative values" in e for e in result.errors)


def test_invalid_conversion_rate_range_fails():
    df = _good_df()
    df.loc[0, "Conversion Rate"] = 150
    result = validate_dataframe(df)
    assert not result.is_valid
    assert any("Conversion Rate" in e for e in result.errors)


def test_missing_values_produce_warning_not_error():
    df = _good_df()
    df.loc[2, "Revenue"] = None
    result = validate_dataframe(df)
    assert result.is_valid
    assert any("missing value" in w.lower() for w in result.warnings)
