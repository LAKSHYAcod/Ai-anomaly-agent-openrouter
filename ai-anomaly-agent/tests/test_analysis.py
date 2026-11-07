import json

import numpy as np
import pandas as pd
import pytest

from src.analysis.anomaly_flagger import FlaggingError, flag_anomalies
from src.analysis.anomaly_free import CleaningError, build_anomaly_free_dataset
from src.analysis.reporting import ReportingError, generate_business_report

METRICS = ["Revenue", "Orders", "Conversion Rate", "Traffic", "Cost", "Refunds"]


def make_df(n=40, seed=0, spikes=None):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=n),
        "Revenue": rng.normal(1000, 20, n), "Orders": rng.normal(100, 3, n),
        "Conversion Rate": rng.normal(3.0, 0.1, n), "Traffic": rng.normal(5000, 100, n),
        "Cost": rng.normal(400, 10, n), "Refunds": rng.normal(30, 2, n),
    })
    for (row, col), val in (spikes or {}).items():
        df.loc[row, col] = val
    return df


# ---------------------------------------------------------------- flagger
def test_flags_injected_spike_and_dip():
    df = make_df(spikes={(20, "Revenue"): 2500, (30, "Cost"): 50})
    f = flag_anomalies(df)
    hits = set(zip(f.details["Date"].dt.day_of_year - 1, f.details["Metric"]))
    assert (20, "Revenue") in hits and (30, "Cost") in hits
    assert f.mask.loc[20, "Revenue"] and not f.mask.loc[5, "Revenue"]
    assert set(f.details["Direction"]) == {"High", "Low"}


def test_clean_noise_produces_few_false_positives():
    f = flag_anomalies(make_df(n=200, seed=3))
    assert f.total <= 200 * 6 * 0.01  # <= 1% of all cells on pure Gaussian noise


def test_constant_series_has_no_anomalies_and_no_division_errors():
    df = make_df()
    df["Traffic"] = 5000.0
    f = flag_anomalies(df)
    assert not f.mask["Traffic"].any()


def test_single_spike_in_otherwise_constant_series_is_caught():
    df = make_df()
    df["Traffic"] = 5000.0
    df.loc[25, "Traffic"] = 9000.0
    assert flag_anomalies(df).mask.loc[25, "Traffic"]


def test_severity_labels():
    f = flag_anomalies(make_df(spikes={(20, "Revenue"): 3000}))
    row = f.details[f.details["Metric"] == "Revenue"].iloc[0]
    assert row["Severity"] == "Critical"


def test_flagger_input_validation():
    with pytest.raises(FlaggingError):
        flag_anomalies(pd.DataFrame())
    with pytest.raises(FlaggingError, match="Missing"):
        flag_anomalies(make_df().drop(columns=["Cost"]))
    with pytest.raises(FlaggingError, match="At least"):
        flag_anomalies(make_df(n=2))


def test_flagger_does_not_mutate_input():
    df = make_df(spikes={(20, "Revenue"): 2500})
    snap = df.copy(deep=True)
    flag_anomalies(df)
    pd.testing.assert_frame_equal(df, snap)


def test_nan_values_are_tolerated():
    df = make_df(spikes={(20, "Revenue"): 2500})
    df.loc[5, "Orders"] = np.nan
    f = flag_anomalies(df)
    assert not f.mask.loc[5, "Orders"] and f.mask.loc[20, "Revenue"]


# ---------------------------------------------------------------- anomaly-free
def test_interpolate_replaces_only_flagged_cells_and_leaves_raw_alone():
    raw = make_df(spikes={(20, "Revenue"): 2500})
    snap = raw.copy(deep=True)
    f = flag_anomalies(raw)
    clean, replaced = build_anomaly_free_dataset(raw, f)
    pd.testing.assert_frame_equal(raw, snap)                    # raw untouched
    assert len(clean) == len(raw)
    assert abs(clean.loc[20, "Revenue"] - 1000) < 100          # spike replaced by a plausible estimate
    untouched = ~f.mask.to_numpy()
    np.testing.assert_allclose(clean[METRICS].to_numpy()[untouched], raw[METRICS].to_numpy()[untouched])
    assert replaced.to_numpy().sum() == f.total
    assert flag_anomalies(clean).total == 0


def test_edge_anomaly_takes_nearest_neighbour():
    raw = make_df(spikes={(39, "Revenue"): 100})
    f = flag_anomalies(raw)
    assert f.mask.loc[39, "Revenue"]
    clean, _ = build_anomaly_free_dataset(raw, f)
    assert clean.loc[39, "Revenue"] == pytest.approx(raw.loc[38, "Revenue"])


def test_drop_rows_strategy():
    raw = make_df(spikes={(20, "Revenue"): 2500})
    f = flag_anomalies(raw)
    clean, replaced = build_anomaly_free_dataset(raw, f, "drop_rows")
    assert len(clean) == len(raw) - int(f.row_mask.sum())
    assert not replaced.to_numpy().any()


def test_bad_strategy_and_mismatched_flags_rejected():
    raw = make_df()
    f = flag_anomalies(raw)
    with pytest.raises(CleaningError, match="Unknown"):
        build_anomaly_free_dataset(raw, f, "delete_everything")
    with pytest.raises(CleaningError, match="different number of rows"):
        build_anomaly_free_dataset(raw.head(10), f)


# ---------------------------------------------------------------- reporting
@pytest.fixture
def raw_with_anomaly():
    return make_df(spikes={(20, "Revenue"): 2500, (30, "Cost"): 50})


def test_report_outputs_and_guardrails(tmp_path, raw_with_anomaly):
    snap = raw_with_anomaly.copy(deep=True)
    res = generate_business_report(raw_with_anomaly, tmp_path, use_ai_narrative=False)

    pd.testing.assert_frame_equal(raw_with_anomaly, snap)       # caller's data untouched
    assert res.ok and res.narrative_source == "template"
    assert res.output_dir.parent == tmp_path and res.output_dir.name.startswith("run_")
    expected = {"report", "summary", "flagged", "annotated_raw", "anomaly_free",
                "chart_timeseries", "chart_summary", "chart_interactive"}
    assert expected == set(res.files) and all(p.exists() for p in res.files.values())

    # the clean dataset is a SEPARATE file and differs from the raw values at the flagged cells
    cleaned = pd.read_csv(res.files["anomaly_free"])
    annotated = pd.read_csv(res.files["annotated_raw"])
    assert annotated.loc[20, "Revenue"] == pytest.approx(2500)  # raw value preserved in annotated copy
    assert cleaned.loc[20, "Revenue"] != annotated.loc[20, "Revenue"]
    assert annotated.loc[20, "Revenue__is_anomaly"]

    summary = json.loads(res.files["summary"].read_text())
    assert summary["total_anomalies"] == res.summary["total_anomalies"] >= 2
    assert "Business Anomaly Report" in res.files["report"].read_text()
    # charts are high resolution
    from PIL import Image
    assert Image.open(res.files["chart_timeseries"]).size[0] >= 3000
    assert "plotly" in res.files["chart_interactive"].read_text().lower()


def test_each_run_gets_its_own_folder(tmp_path, raw_with_anomaly):
    a = generate_business_report(raw_with_anomaly, tmp_path, use_ai_narrative=False)
    b = generate_business_report(raw_with_anomaly, tmp_path, use_ai_narrative=False)
    assert a.output_dir != b.output_dir and a.files["report"].exists() and b.files["report"].exists()


def test_source_file_is_never_overwritten(tmp_path, raw_with_anomaly, monkeypatch):
    import src.analysis.reporting as reporting
    # Simulate the worst case: the 'source' is exactly where a report file would go.
    monkeypatch.setattr(reporting, "_make_run_dir", lambda base: tmp_path)
    source = tmp_path / reporting.FILENAMES["anomaly_free"]
    source.write_text("ORIGINAL")
    with pytest.raises(ReportingError, match="original data source"):
        generate_business_report(raw_with_anomaly, tmp_path, source_path=source, use_ai_narrative=False)
    assert source.read_text() == "ORIGINAL"


def test_no_anomalies_still_produces_complete_report(tmp_path):
    flat = make_df(n=30)
    flat[METRICS] = 1.0  # every metric constant -> nothing can be anomalous
    res = generate_business_report(flat, tmp_path, use_ai_narrative=False)
    assert res.ok and res.summary["total_anomalies"] == 0
    assert "No anomalies were detected" in res.files["report"].read_text()
    assert res.flagged_df.empty and len(res.anomaly_free_df) == 30
    assert "chart_timeseries" in res.files  # charts are still produced


def test_drop_rows_report(tmp_path, raw_with_anomaly):
    res = generate_business_report(raw_with_anomaly, tmp_path, strategy="drop_rows", use_ai_narrative=False)
    assert res.ok and res.summary["rows_after_cleaning"] < res.summary["rows_analysed"]


def test_report_input_validation(tmp_path):
    for bad in (None, [1, 2, 3], pd.DataFrame()):
        with pytest.raises(ReportingError):
            generate_business_report(bad, tmp_path)
    with pytest.raises(ReportingError, match="Missing"):
        generate_business_report(make_df().drop(columns=["Revenue"]), tmp_path)
    with pytest.raises(ReportingError, match="Unknown cleaning strategy"):
        generate_business_report(make_df(), tmp_path, strategy="nope")


def test_unwritable_output_dir_raises_cleanly(tmp_path, raw_with_anomaly):
    blocker = tmp_path / "file_not_dir"
    blocker.write_text("x")
    with pytest.raises(ReportingError, match="output folder"):
        generate_business_report(raw_with_anomaly, blocker / "sub", use_ai_narrative=False)


def test_one_failing_artifact_does_not_lose_the_rest(tmp_path, raw_with_anomaly, monkeypatch):
    import src.analysis.reporting as reporting

    def broken(*a, **k):
        raise reporting.VisualisationError("simulated chart failure")

    monkeypatch.setattr(reporting, "plot_interactive_html", broken)
    res = generate_business_report(raw_with_anomaly, tmp_path, use_ai_narrative=False)
    assert not res.ok and any("simulated chart failure" in e for e in res.errors)
    assert "chart_interactive" not in res.files and "report" in res.files and "chart_timeseries" in res.files


def test_unsorted_and_string_dates_and_text_numbers_are_handled(tmp_path, raw_with_anomaly):
    messy = raw_with_anomaly.sample(frac=1, random_state=1).copy()
    messy["Date"] = messy["Date"].dt.strftime("%Y-%m-%d")
    messy["Orders"] = messy["Orders"].astype(str)
    res = generate_business_report(messy, tmp_path, use_ai_narrative=False)
    assert res.ok and res.summary["rows_analysed"] == len(messy)
