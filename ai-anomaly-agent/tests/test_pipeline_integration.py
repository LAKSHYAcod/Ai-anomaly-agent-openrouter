import os
import shutil

import pandas as pd
import pytest

from src.graph.pipeline import run_pipeline

SAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample", "sample_metrics.xlsx")


@pytest.fixture
def no_ai(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from src.ai import openai_client
    openai_client.reset_client()


def test_pipeline_generates_report_and_never_touches_source(tmp_path, no_ai):
    src = tmp_path / "input.xlsx"
    shutil.copy(SAMPLE, src)
    before = src.read_bytes()
    report = run_pipeline(str(src), persist=False, generate_report=True, output_dir=str(tmp_path / "out"))
    assert src.read_bytes() == before
    assert report["analysis"]["summary"]["total_anomalies"] >= 1
    assert os.path.exists(report["analysis"]["files"]["anomaly_free"])
    assert report["critical_count"] >= 1          # the existing monitoring path still works


def test_pipeline_default_writes_no_report_files(tmp_path, no_ai, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = run_pipeline(SAMPLE, persist=False)
    assert "analysis" not in report and not (tmp_path / "outputs").exists()


def test_pipeline_survives_analysis_failure(tmp_path, no_ai, monkeypatch):
    import src.graph.pipeline as pl
    from src.analysis.reporting import ReportingError

    def boom(*a, **k):
        raise ReportingError("simulated")

    monkeypatch.setattr(pl, "generate_business_report", boom)
    report = pl.run_pipeline(SAMPLE, persist=False, generate_report=True, output_dir=str(tmp_path))
    assert report["analysis_error"] == "simulated" and report["total_anomalies"] >= 1


def test_bad_file_reports_error(tmp_path, no_ai):
    report = run_pipeline(str(tmp_path / "missing.xlsx"), persist=False, generate_report=True)
    assert report["validation_errors"] and "analysis" not in report
