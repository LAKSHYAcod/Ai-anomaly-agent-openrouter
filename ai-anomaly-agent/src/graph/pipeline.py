"""TRD Sec. 21 & 29: the High-Level Architecture and Module/API Flow,
implemented as a LangGraph StateGraph.

    Excel File -> Ingestion -> Validation -> Cleaning -> Historical Analysis
    (optional: anomaly-free dataset, business report, charts) -> Metric Engine
    -> Anomaly Engine -> Severity -> AI Explanation -> Save -> Email Alert
    -> Report

Conditional edges short-circuit the graph when validation fails or when
no anomalies are found, so we never call the LLM or send email unless
there's actually something worth explaining.
"""
from __future__ import annotations

import json
import logging
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from src.ai.explainer import explain_anomaly
from src.analysis.reporting import ReportingError, generate_business_report
from src.alerts.email_alert import send_alert_email
from src.anomaly.detector import detect_anomalies
from src.anomaly.severity import annotate_severity
from src.cleaning.cleaner import clean_dataframe
from src.config_loader import load_config
from src.database.db import save_alert, save_anomaly, save_metric_records
from src.ingestion.excel_reader import ExcelReadError, read_excel_file
from src.metrics.metric_engine import calculate_metrics
from src.reports.report_generator import build_report
from src.validation.validator import ValidationResult, validate_dataframe

logger = logging.getLogger(__name__)


class PipelineState(TypedDict, total=False):
    file_path: str
    persist: bool  # whether to write to the DB / send email (off for dry runs & tests)
    generate_report: bool  # whether to produce the anomaly-free dataset, report and charts
    output_dir: str | None  # parent folder for report outputs (default from config)
    raw_df: pd.DataFrame
    clean_df: pd.DataFrame
    metrics_df: pd.DataFrame
    validation: ValidationResult
    anomalies: list[dict[str, Any]]
    fatal_error: str | None
    analysis: dict[str, Any]
    analysis_error: str | None
    report: dict[str, Any]


# --- Nodes ---------------------------------------------------------------


def ingest_node(state: PipelineState) -> PipelineState:
    try:
        raw_df = read_excel_file(state["file_path"])
        return {"raw_df": raw_df}
    except ExcelReadError as exc:
        return {"fatal_error": str(exc)}


def validate_node(state: PipelineState) -> PipelineState:
    result = validate_dataframe(state["raw_df"])
    return {"validation": result}


def clean_node(state: PipelineState) -> PipelineState:
    clean_df = clean_dataframe(state["raw_df"])
    return {"clean_df": clean_df}


def analyze_node(state: PipelineState) -> PipelineState:
    """Whole-history anomaly analysis -> anomaly-free dataset, report and charts.

    Failures here are recorded but never stop the monitoring path that follows.
    The raw data is not touched: the report is written to a fresh run folder.
    """
    if not state.get("generate_report", False):
        return {}
    try:
        artifacts = generate_business_report(
            state["clean_df"],
            output_dir=state.get("output_dir"),
            source_path=state.get("file_path"),
        )
    except ReportingError as exc:
        logger.error("Historical analysis failed: %s", exc)
        return {"analysis_error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - keep the monitoring pipeline alive
        logger.exception("Unexpected failure in historical analysis")
        return {"analysis_error": f"Unexpected error: {exc}"}
    return {
        "analysis": {
            "output_dir": str(artifacts.output_dir),
            "files": {k: str(v) for k, v in artifacts.files.items()},
            "errors": artifacts.errors,
            "narrative_source": artifacts.narrative_source,
            "summary": artifacts.summary,
        }
    }


def metrics_node(state: PipelineState) -> PipelineState:
    metrics_df = calculate_metrics(state["clean_df"])
    return {"metrics_df": metrics_df}


def detect_and_score_node(state: PipelineState) -> PipelineState:
    candidates = detect_anomalies(state["metrics_df"])
    annotated = annotate_severity(candidates)
    return {"anomalies": annotated}


def explain_node(state: PipelineState) -> PipelineState:
    latest_row = state["metrics_df"].iloc[-1]
    cfg = load_config()
    metric_cols = list(cfg["metrics"].values())
    related_metrics = [
        {"metric": m, "current_value": float(latest_row[m])} for m in metric_cols
    ]

    explained = []
    for anomaly in state["anomalies"]:
        explanation = explain_anomaly(anomaly, related_metrics)
        enriched = dict(anomaly)
        enriched["explanation"] = explanation.model_dump()
        explained.append(enriched)
    return {"anomalies": explained}


def persist_node(state: PipelineState) -> PipelineState:
    if not state.get("persist", True):
        return {}
    save_metric_records(state["clean_df"])
    updated = []
    for anomaly in state["anomalies"]:
        explanation_json = json.dumps(anomaly.get("explanation", {}))
        anomaly_id = save_anomaly(anomaly, explanation_json)
        enriched = dict(anomaly)
        enriched["id"] = anomaly_id
        updated.append(enriched)
    return {"anomalies": updated}


def alert_node(state: PipelineState) -> PipelineState:
    if not state.get("persist", True):
        return {}
    import os

    cfg = load_config()["alerts"]
    notify_severities = set(cfg["notify_on"])
    recipient = os.environ.get("ALERT_RECIPIENT")

    for anomaly in state["anomalies"]:
        if anomaly["severity"] not in notify_severities:
            continue
        subject, message, status = send_alert_email(anomaly, anomaly.get("explanation", {}))
        save_alert(
            anomaly_id=anomaly.get("id"),
            recipient=recipient,
            subject=subject,
            message=message,
            status=status,
        )
    return {}


def report_node(state: PipelineState) -> PipelineState:
    if state.get("fatal_error"):
        report = {"validation_errors": [state["fatal_error"]], "validation_warnings": [], "anomalies": []}
        return {"report": report}
    if not state["validation"].is_valid:
        report = build_report(state["validation"], [])
        return {"report": report}
    report = build_report(state["validation"], state.get("anomalies", []))
    if state.get("analysis"):
        report["analysis"] = state["analysis"]
    if state.get("analysis_error"):
        report["analysis_error"] = state["analysis_error"]
    return {"report": report}


# --- Conditional routing ---------------------------------------------------


def route_after_ingest(state: PipelineState) -> str:
    return "report" if state.get("fatal_error") else "validate"


def route_after_validate(state: PipelineState) -> str:
    return "clean" if state["validation"].is_valid else "report"


def route_after_detect(state: PipelineState) -> str:
    return "explain" if state.get("anomalies") else "report"


def build_pipeline():
    graph = StateGraph(PipelineState)

    graph.add_node("ingest", ingest_node)
    graph.add_node("validate", validate_node)
    graph.add_node("clean", clean_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("metrics", metrics_node)
    graph.add_node("detect_and_score", detect_and_score_node)
    graph.add_node("explain", explain_node)
    graph.add_node("persist", persist_node)
    graph.add_node("alert", alert_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "ingest")
    graph.add_conditional_edges("ingest", route_after_ingest, {"validate": "validate", "report": "report"})
    graph.add_conditional_edges("validate", route_after_validate, {"clean": "clean", "report": "report"})
    graph.add_edge("clean", "analyze")
    graph.add_edge("analyze", "metrics")
    graph.add_edge("metrics", "detect_and_score")
    graph.add_conditional_edges(
        "detect_and_score", route_after_detect, {"explain": "explain", "report": "report"}
    )
    graph.add_edge("explain", "persist")
    graph.add_edge("persist", "alert")
    graph.add_edge("alert", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_pipeline(
    file_path: str,
    persist: bool = True,
    generate_report: bool = False,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Convenience entry point: upload_file() -> ... -> display_dashboard()
    (TRD Sec. 29), minus the dashboard rendering itself, which lives in app.py.

    Args:
        file_path: ``[DATA_SOURCE_PATH]`` - the .xlsx file to analyse (read-only).
        persist: save to the DB and send alert emails.
        generate_report: also produce the anomaly-free dataset, business report and charts.
        output_dir: parent folder for those outputs (default: config ``anomaly_analysis.output_dir``).
    """
    app = build_pipeline()
    final_state = app.invoke(
        {"file_path": file_path, "persist": persist, "generate_report": generate_report, "output_dir": output_dir}
    )
    return final_state["report"]
