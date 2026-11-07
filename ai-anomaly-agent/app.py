"""FR-10: Dashboard.

KPI cards, trend charts, an anomaly panel, and access to historical
alerts (Sec. 15). Also the entry point that runs the LangGraph pipeline
on an uploaded file (Sec. 29: upload_file() -> ... -> display_dashboard()).

Run with: streamlit run app.py
"""
from __future__ import annotations

import json
import os
import tempfile

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.config_loader import load_config
from src.database.db import get_recent_anomalies, get_session, MetricRecord
from src.graph.pipeline import run_pipeline

load_dotenv()

st.set_page_config(page_title="AI Anomaly Agent", layout="wide")
st.title("📊 AI Anomaly Agent")
st.caption("Detect → Explain → Alert — statistical detection, AI explanation, email alerts.")

cfg = load_config()

# --- Sidebar: upload & run -------------------------------------------------
with st.sidebar:
    st.header("Upload business data")
    uploaded = st.file_uploader("Excel file (.xlsx)", type=["xlsx"])
    persist = st.checkbox("Save results & send email alerts", value=False)
    run_clicked = st.button("Run analysis", type="primary", disabled=uploaded is None)

if run_clicked and uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name
    with st.spinner("Running pipeline: validate → clean → detect → explain..."):
        report = run_pipeline(tmp_path, persist=persist, generate_report=True)
    os.unlink(tmp_path)
    st.session_state["last_report"] = report

report = st.session_state.get("last_report")

# --- Validation feedback ---------------------------------------------------
if report:
    if report.get("validation_errors"):
        st.error("Validation failed:")
        for e in report["validation_errors"]:
            st.write(f"- {e}")
    else:
        if report.get("validation_warnings"):
            with st.expander(f"{len(report['validation_warnings'])} validation warning(s)"):
                for w in report["validation_warnings"]:
                    st.write(f"- {w}")

        # KPI cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Anomalies found", report["total_anomalies"])
        col2.metric("Critical", report["critical_count"])
        col3.metric("Warning", report["warning_count"])
        col4.metric("Low", report["low_count"])

        analysis = report.get("analysis")
        if report.get("analysis_error"):
            st.warning(f"Historical analysis unavailable: {report['analysis_error']}")
        elif analysis:
            st.subheader("Raw vs anomaly-free data")
            s = analysis["summary"]
            st.caption(
                f"{s['total_anomalies']} anomalous value(s) flagged across {s['rows_with_anomalies']} of "
                f"{s['rows_analysed']} rows. Original data is unchanged; the anomaly-free dataset is a separate output."
            )
            files = analysis["files"]
            if "chart_timeseries" in files:
                st.image(files["chart_timeseries"], use_container_width=True)
            d1, d2, d3 = st.columns(3)
            for col, key, label in ((d1, "report", "Business report (.md)"),
                                    (d2, "anomaly_free", "Anomaly-free data (.csv)"),
                                    (d3, "flagged", "Flagged anomalies (.csv)")):
                if key in files:
                    with open(files[key], "rb") as fh:
                        col.download_button(label, fh.read(), file_name=os.path.basename(files[key]))
            st.caption(f"All outputs saved in: {analysis['output_dir']}")

        st.subheader("Anomaly panel")
        if report["anomalies"]:
            for a in report["anomalies"]:
                severity_emoji = {"Critical": "🔴", "Warning": "🟠", "Low": "🟡"}.get(a["severity"], "⚪")
                with st.expander(
                    f"{severity_emoji} {a['metric']} — {a['severity']} ({(a.get('change_percent') or 0):.1f}% change) on {a['date']}"
                ):
                    c1, c2, c3 = st.columns(3)
                    c1.write(f"**Current:** {a.get('current_value')}")
                    c2.write(f"**Baseline:** {a.get('baseline_value')}")
                    c3.write(f"**Z-score:** {a.get('z_score')}")
                    explanation = a.get("explanation", {})
                    if explanation:
                        st.write(f"**What happened:** {explanation.get('what_happened')}")
                        st.write("**Possible causes (hypotheses):**")
                        for cause in explanation.get("possible_causes", []):
                            st.write(f"- {cause}")
                        st.write(f"**Business impact:** {explanation.get('business_impact')}")
                        st.write("**Recommended investigation:**")
                        for step in explanation.get("recommended_investigation", []):
                            st.write(f"- {step}")
        else:
            st.success("No anomalies detected in this file.")
else:
    st.info("Upload an Excel file in the sidebar and click **Run analysis** to get started.")

# --- Trend charts from historical DB data ---------------------------------
st.subheader("Historical trends")
try:
    with get_session() as session:
        records = session.query(MetricRecord).order_by(MetricRecord.date).all()
    if records:
        hist_df = pd.DataFrame(
            [
                {
                    "Date": r.date,
                    "Revenue": r.revenue,
                    "Orders": r.orders,
                    "Conversion Rate": r.conversion_rate,
                    "Traffic": r.traffic,
                    "Cost": r.cost,
                    "Refunds": r.refunds,
                }
                for r in records
            ]
        )
        metric_choice = st.selectbox("Metric", list(cfg["metrics"].values()))
        fig = px.line(hist_df, x="Date", y=metric_choice, markers=True, title=f"{metric_choice} over time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No historical data saved yet — run an analysis with 'Save results' checked.")
except Exception as exc:  # noqa: BLE001
    st.caption(f"Historical trends unavailable: {exc}")

# --- Alert history ----------------------------------------------------------
st.subheader("Alert history")
try:
    recent = get_recent_anomalies(limit=25)
    if recent:
        history_df = pd.DataFrame(
            [
                {
                    "Date": a.date,
                    "Metric": a.metric,
                    "Severity": a.severity,
                    "Change %": a.change_percent,
                    "Z-score": a.z_score,
                    "Logged at": a.created_at,
                }
                for a in recent
            ]
        )
        st.dataframe(history_df, use_container_width=True)
    else:
        st.caption("No anomaly history yet.")
except Exception as exc:  # noqa: BLE001
    st.caption(f"Alert history unavailable: {exc}")
