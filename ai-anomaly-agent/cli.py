"""Command-line entry point for the AI Anomaly Agent.

Runs the same LangGraph pipeline the Streamlit dashboard uses, but
prints a readable report to the terminal - useful for scheduled jobs
(cron, GitHub Actions, a cloud scheduler - see TRD Sec. 33 Deployment)
where there's no browser to look at.

Usage (``[DATA_SOURCE_PATH]`` = path to the .xlsx input; it is only ever read):
    python cli.py [DATA_SOURCE_PATH]
    python cli.py [DATA_SOURCE_PATH] --persist
    python cli.py [DATA_SOURCE_PATH] --output-dir reports --json
    python cli.py [DATA_SOURCE_PATH] --no-report

The LLM key is read from the environment variable named in
config.yaml (ai.api_key_env_var, default OPENROUTER_API_KEY - the ``[OPENROUTER_API_KEY]`` placeholder).
"""
from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

from src.graph.pipeline import run_pipeline

SEVERITY_ICON = {"Critical": "🔴", "Warning": "🟠", "Low": "🟡", "Normal": "⚪"}


def _print_human_report(report: dict) -> None:
    if report.get("validation_errors"):
        print("❌ Validation failed:")
        for err in report["validation_errors"]:
            print(f"   - {err}")
        return

    if report.get("validation_warnings"):
        print(f"⚠️  {len(report['validation_warnings'])} validation warning(s):")
        for w in report["validation_warnings"]:
            print(f"   - {w}")
        print()

    _print_analysis(report)

    total = report["total_anomalies"]
    print(
        f"Summary: {total} anomaly(ies) — "
        f"{report['critical_count']} critical, "
        f"{report['warning_count']} warning, "
        f"{report['low_count']} low"
    )

    if not report["anomalies"]:
        print("✅ No anomalies detected.")
        return

    print()
    for a in report["anomalies"]:
        icon = SEVERITY_ICON.get(a["severity"], "⚪")
        change = a.get("change_percent")
        change_str = f"{change:.1f}%" if change is not None else "n/a"
        print(f"{icon} {a['metric']} — {a['severity']} ({change_str} change) on {a['date']}")
        print(f"   current={a.get('current_value')}  baseline={a.get('baseline_value')}  z={a.get('z_score')}")
        explanation = a.get("explanation")
        if explanation:
            print(f"   What happened: {explanation.get('what_happened')}")
            print(f"   Business impact: {explanation.get('business_impact')}")
            causes = explanation.get("possible_causes", [])
            if causes:
                print("   Possible causes (verify, don't assume):")
                for c in causes:
                    print(f"     - {c}")
            steps = explanation.get("recommended_investigation", [])
            if steps:
                print("   Recommended investigation:")
                for s in steps:
                    print(f"     - {s}")
        print()


def _print_analysis(report: dict) -> None:
    if report.get("analysis_error"):
        print(f"⚠️  Historical analysis failed: {report['analysis_error']}\n")
        return
    analysis = report.get("analysis")
    if not analysis:
        return
    s = analysis["summary"]
    print(
        f"📈 Historical analysis: {s['total_anomalies']} anomalous value(s) in {s['rows_with_anomalies']} of "
        f"{s['rows_analysed']} rows → anomaly-free dataset built ({s['cleaning_strategy']})."
    )
    print(f"   Report & charts: {analysis['output_dir']}")
    for err in analysis.get("errors", []):
        print(f"   ⚠️  {err}")
    print()


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run the AI Anomaly Agent pipeline (validate → clean → detect → explain → alert) on an Excel file."
    )
    parser.add_argument("file_path", metavar="[DATA_SOURCE_PATH]", help="Path to the .xlsx file to analyze (read-only)")
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Save results to the database and send email alerts for Warning/Critical anomalies. "
        "Off by default (dry run).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw JSON report instead of a human-readable summary.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Parent folder for the business report, anomaly-free dataset and charts "
        "(default: anomaly_analysis.output_dir from config). A new run_<timestamp> folder is created each run.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip the anomaly-free dataset, business report and charts (monitoring/alerting only).",
    )
    args = parser.parse_args(argv)

    try:
        report = run_pipeline(
            args.file_path,
            persist=args.persist,
            generate_report=not args.no_report,
            output_dir=args.output_dir,
        )
    except Exception as exc:  # noqa: BLE001 - last-resort handler: clean message + distinct exit code
        print(f"❌ Unexpected failure: {exc}", file=sys.stderr)
        return 3

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print_human_report(report)

    # Non-zero exit code on validation failure or any Critical anomaly -
    # convenient for CI/cron jobs that want to signal "needs attention".
    if report.get("validation_errors"):
        return 2
    if report.get("critical_count", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
