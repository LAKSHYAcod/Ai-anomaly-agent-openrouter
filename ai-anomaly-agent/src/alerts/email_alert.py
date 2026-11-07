"""FR-08 / Sec. 13: Email Alert.

Sends alerts for Warning/Critical anomalies via SMTP. Credentials are
read exclusively from environment variables (Sec. 31) - never hard-code
them here.
"""
from __future__ import annotations

import os
import smtplib
from email.mime.text import MIMEText

from src.config_loader import load_config


class EmailSendError(Exception):
    pass


def _format_message(anomaly: dict, explanation: dict) -> str:
    lines = [
        f"Metric: {anomaly['metric']}",
        f"Date: {anomaly['date']}",
        f"Current value: {anomaly.get('current_value')}",
        f"Baseline: {anomaly.get('baseline_value')}",
        f"Change: {anomaly.get('change_percent')}%",
        f"Severity: {anomaly['severity']}",
        "",
        f"What happened: {explanation.get('what_happened')}",
        "",
        "Possible causes (hypotheses - verify against source data):",
    ]
    for cause in explanation.get("possible_causes", []):
        lines.append(f"  - {cause}")
    lines += [
        "",
        f"Business impact: {explanation.get('business_impact')}",
        "",
        "Recommended investigation:",
    ]
    for step in explanation.get("recommended_investigation", []):
        lines.append(f"  - {step}")
    return "\n".join(lines)


def send_alert_email(anomaly: dict, explanation: dict, recipient: str | None = None) -> tuple[str, str, str]:
    """Send the alert email. Returns (subject, message, status) so the
    caller can persist it to alert history regardless of outcome.
    """
    cfg = load_config()["alerts"]
    subject = f"{cfg['subject_prefix']} — {anomaly['metric']} ({anomaly['severity']})"
    message = _format_message(anomaly, explanation)

    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    to_addr = recipient or os.environ.get("ALERT_RECIPIENT")

    if not all([smtp_host, smtp_user, smtp_password, to_addr]):
        # Don't crash the pipeline - report the report is still valuable.
        return subject, message, "skipped_missing_config"

    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_addr

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, [to_addr], msg.as_string())
        return subject, message, "sent"
    except Exception as exc:  # noqa: BLE001
        return subject, message, f"failed: {exc}"
