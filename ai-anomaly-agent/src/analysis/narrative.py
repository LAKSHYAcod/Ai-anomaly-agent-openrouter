"""Executive narrative for the business report.

All numbers are computed deterministically elsewhere. The model only turns
an already-aggregated findings summary into readable prose, mirroring the
repo's principle that statistics decide *what* happened and the LLM only
helps explain it. If the API is unavailable, a template-based narrative is
produced from the same facts, so the report is always complete.
"""
from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from src.ai.openai_client import OpenAIConfigError, StructuredCallError, request_structured

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior business analyst writing for executives. You are given \
pre-computed findings from a statistical anomaly-detection run on business metrics.

Rules:
- Use ONLY the numbers and facts in the findings. Never invent figures, causes or events.
- Flagged values may be genuine business events (promotions, outages, stock-outs) as well as \
data errors - say they need verification, and do not assert a cause.
- Values in the anomaly-free dataset are statistical estimates, not observations - say so where relevant.
- Be concise, plain-English, and avoid statistical jargon."""

USER_PROMPT = """Findings (JSON):
{findings_json}"""


class ExecutiveNarrative(BaseModel):
    headline: str = Field(description="One-sentence takeaway for an executive.")
    key_findings: list[str] = Field(description="3-5 specific findings grounded in the supplied numbers.")
    business_risks: list[str] = Field(description="Risks if the flagged anomalies are real or are ignored.")
    recommended_actions: list[str] = Field(description="Concrete, prioritised next steps for a human to take.")


def _template_narrative(summary: dict) -> ExecutiveNarrative:
    n = summary["total_anomalies"]
    if n == 0:
        return ExecutiveNarrative(
            headline="No anomalies were detected; the dataset can be used as supplied.",
            key_findings=[f"{summary['rows_analysed']} rows across {len(summary['metrics'])} metrics were analysed with no flagged values."],
            business_risks=["None identified by the statistical checks."],
            recommended_actions=["Continue routine monitoring."],
        )
    worst = summary["top_anomalies"][0]
    findings = [
        f"{n} anomalous value(s) were flagged across {summary['rows_with_anomalies']} of "
        f"{summary['rows_analysed']} rows ({summary['critical_count']} critical, {summary['warning_count']} warning).",
        f"The largest deviation was {worst['Metric']} on {worst['Date']}: {worst['Value']:,.2f} "
        f"versus an expected {worst['Expected']:,.2f} ({worst['Deviation_Pct']:+.1f}%).",
    ]
    for impact in summary["metric_impacts"][:3]:
        if impact["anomalies"] and impact["total_difference"] is not None:
            findings.append(
                f"{impact['metric']}: {impact['anomalies']} flagged value(s); replacing them changes the period "
                f"total by {impact['total_difference']:+,.2f} ({impact['total_difference_pct']:+.1f}%)."
            )
    return ExecutiveNarrative(
        headline=f"{n} anomalous value(s) found; the most severe is in {worst['Metric']} on {worst['Date']}.",
        key_findings=findings,
        business_risks=["If the flagged values are real events they may signal revenue, cost or funnel problems; "
                        "if they are errors, unadjusted reports will misstate performance."],
        recommended_actions=[
            "Verify each Critical item against source systems before acting on the cleaned figures.",
            "Use the anomaly-free dataset for trend analysis only after the flags have been reviewed.",
        ],
    )


def build_narrative(summary: dict, use_ai: bool = True) -> tuple[ExecutiveNarrative, str]:
    """Return ``(narrative, source)`` where source is ``"llm"`` or ``"template"``."""
    if use_ai and summary["total_anomalies"] > 0:
        try:
            findings = json.dumps(
                {k: summary[k] for k in ("period", "rows_analysed", "metrics", "total_anomalies", "critical_count",
                                         "warning_count", "top_anomalies", "metric_impacts", "cleaning_strategy")},
                default=str,
            )
            return request_structured(SYSTEM_PROMPT, USER_PROMPT.format(findings_json=findings), ExecutiveNarrative), "llm"
        except (OpenAIConfigError, StructuredCallError) as exc:
            logger.warning("AI narrative unavailable, using template: %s", exc)
        except Exception:  # noqa: BLE001 - never let narrative generation break reporting
            logger.exception("Unexpected error generating AI narrative; using template")
    return _template_narrative(summary), "template"
