"""FR-07 / Sec. 12 & 28: AI Explanation Layer.

This is the ONLY place an LLM is used. It receives structured anomaly
data (never the raw spreadsheet) and returns a strict JSON schema:
what_happened, possible_causes, business_impact, recommended_investigation.

The statistical engine has already decided *that* something is
anomalous; the model only explains *what it might mean*, and is
explicitly instructed to separate observation from hypothesis and to
avoid inventing facts (PRD Sec. 35).
"""
from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from src.ai.openai_client import OpenAIConfigError, StructuredCallError, request_structured

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a business analytics assistant. You will be given \
structured, pre-computed anomaly data from a statistical detection system. \
The system has already determined that this data point is anomalous - your \
job is ONLY to explain what it might mean in plain business language.

Rules you must follow:
- Never invent numbers, causes, or facts that are not implied by the data given to you.
- Clearly separate "what happened" (an observation of the data) from "possible causes" \
(hypotheses that must be verified against source data, not treated as fact).
- Be concise and business-friendly - avoid statistical jargon like "z-score" in your prose.
- If related metrics are provided, use them to make possible_causes more specific \
(e.g. stable traffic + falling conversion suggests a funnel or pricing issue, not a traffic issue)."""

USER_PROMPT = """Anomaly data:
{anomaly_json}

Related metric context (other metrics on the same date, for comparison):
{related_json}"""


class AnomalyExplanation(BaseModel):
    what_happened: str = Field(description="Plain-language observation of the data change.")
    possible_causes: list[str] = Field(
        description="Hypotheses only - phrased as possibilities to verify, not facts."
    )
    business_impact: str = Field(description="Why this change matters to the business.")
    recommended_investigation: list[str] = Field(
        description="Concrete next steps a human should take to verify the cause."
    )


def explain_anomaly(anomaly: dict, related_metrics: list[dict] | None = None) -> AnomalyExplanation:
    """Generate a business explanation for a single anomaly dict
    (as produced by src.anomaly.severity.annotate_severity).

    Falls back to a clearly-labeled statistical-only explanation if the
    AI call fails, per TRD Sec. 30: "If the AI service fails, the
    statistical anomaly report should still be available."
    """
    try:
        user_prompt = USER_PROMPT.format(
            anomaly_json=json.dumps(anomaly, default=str),
            related_json=json.dumps(related_metrics or [], default=str),
        )
        return request_structured(SYSTEM_PROMPT, user_prompt, AnomalyExplanation)
    except (OpenAIConfigError, StructuredCallError) as exc:
        # Expected failure modes: missing key, network, quota, refusal, truncation.
        logger.warning("AI explanation unavailable, using statistical fallback: %s", exc)
        return _fallback_explanation(anomaly, str(exc))
    except Exception as exc:  # noqa: BLE001 - last-resort net: AI must never break the pipeline
        logger.exception("Unexpected error while generating AI explanation")
        return _fallback_explanation(anomaly, str(exc))


def _fallback_explanation(anomaly: dict, reason: str) -> AnomalyExplanation:
    """Statistics-only explanation used whenever the AI layer is unavailable."""
    change = anomaly.get("change_percent")
    if change is not None:
        what = f"{anomaly.get('metric')} changed by {change:.1f}% versus its recent baseline on {anomaly.get('date')}."
    else:
        what = f"{anomaly.get('metric')} deviated from its recent baseline on {anomaly.get('date')}."
    return AnomalyExplanation(
        what_happened=what,
        possible_causes=[f"AI explanation unavailable - statistical detection only. (reason: {reason})"],
        business_impact="Unable to generate AI business-impact assessment at this time.",
        recommended_investigation=["Review the raw metric data manually for this date."],
    )
