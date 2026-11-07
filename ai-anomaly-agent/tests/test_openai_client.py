"""Exercises the real OpenAI SDK request/parse path against a fake HTTP transport.

The gateway talks to OpenRouter through the OpenAI-compatible Chat Completions
protocol. No network and no API key are needed: an ``OpenAI`` client with the
same ``base_url``/headers the gateway would build is wired to an
``httpx.MockTransport`` and injected into ``src.ai.openai_client``.
"""
import json

import httpx
import pytest
from openai import OpenAI

from src.ai import openai_client
from src.ai.explainer import AnomalyExplanation, explain_anomaly
from src.ai.openai_client import OpenAIConfigError, StructuredCallError, request_structured
from src.config_loader import load_config

EXPLANATION = {
    "what_happened": "Revenue fell sharply.",
    "possible_causes": ["Checkout issue (verify)"],
    "business_impact": "Lost sales.",
    "recommended_investigation": ["Check payment logs"],
}


def completion(content: str, finish_reason: str = "stop", **message_extra) -> dict:
    return {
        "id": "gen-test", "object": "chat.completion", "created": 0, "model": "openai/gpt-5-mini",
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": content, **message_extra}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.fixture
def install_client(monkeypatch):
    """Install a client whose HTTP layer is ``handler``; returns the captured requests."""
    captured: list[httpx.Request] = []

    def _install(handler):
        def wrapped(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return handler(request)

        # Build the client through the gateway's own code path so base_url/headers are the real ones.
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("OPENROUTER_SITE_URL", "https://example.test")
        openai_client.reset_client()
        real = OpenAI

        def factory(**kw):
            kw["max_retries"] = 0
            kw["http_client"] = httpx.Client(transport=httpx.MockTransport(wrapped))
            return real(**kw)

        monkeypatch.setattr(openai_client, "OpenAI", factory)
        return captured

    yield _install
    openai_client.reset_client()


def ok(payload=EXPLANATION):
    return lambda r: httpx.Response(200, json=completion(json.dumps(payload)))


def test_request_targets_openrouter_with_bearer_auth(install_client):
    captured = install_client(ok())
    result = request_structured("SYSTEM", "USER", AnomalyExplanation)
    assert isinstance(result, AnomalyExplanation) and result.what_happened == "Revenue fell sharply."

    req = captured[0]
    assert str(req.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer test-key"
    assert req.headers["http-referer"] == "https://example.test"
    assert req.headers["x-title"] == "AI Anomaly Agent"


def test_request_body_shape(install_client):
    captured = install_client(ok())
    request_structured("SYSTEM", "USER", AnomalyExplanation)
    body = json.loads(captured[0].content)

    assert body["model"] == "openai/gpt-5-mini"
    assert body["messages"] == [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "USER"}]
    assert body["max_tokens"] == 2000
    rf = body["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    # OpenRouter routing preferences travel in the body
    assert body["provider"] == {"require_parameters": True, "data_collection": "deny"}
    # parameters that belong to the OLD (OpenAI Responses) integration must not leak through
    for stale in ("instructions", "input", "text", "max_output_tokens", "store", "temperature", "reasoning"):
        assert stale not in body


def test_model_env_override_and_fallback_models(install_client, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "anthropic/claude-sonnet-4.5")
    cfg = load_config()["ai"]
    monkeypatch.setitem(cfg, "fallback_models", ["openai/gpt-5-mini"])
    captured = install_client(ok())
    request_structured("s", "u", AnomalyExplanation)
    body = json.loads(captured[0].content)
    assert body["model"] == "anthropic/claude-sonnet-4.5"
    assert body["models"] == ["anthropic/claude-sonnet-4.5", "openai/gpt-5-mini"]


def test_openai_provider_mode_sends_no_openrouter_extras(install_client, monkeypatch):
    cfg = load_config()["ai"]
    monkeypatch.setitem(cfg, "provider", "openai")
    monkeypatch.setitem(cfg, "base_url", "https://api.openai.com/v1")
    captured = install_client(ok())
    request_structured("s", "u", AnomalyExplanation)
    req = captured[0]
    assert str(req.url) == "https://api.openai.com/v1/chat/completions"
    assert "http-referer" not in req.headers and "provider" not in json.loads(req.content)


def test_reasoning_effort_is_forwarded_the_openrouter_way(install_client, monkeypatch):
    monkeypatch.setitem(load_config()["ai"], "reasoning_effort", "low")
    captured = install_client(ok())
    request_structured("s", "u", AnomalyExplanation)
    assert json.loads(captured[0].content)["reasoning"] == {"effort": "low"}


@pytest.mark.parametrize(
    "status, needle",
    [(401, "credentials"), (402, "Insufficient credits"), (403, "blocked"), (429, "Rate limit"),
     (400, "rejected the request"), (502, "Provider API error 502")],
)
def test_http_errors_become_structured_call_error(install_client, status, needle):
    install_client(lambda r: httpx.Response(status, json={"error": {"code": status, "message": "boom"}}))
    with pytest.raises(StructuredCallError, match=needle):
        request_structured("s", "u", AnomalyExplanation)


def test_connection_error_is_translated(install_client):
    def boom(request):
        raise httpx.ConnectError("no route", request=request)

    install_client(boom)
    with pytest.raises(StructuredCallError, match="Could not reach https://openrouter.ai/api/v1"):
        request_structured("s", "u", AnomalyExplanation)


def test_truncated_reply_is_reported(install_client):
    install_client(lambda r: httpx.Response(200, json=completion('{"what_happened": "cut', finish_reason="length")))
    with pytest.raises(StructuredCallError, match="token limit"):
        request_structured("s", "u", AnomalyExplanation)


def test_invalid_json_from_a_weak_model_is_reported(install_client):
    install_client(lambda r: httpx.Response(200, json=completion("Sure! Here is your answer.")))
    with pytest.raises(StructuredCallError, match="could not be parsed"):
        request_structured("s", "u", AnomalyExplanation)


def test_refusal_is_reported(install_client):
    install_client(lambda r: httpx.Response(200, json=completion(None, refusal="I can't help with that.")))
    with pytest.raises(StructuredCallError, match="refused"):
        request_structured("s", "u", AnomalyExplanation)


def test_openrouter_http_200_with_finish_reason_error(install_client):
    """OpenRouter reports upstream-model failures as HTTP 200 + finish_reason 'error'."""
    body = completion("", finish_reason="error")
    body["choices"][0]["error"] = {"code": 502, "message": "upstream down"}
    install_client(lambda r: httpx.Response(200, json=body))
    with pytest.raises(StructuredCallError, match="finish_reason=error"):
        request_structured("s", "u", AnomalyExplanation)


def test_http_200_with_error_object_instead_of_completion(install_client):
    install_client(lambda r: httpx.Response(200, json={"error": {"code": 502, "message": "Provider returned error"}}))
    with pytest.raises(StructuredCallError):
        request_structured("s", "u", AnomalyExplanation)


def test_missing_api_key_raises_config_error(monkeypatch):
    openai_client.reset_client()
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(OpenAIConfigError, match="OPENROUTER_API_KEY"):
        openai_client.get_client()


def test_explain_anomaly_uses_ai_when_available(install_client):
    install_client(ok())
    out = explain_anomaly({"metric": "Revenue", "date": "2026-01-10", "change_percent": -32.0})
    assert out.business_impact == "Lost sales."


@pytest.mark.parametrize("status", [402, 429, 502])
def test_explain_anomaly_falls_back_on_failure(install_client, status):
    install_client(lambda r: httpx.Response(status, json={"error": {"code": status, "message": "x"}}))
    out = explain_anomaly({"metric": "Revenue", "date": "2026-01-10", "change_percent": -32.0})
    assert "-32.0%" in out.what_happened and "AI explanation unavailable" in out.possible_causes[0]


NARRATIVE = {
    "headline": "Revenue dropped sharply on the last day.",
    "key_findings": ["Revenue -32%"], "business_risks": ["Lost sales"], "recommended_actions": ["Check checkout"],
}
SUMMARY = {
    "period": {"start": "a", "end": "b"}, "rows_analysed": 30, "metrics": ["Revenue"], "total_anomalies": 1,
    "critical_count": 1, "warning_count": 0, "cleaning_strategy": "interpolate", "metric_impacts": [],
    "top_anomalies": [{"Metric": "Revenue", "Date": "2026-01-30", "Value": 68.0, "Expected": 100.0, "Deviation_Pct": -32.0}],
    "rows_with_anomalies": 1,
}


def test_narrative_uses_llm_when_available(install_client):
    from src.analysis.narrative import build_narrative
    install_client(ok(NARRATIVE))
    nar, source = build_narrative(SUMMARY, use_ai=True)
    assert source == "llm" and nar.headline.startswith("Revenue dropped")


def test_narrative_falls_back_to_template_on_api_error(install_client):
    from src.analysis.narrative import build_narrative
    install_client(lambda r: httpx.Response(402, json={"error": {"code": 402, "message": "credits"}}))
    nar, source = build_narrative(SUMMARY, use_ai=True)
    assert source == "template" and "Revenue" in nar.headline
