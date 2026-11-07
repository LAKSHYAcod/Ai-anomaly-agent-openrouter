"""Single gateway to the LLM provider (OpenRouter by default, OpenAI optional).

Every LLM call in the project goes through :func:`request_structured`, so
endpoint, authentication, retries, structured-output parsing and error
translation are implemented exactly once.

Why this works for both providers
---------------------------------
OpenRouter exposes the OpenAI Chat Completions protocol, so the official
``openai`` SDK is used unchanged - only ``base_url``, the API key and the
model ID differ (all in ``config.yaml``). ``client.chat.completions.parse``
with a Pydantic model sends a strict ``response_format`` JSON schema and
returns a validated object.

Chat Completions (not the Responses API) is used deliberately: it is
OpenRouter's stable, canonical endpoint, whereas its ``/responses`` endpoint
is beta and stateless (it rejects ``store=true``).

OpenRouter specifics handled here
---------------------------------
* ``Authorization: Bearer <key>`` is added by the SDK from ``api_key``.
* Optional attribution headers (``HTTP-Referer`` / ``X-Title``).
* ``provider.require_parameters`` so a request is only routed to providers
  that honour ``response_format``; ``provider.data_collection="deny"`` for privacy.
* Optional fallback models (``models`` list).
* Errors that arrive as HTTP 200 with ``finish_reason == "error"``.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import TypeVar

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from src.config_loader import load_config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OpenAIConfigError(RuntimeError):
    """Raised when the client cannot be built (e.g. the API key env var is unset)."""


class StructuredCallError(RuntimeError):
    """Raised when an LLM call fails for any reason, with a user-readable message."""


_client: OpenAI | None = None
_client_lock = threading.Lock()


def _ai_cfg() -> dict:
    return load_config()["ai"]


def _is_openrouter(cfg: dict | None = None) -> bool:
    return (cfg or _ai_cfg()).get("provider", "openrouter") == "openrouter"


def resolve_model() -> str:
    """LLM_MODEL (env) wins over ``ai.model`` (config)."""
    return os.environ.get("LLM_MODEL") or _ai_cfg()["model"]


def _attribution_headers(cfg: dict) -> dict[str, str]:
    """Optional OpenRouter app-attribution headers (never required)."""
    if not _is_openrouter(cfg):
        return {}
    headers: dict[str, str] = {}
    site = os.environ.get("OPENROUTER_SITE_URL") or cfg.get("site_url")
    title = cfg.get("app_title")
    if site:
        headers["HTTP-Referer"] = site
    if title:
        headers["X-Title"] = title
    return headers


def get_client() -> OpenAI:
    """Return a process-wide client, creating it on first use.

    Built lazily so that importing this module (and running the statistical
    test-suite) never requires credentials.
    """
    global _client
    if _client is not None:
        return _client

    with _client_lock:
        if _client is not None:
            return _client
        cfg = _ai_cfg()
        env_name = cfg["api_key_env_var"]  # [OPENROUTER_API_KEY]
        api_key = os.environ.get(env_name)
        if not api_key:
            raise OpenAIConfigError(
                f"Environment variable '{env_name}' is not set. "
                "Export it (or add it to your .env file) to enable AI features."
            )
        _client = OpenAI(
            api_key=api_key,                      # sent as "Authorization: Bearer <api_key>"
            base_url=cfg["base_url"],             # https://openrouter.ai/api/v1
            default_headers=_attribution_headers(cfg) or None,
            timeout=cfg.get("timeout_seconds", 60),
            max_retries=cfg.get("max_retries", 3),
        )
        return _client


def reset_client() -> None:
    """Drop the cached client (used by tests and after rotating credentials)."""
    global _client
    with _client_lock:
        _client = None


def _build_request_kwargs(system_prompt: str, user_prompt: str, schema: type[T]) -> dict:
    cfg = _ai_cfg()
    model = resolve_model()
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},   # Responses API: `instructions`
            {"role": "user", "content": user_prompt},       # Responses API: `input`
        ],
        "response_format": schema,                           # Responses API: `text_format`
        "max_tokens": cfg["max_tokens"],                     # Responses API: `max_output_tokens`
    }
    # Optional knobs are only sent when explicitly configured, because
    # reasoning models reject `temperature`.
    if cfg.get("temperature") is not None:
        kwargs["temperature"] = cfg["temperature"]

    if _is_openrouter(cfg):
        extra: dict = {}
        prefs: dict = {}
        if cfg.get("require_parameters"):
            prefs["require_parameters"] = True
        if cfg.get("data_collection"):
            prefs["data_collection"] = cfg["data_collection"]
        if prefs:
            extra["provider"] = prefs
        if cfg.get("fallback_models"):
            extra["models"] = [model, *cfg["fallback_models"]]
        if cfg.get("reasoning_effort"):
            extra["reasoning"] = {"effort": cfg["reasoning_effort"]}
        if extra:
            kwargs["extra_body"] = extra
    elif cfg.get("reasoning_effort"):
        kwargs["reasoning_effort"] = cfg["reasoning_effort"]  # native OpenAI parameter
    return kwargs


def request_structured(system_prompt: str, user_prompt: str, schema: type[T]) -> T:
    """Call the model and return a validated instance of ``schema``.

    Raises:
        OpenAIConfigError: credentials are missing.
        StructuredCallError: the call failed, was refused, was truncated,
            or returned something that does not match ``schema``.
    """
    client = get_client()
    cfg = _ai_cfg()
    kwargs = _build_request_kwargs(system_prompt, user_prompt, schema)

    try:
        completion = client.chat.completions.parse(**kwargs)
    except openai.AuthenticationError as exc:
        raise StructuredCallError(
            f"The provider rejected the credentials in '{cfg['api_key_env_var']}' (401)."
        ) from exc
    except openai.RateLimitError as exc:
        raise StructuredCallError(
            "Rate limit or quota exceeded after retries (429). Check usage limits and credits."
        ) from exc
    except openai.PermissionDeniedError as exc:
        raise StructuredCallError(
            f"The request was blocked (403) - moderation or key/guardrail policy: {exc.message}"
        ) from exc
    except openai.BadRequestError as exc:
        # Typically an unsupported parameter, or (OpenRouter) a model/provider without structured outputs.
        raise StructuredCallError(
            f"The provider rejected the request for model '{kwargs['model']}' (400): {exc.message}"
        ) from exc
    except openai.APITimeoutError as exc:
        raise StructuredCallError("The request timed out after retries.") from exc
    except openai.APIConnectionError as exc:
        raise StructuredCallError(f"Could not reach {cfg['base_url']}: {exc}") from exc
    except openai.APIStatusError as exc:
        if exc.status_code == 402:  # OpenRouter: insufficient credits
            raise StructuredCallError(
                "Insufficient credits (402). Add credits at https://openrouter.ai/credits and retry."
            ) from exc
        raise StructuredCallError(f"Provider API error {exc.status_code}: {exc.message}") from exc
    except openai.LengthFinishReasonError as exc:
        raise StructuredCallError(
            "The model's reply was cut off by the token limit. Increase ai.max_tokens in config.yaml."
        ) from exc
    except openai.ContentFilterFinishReasonError as exc:
        raise StructuredCallError("The reply was withheld by a content filter.") from exc
    except openai.OpenAIError as exc:
        raise StructuredCallError(f"Unexpected SDK error: {exc}") from exc
    except ValidationError as exc:
        raise StructuredCallError(
            "The model's reply could not be parsed into the expected schema. If this keeps happening, "
            "pick a model/provider that supports structured outputs."
        ) from exc
    except (TypeError, AttributeError, IndexError, KeyError) as exc:
        # e.g. an HTTP 200 whose body is an {"error": ...} object instead of a completion.
        raise StructuredCallError(f"The provider returned a malformed response: {exc}") from exc

    choices = getattr(completion, "choices", None) or []
    if not choices:
        err = getattr(completion, "error", None)
        raise StructuredCallError(f"The provider returned no choices. {err or ''}".strip())

    choice = choices[0]
    # OpenRouter reports upstream-model failures as HTTP 200 + finish_reason "error".
    if getattr(choice, "finish_reason", None) == "error":
        detail = getattr(choice, "error", None) or getattr(completion, "error", None) or "no detail provided"
        raise StructuredCallError(f"The upstream model failed mid-request (finish_reason=error): {detail}")

    message = choice.message
    if getattr(message, "refusal", None):
        raise StructuredCallError(f"The model refused the request: {message.refusal}")
    if message.parsed is None:
        raise StructuredCallError("The model returned no structured output.")
    return message.parsed
