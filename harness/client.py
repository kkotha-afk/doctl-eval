"""One inference call = one issue. This module owns the network call, retries,
error categorization, and per-call measurement (latency, tokens, cost).

Per the brief: every issue is its own request (no batching), so each call is
independently retryable and individually routable to a fallback model.
"""
from __future__ import annotations

import asyncio
import random
import time

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    RateLimitError,
)

import classify
import config


def make_client(timeout: float = 60.0) -> AsyncOpenAI:
    if not config.SI_API_KEY:
        raise RuntimeError("SI_API_KEY is not set (put it in .env).")
    return AsyncOpenAI(api_key=config.SI_API_KEY, base_url=config.SI_BASE_URL,
                       timeout=timeout, max_retries=0)  # we do our own retries


def _completion_kwargs(model_id: str, messages: list[dict]) -> dict:
    """Per-model param quirks. Reasoning ('thinking') models reject temperature
    and meter reasoning tokens as output, so they need headroom and the
    max_completion_tokens param instead of max_tokens."""
    m = config.CANDIDATES_BY_ID.get(model_id)
    is_reasoning = bool(m and m.reasoning)
    if is_reasoning:
        return {"model": model_id, "messages": messages, "max_completion_tokens": 2000}
    return {"model": model_id, "messages": messages, "temperature": 0, "max_tokens": 200}


def _classify_error(exc: Exception) -> str:
    """Bucket exceptions into the taxonomy the UI reports."""
    if isinstance(exc, RateLimitError):
        return "rate_limit"
    if isinstance(exc, APITimeoutError):
        return "timeout"
    if isinstance(exc, BadRequestError):
        return "bad_request"          # our bug (bad param) — do not retry
    if isinstance(exc, APIStatusError):
        return "server" if exc.status_code >= 500 else "other"
    if isinstance(exc, APIConnectionError):
        return "connection"
    return "other"


_RETRYABLE = {"rate_limit", "timeout", "server", "connection"}


async def classify_one(client: AsyncOpenAI, model_id: str, issue: dict,
                       max_attempts: int = 4) -> dict:
    """Classify a single issue. Always returns a record (never raises): failures
    are captured as error/error_type so the run can score and report them."""
    messages = classify.build_messages(issue)
    kwargs = _completion_kwargs(model_id, messages)

    attempts = 0
    last_error_type = None
    last_error_msg = None
    t0 = time.perf_counter()

    while attempts < max_attempts:
        attempts += 1
        try:
            resp = await client.chat.completions.create(**kwargs)
            latency = time.perf_counter() - t0
            text = resp.choices[0].message.content or ""
            usage = resp.usage
            pt = getattr(usage, "prompt_tokens", 0) or 0
            ct = getattr(usage, "completion_tokens", 0) or 0
            parsed = classify.parse_response(text)
            # A successful HTTP call that yields no usable label is a parse error
            # (route to review) rather than a silently-wrong classification.
            err_type = None if parsed["label"] is not None else "parse_error"
            return {
                "number": issue["number"],
                "label": parsed["label"],
                "confidence": parsed["confidence"],
                "reason": parsed["reason"],
                "parse_ok": parsed["parse_ok"],
                "raw": text,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": pt + ct,
                "cost": config.call_cost(model_id, pt, ct),
                "latency_s": round(latency, 4),
                "attempts": attempts,
                "error": None if err_type is None else "no parsable label",
                "error_type": err_type,
            }
        except Exception as exc:  # noqa: BLE001 — we intentionally capture all
            last_error_type = _classify_error(exc)
            last_error_msg = f"{type(exc).__name__}: {exc}"[:300]
            if last_error_type not in _RETRYABLE or attempts >= max_attempts:
                break
            # Exponential backoff with jitter; honor Retry-After on 429 if present.
            delay = min(2 ** (attempts - 1), 30) + random.uniform(0, 0.5)
            retry_after = getattr(getattr(exc, "response", None), "headers", {}) or {}
            if "retry-after" in {k.lower() for k in retry_after}:
                try:
                    delay = float(retry_after.get("retry-after") or retry_after.get("Retry-After"))
                except (TypeError, ValueError):
                    pass
            await asyncio.sleep(delay)

    # All attempts failed.
    return {
        "number": issue["number"],
        "label": None, "confidence": None, "reason": "", "parse_ok": False, "raw": "",
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0,
        "latency_s": round(time.perf_counter() - t0, 4),
        "attempts": attempts,
        "error": last_error_msg, "error_type": last_error_type or "other",
    }
