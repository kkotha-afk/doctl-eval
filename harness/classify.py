"""Prompt construction and response parsing for issue classification.

Kept separate from the network/concurrency code so the *decision logic* (how we
ask and how we interpret the answer) is easy to read and reason about on its own.
"""
from __future__ import annotations

import json
import re

import config

PROMPT_VERSION = "v1"
SYSTEM_PROMPT = (config.PROMPTS_DIR / f"classify_{PROMPT_VERSION}.txt").read_text(encoding="utf-8")

# Issue bodies can contain huge logs/stack traces. Truncate to bound input tokens
# (and thus cost/latency). ~6000 chars ~= ~1.5k tokens; enough signal for a label.
MAX_BODY_CHARS = 6000


def build_user_message(issue: dict) -> str:
    """Render one issue into the user turn. Title + (truncated) body only —
    deliberately NOT the maintainer labels, so the model can't cheat off them."""
    body = (issue.get("body") or "").strip()
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + "\n...[truncated]"
    title = (issue.get("title") or "").strip()
    return f"Issue #{issue.get('number')}\nTitle: {title}\n\nBody:\n{body or '(no body)'}"


def build_messages(issue: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(issue)},
    ]


# Match a bare label token as a fallback when JSON parsing fails.
_LABEL_RE = re.compile(r"\b(" + "|".join(config.LABELS) + r")\b", re.IGNORECASE)


def parse_response(text: str) -> dict:
    """Turn raw model text into {label, confidence, reason, parse_ok}.

    Robust to models that wrap JSON in markdown fences or add prose. If we can't
    recover a valid label we return parse_ok=False with label=None so the caller
    can route it to review rather than silently coercing to a wrong class.
    """
    raw = (text or "").strip()
    label, confidence, reason, parse_ok = None, None, "", False

    # 1) Try strict JSON, including JSON embedded in a larger string.
    candidate = raw
    if "```" in candidate:  # strip markdown fences
        candidate = re.sub(r"```(?:json)?", "", candidate).strip()
    m = re.search(r"\{.*\}", candidate, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            lbl = str(obj.get("label", "")).strip().lower()
            if lbl in config.LABEL_SET:
                label, parse_ok = lbl, True
                conf = obj.get("confidence")
                if isinstance(conf, (int, float)):
                    confidence = max(0.0, min(1.0, float(conf)))
                reason = str(obj.get("reason", ""))[:200]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # 2) Fallback: first valid label token anywhere in the text.
    if not parse_ok:
        m2 = _LABEL_RE.search(raw)
        if m2:
            label = m2.group(1).lower()
            reason = "recovered-from-unstructured-output"
            # parse_ok stays False: we got a label but the model didn't follow format.

    return {"label": label, "confidence": confidence, "reason": reason, "parse_ok": parse_ok}
