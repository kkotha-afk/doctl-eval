"""Central configuration: paths, env, the label schema, pricing, and the model registry.

Everything that the rest of the harness needs to agree on lives here so there is a
single source of truth. Cost math reads rates from data/pricing.json (loaded here)
so every dollar figure is traceable back to a published per-token rate.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# --- Paths -----------------------------------------------------------------
# ROOT is the repo root (parent of the harness/ package directory). Resolving
# relative to __file__ means scripts work regardless of the current working dir.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PROMPTS_DIR = ROOT / "prompts"

CORPUS_PATH = DATA_DIR / "corpus.jsonl"
GOLD_PATH = DATA_DIR / "gold.jsonl"
PRICING_PATH = DATA_DIR / "pricing.json"

RESULTS_DIR.mkdir(exist_ok=True)

# --- Environment -----------------------------------------------------------
load_dotenv(ROOT / ".env")

SI_BASE_URL = os.getenv("SI_BASE_URL", "https://inference.do-ai.run/v1")
SI_API_KEY = os.getenv("SI_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
# Concurrency is read at runtime (not baked into the image) so it can be tuned
# per-run via an env var without rebuilding the container.
DEFAULT_CONCURRENCY = int(os.getenv("CONCURRENCY", "8"))

# --- The customer's label schema -------------------------------------------
# Exactly one of these per issue. Order is fixed so confusion matrices and
# distributions are always laid out the same way.
LABELS = ["bug", "enhancement", "question", "documentation", "security", "other"]
LABEL_SET = set(LABELS)

# --- doctl repository ------------------------------------------------------
GITHUB_OWNER = "digitalocean"
GITHUB_REPO = "doctl"


# --- Pricing ---------------------------------------------------------------
def load_pricing() -> dict[str, dict[str, float]]:
    """Return {model_id: {"input": $/1M, "output": $/1M}} from data/pricing.json."""
    with open(PRICING_PATH, encoding="utf-8") as f:
        return json.load(f)["models"]


PRICING = load_pricing()


def price_per_token(model_id: str) -> tuple[float, float]:
    """(input_$_per_token, output_$_per_token) for a model. Raises if unpriced,
    so we never silently report $0.00 for a model we forgot to add rates for."""
    if model_id not in PRICING:
        raise KeyError(
            f"No pricing for {model_id!r}. Add it to data/pricing.json so cost is traceable."
        )
    rates = PRICING[model_id]
    return rates["input"] / 1_000_000, rates["output"] / 1_000_000


def call_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Dollar cost of one call: tokens * per-token rate. This is the *only* place
    cost is computed, so the formula is auditable in one spot."""
    in_rate, out_rate = price_per_token(model_id)
    return prompt_tokens * in_rate + completion_tokens * out_rate


# --- Model registry --------------------------------------------------------
@dataclass(frozen=True)
class Model:
    id: str          # exact SI model id passed to the API
    label: str       # human-friendly name for the UI
    family: str      # anthropic / openai / open-weight
    open_weight: bool
    reasoning: bool  # reasoning/"thinking" model?
    note: str = ""


# The broader-evaluation candidate set. The two we recommend are an OUTPUT of
# comparing these, not a starting assumption. Spread spans the axes the brief
# names: open-weight vs frontier, small vs large, reasoning vs non-reasoning.
CANDIDATES: list[Model] = [
    Model("anthropic-claude-opus-4.8", "Claude Opus 4.8", "anthropic", False, False,
          "Expensive frontier baseline — the 'what they run today'."),
    Model("openai-gpt-5.4", "GPT-5.4", "openai", False, False,
          "Frontier, non-Anthropic."),
    Model("anthropic-claude-haiku-4.5", "Claude Haiku 4.5", "anthropic", False, False,
          "Cheap frontier-family; fallback candidate."),
    Model("openai-gpt-4o-mini", "GPT-4o mini", "openai", False, False,
          "Cheap small frontier."),
    Model("openai-gpt-oss-120b", "gpt-oss-120b", "openai", True, False,
          "Open-weight, large, ~48x cheaper than Opus; primary candidate."),
    Model("openai-gpt-oss-20b", "gpt-oss-20b", "openai", True, False,
          "Open-weight, small; cost floor."),
    Model("alibaba-qwen3-32b", "Qwen3-32B", "open-weight", True, False,
          "Open-weight mid-size."),
    Model("gemma-4-31B-it", "Gemma 4 31B", "open-weight", True, False,
          "Open-weight mid-size (Google)."),
    Model("llama3.3-70b-instruct", "Llama 3.3 70B", "open-weight", True, False,
          "Open-weight large workhorse."),
    Model("mistral-3-14B", "Ministral 3 14B", "open-weight", True, False,
          "Open-weight small, flat $0.20 in/out."),
    Model("openai-o3-mini", "o3-mini (reasoning)", "openai", False, True,
          "Reasoning model — included to show reasoning is overkill for classification."),
]

CANDIDATES_BY_ID = {m.id: m for m in CANDIDATES}

# The expensive baseline the customer is assumed to run today.
BASELINE_MODEL_ID = "anthropic-claude-opus-4.8"
