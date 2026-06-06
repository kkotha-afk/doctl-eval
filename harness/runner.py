"""Run a model over a corpus (subset) with bounded concurrency, then persist the
per-issue records and an aggregate run-metadata block.

Concurrency is a runtime arg (CLI flag or CONCURRENCY env), never baked in, so it
can be tuned without rebuilding the image. Each result file is self-describing:
it carries the config it was produced under (model, concurrency, prompt version,
corpus, timestamp) plus the operational metrics, so the UI just reads and renders.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from datetime import datetime, timezone

import client as si_client
import config


# --- corpus loading --------------------------------------------------------
def load_corpus() -> list[dict]:
    return [json.loads(l) for l in open(config.CORPUS_PATH, encoding="utf-8")]


def load_gold_numbers() -> set[int]:
    if not config.GOLD_PATH.exists():
        raise FileNotFoundError("data/gold.jsonl not found — build the gold set first.")
    return {json.loads(l)["number"] for l in open(config.GOLD_PATH, encoding="utf-8")}


def select_issues(corpus_tag: str, limit: int | None) -> list[dict]:
    issues = load_corpus()
    if corpus_tag == "gold":
        nums = load_gold_numbers()
        issues = [i for i in issues if i["number"] in nums]
    elif corpus_tag != "full":
        raise ValueError("corpus must be 'full' or 'gold'")
    issues.sort(key=lambda x: x["number"])
    return issues[:limit] if limit else issues


# --- metrics ---------------------------------------------------------------
def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def safe_model_name(model_id: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in model_id)


def result_path(corpus_tag: str, model_id: str):
    return config.RESULTS_DIR / f"{corpus_tag}__{safe_model_name(model_id)}.json"


# --- run -------------------------------------------------------------------
async def run_model(model_id: str, corpus_tag: str, concurrency: int,
                    max_attempts: int, limit: int | None) -> dict:
    issues = select_issues(corpus_tag, limit)
    if model_id not in config.PRICING:
        raise KeyError(f"{model_id} has no pricing entry; refusing to run un-costable model.")

    client = si_client.make_client()
    sem = asyncio.Semaphore(concurrency)
    done = 0
    n = len(issues)

    async def worker(issue: dict) -> dict:
        nonlocal done
        async with sem:
            rec = await si_client.classify_one(client, model_id, issue, max_attempts)
        done += 1
        if done % 25 == 0 or done == n:
            print(f"  [{model_id}] {done}/{n}")
        return rec

    print(f"Running {model_id} on '{corpus_tag}' corpus: {n} issues @ concurrency={concurrency}")
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    records = await asyncio.gather(*(worker(i) for i in issues))
    wall = time.perf_counter() - t0
    await client.close()

    # --- aggregate operational metrics ---
    # Latency percentiles measured over requests that got an HTTP response
    # (success or parse_error), excluding hard failures whose timing includes
    # retry backoff sleeps and would distort serving latency.
    responded = [r["latency_s"] for r in records if r["error_type"] in (None, "parse_error")]
    responded.sort()
    total_cost = sum(r["cost"] for r in records)
    n_failed = sum(1 for r in records if r["label"] is None)
    err_breakdown = Counter(r["error_type"] for r in records if r["error_type"] is not None)
    n_parse_recovered = sum(1 for r in records if r["label"] is not None and not r["parse_ok"])

    meta = {
        "model_id": model_id,
        "corpus_tag": corpus_tag,
        "n_issues": n,
        "concurrency": concurrency,
        "max_attempts": max_attempts,
        "prompt_version": __import__("classify").PROMPT_VERSION,
        "started_at": started.isoformat(),
        "wall_clock_s": round(wall, 3),
        "throughput_rps": round(n / wall, 3) if wall > 0 else 0.0,
        "total_cost_usd": round(total_cost, 6),
        "cost_per_call_usd": round(total_cost / n, 8) if n else 0.0,
        "latency_p50_s": round(_pct(responded, 0.50), 4),
        "latency_p95_s": round(_pct(responded, 0.95), 4),
        "latency_mean_s": round(sum(responded) / len(responded), 4) if responded else 0.0,
        "latency_sample_n": len(responded),
        "tokens_in_total": sum(r["prompt_tokens"] for r in records),
        "tokens_out_total": sum(r["completion_tokens"] for r in records),
        "n_failed": n_failed,
        "error_rate": round(n_failed / n, 4) if n else 0.0,
        "n_parse_recovered": n_parse_recovered,
        "error_breakdown": dict(err_breakdown),
        "pricing": config.PRICING[model_id],  # snapshot the rates used for traceability
    }

    out = {"meta": meta, "records": records}
    path = result_path(corpus_tag, model_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=0)

    print(f"  done: {wall:.1f}s  cost=${total_cost:.4f}  p50={meta['latency_p50_s']}s "
          f"p95={meta['latency_p95_s']}s  fail={n_failed}  -> {path.name}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Classify a corpus with one or more models.")
    ap.add_argument("--model", help="single SI model id")
    ap.add_argument("--all-candidates", action="store_true", help="run every registry candidate")
    ap.add_argument("--corpus", default="full", choices=["full", "gold"])
    ap.add_argument("--concurrency", type=int, default=config.DEFAULT_CONCURRENCY)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="cap issue count (smoke tests)")
    args = ap.parse_args()

    if args.all_candidates:
        models = [m.id for m in config.CANDIDATES]
    elif args.model:
        models = [args.model]
    else:
        ap.error("pass --model <id> or --all-candidates")

    async def run_all():
        for mid in models:
            try:
                await run_model(mid, args.corpus, args.concurrency, args.max_attempts, args.limit)
            except Exception as exc:  # one model failing shouldn't kill the sweep
                print(f"  ERROR running {mid}: {type(exc).__name__}: {exc}")

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
