"""Pull doctl issues from the GitHub public API into a stable local snapshot.

Deliberately thin (the brief says so). The output data/corpus.jsonl is the
*stable corpus*: we fetch once, commit it, and every eval run reads that file.
Re-fetching is opt-in (--force) because the live repo keeps changing and the
whole point is reproducible numbers across runs.

Run:  python harness/ingest.py            # fetch if corpus.jsonl is missing
      python harness/ingest.py --force    # re-snapshot from GitHub
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import httpx

import config

API = "https://api.github.com"
PER_PAGE = 100


def _headers() -> dict[str, str]:
    h = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "doctl-eval-ingest",
    }
    if config.GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {config.GITHUB_TOKEN}"
    return h


def fetch_all_issues() -> list[dict]:
    """Fetch every issue (open + closed), excluding pull requests.

    Two GitHub quirks handled here:
      1. The /issues endpoint returns PRs too; GitHub marks those with a
         'pull_request' key, which is the documented way to tell them apart.
      2. REST pagination is capped at ~1000 items (page 11 -> HTTP 422). doctl
         has more issues+PRs than that, so we walk the history in updated-at
         order and, when we hit the cap, advance a `since` cursor to the last
         seen timestamp and resume from page 1. Dedup by issue number absorbs
         the boundary overlap. This is the documented workaround for the cap.
    """
    seen: dict[int, dict] = {}
    since: str | None = None
    page = 1
    last_updated = None
    windows = 0
    with httpx.Client(headers=_headers(), timeout=30.0) as client:
        while True:
            params = {"state": "all", "per_page": PER_PAGE, "page": page,
                      "sort": "updated", "direction": "asc"}
            if since:
                params["since"] = since
            resp = client.get(
                f"{API}/repos/{config.GITHUB_OWNER}/{config.GITHUB_REPO}/issues",
                params=params,
            )
            # Respect secondary rate limits politely.
            if resp.status_code == 403 and "rate limit" in resp.text.lower():
                reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
                wait = max(reset - int(time.time()), 5)
                print(f"  rate limited; sleeping {wait}s (set GITHUB_TOKEN to avoid)", file=sys.stderr)
                time.sleep(wait)
                continue
            # 1000-item pagination cap: window forward via `since`.
            if resp.status_code == 422:
                if last_updated and last_updated != since and windows < 50:
                    since, page, windows = last_updated, 1, windows + 1
                    print(f"  hit pagination cap; advancing since={since}")
                    continue
                break
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            for it in batch:
                if "pull_request" in it:  # skip PRs
                    continue
                seen.setdefault(it["number"], _slim(it))
            last_updated = batch[-1]["updated_at"]
            print(f"  page {page}: +{len(batch)} raw, {len(seen)} unique issues kept")
            if len(batch) < PER_PAGE:
                break
            page += 1
    return list(seen.values())


def _slim(it: dict) -> dict:
    """Keep only the fields the eval needs. Full body is preserved (truncation
    happens at classify time, not here) so the snapshot stays faithful."""
    return {
        "number": it["number"],
        "title": it.get("title") or "",
        "body": it.get("body") or "",
        "state": it.get("state"),
        "maintainer_labels": [lbl["name"] for lbl in it.get("labels", [])],
        "created_at": it.get("created_at"),
        "closed_at": it.get("closed_at"),
        "comments": it.get("comments", 0),
        "url": it.get("html_url"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Snapshot doctl issues to data/corpus.jsonl")
    ap.add_argument("--force", action="store_true",
                    help="re-fetch even if corpus.jsonl already exists")
    args = ap.parse_args()

    if config.CORPUS_PATH.exists() and not args.force:
        n = sum(1 for _ in open(config.CORPUS_PATH, encoding="utf-8"))
        print(f"corpus.jsonl already exists ({n} issues). Use --force to re-snapshot.")
        return

    print(f"Fetching {config.GITHUB_OWNER}/{config.GITHUB_REPO} issues...")
    issues = fetch_all_issues()
    issues.sort(key=lambda x: x["number"])  # deterministic order

    config.DATA_DIR.mkdir(exist_ok=True)
    with open(config.CORPUS_PATH, "w", encoding="utf-8") as f:
        for it in issues:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    open_n = sum(1 for it in issues if it["state"] == "open")
    labeled_n = sum(1 for it in issues if it["maintainer_labels"])
    print(f"\nWrote {len(issues)} issues -> {config.CORPUS_PATH}")
    print(f"  open: {open_n}   closed: {len(issues) - open_n}")
    print(f"  with >=1 maintainer label: {labeled_n}")


if __name__ == "__main__":
    main()
