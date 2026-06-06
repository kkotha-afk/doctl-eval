"""Construct the ground-truth gold set for the scored evaluation.

Methodology (defensible, and honest about its limits):

  Ground truth here is a *triangulation*, not a single source:
    1. Maintainer labels (human signal) mapped to the 6-class schema. These are
       noisy and inconsistent (the brief's whole point), so they are an anchor,
       not gospel.
    2. Two INDEPENDENT frontier labelers (Claude Opus 4.8 + GPT-5.4) propose a
       label for every sampled issue.

  Resolution per issue:
    - maintainer present & both models agree with it  -> 'unanimous'        (auto)
    - maintainer present & one model agrees           -> 'majority_human'   (auto, gold=maintainer)
    - maintainer absent  & both models agree          -> 'model_consensus'  (auto, UNLESS security/other)
    - everything else (human-vs-model conflict, model split, any security/other)
                                                      -> 'review'           (manual adjudication)

  Why this shape:
    - It keeps gold anchored to human labels where they exist.
    - It quantifies maintainer-label noise (we record maintainer vs final gold).
    - The two labelers are FRONTIER models that are NOT the cheap models we will
      recommend, so the production recommendation never rests on a model grading
      its own homework. Their scores being high on gold they co-defined is
      expected and disclosed.

  Sampling is stratified (fixed seed -> stable set) so rare, high-stakes classes
  (security, documentation, question, other) get usable support instead of 2-3
  examples from a naive random draw. Per-class support is reported so thin
  numbers are visible.

Run:
  python harness/build_gold.py            # build sample, get proposals, auto-resolve
                                          # -> writes data/gold.jsonl (+ gold_review.jsonl)
  Then manually adjudicate gold_review.jsonl entries and merge (see --merge-review).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random

import client as si_client
import config

SEED = 20260606
SAMPLE_PATH = config.DATA_DIR / "gold_sample.json"
REVIEW_PATH = config.DATA_DIR / "gold_review.jsonl"
PROPOSALS_PATH = config.DATA_DIR / "gold_proposals.json"

LABELER_A = "anthropic-claude-opus-4.8"
LABELER_B = "openai-gpt-5.4"

# Maintainer label -> schema. Only category-bearing labels appear; process/area
# labels (hacktoberfest, good first issue, do-api, snap, ...) intentionally omitted.
CATEGORY_MAP = {
    "bug": "bug",
    "suggestion": "enhancement",
    "enhancement": "enhancement",
    "security vulnerability": "security",
    "question": "question",
    "docs": "documentation",
}
# When an issue carries several category labels, pick by this precedence.
PRECEDENCE = ["security", "bug", "question", "documentation", "enhancement"]

# Stratified targets (counts). 'None' = issues with no mappable maintainer label.
STRATA_TARGETS = {
    "security": 26,        # rare + high-stakes -> take (nearly) all
    "question": 12,        # take all
    "documentation": 8,    # few exist; take all + borderline
    "bug": 45,
    "enhancement": 45,
    None: 30,              # unlabeled -> surface 'other' and natural mix
}


def map_maintainer(labels: list[str]) -> str | None:
    mapped = {CATEGORY_MAP[l.lower()] for l in labels if l.lower() in CATEGORY_MAP}
    for c in PRECEDENCE:
        if c in mapped:
            return c
    return None


def load_corpus_by_number() -> dict[int, dict]:
    return {json.loads(l)["number"]: json.loads(l)
            for l in open(config.CORPUS_PATH, encoding="utf-8")}


# --- 1. stratified sample --------------------------------------------------
def build_sample() -> list[int]:
    if SAMPLE_PATH.exists():
        return json.load(open(SAMPLE_PATH, encoding="utf-8"))

    corpus = load_corpus_by_number()
    buckets: dict[str | None, list[int]] = {}
    for num, issue in corpus.items():
        key = map_maintainer(issue["maintainer_labels"])
        buckets.setdefault(key, []).append(num)

    rng = random.Random(SEED)
    chosen: list[int] = []
    for stratum, target in STRATA_TARGETS.items():
        pool = sorted(buckets.get(stratum, []))
        rng.shuffle(pool)
        chosen.extend(pool[:target])
    chosen = sorted(set(chosen))
    json.dump(chosen, open(SAMPLE_PATH, "w", encoding="utf-8"))
    print(f"Built stratified sample: {len(chosen)} issues -> {SAMPLE_PATH.name}")
    return chosen


# --- 2. frontier labeler proposals (cached) --------------------------------
async def get_proposals(numbers: list[int]) -> dict[str, dict[int, dict]]:
    if PROPOSALS_PATH.exists():
        cached = json.load(open(PROPOSALS_PATH, encoding="utf-8"))
        return {m: {int(k): v for k, v in d.items()} for m, d in cached.items()}

    corpus = load_corpus_by_number()
    issues = [corpus[n] for n in numbers]
    client = si_client.make_client()
    sem = asyncio.Semaphore(config.DEFAULT_CONCURRENCY)

    async def one(model_id: str, issue: dict):
        async with sem:
            return issue["number"], await si_client.classify_one(client, model_id, issue)

    proposals: dict[str, dict[int, dict]] = {}
    for model_id in (LABELER_A, LABELER_B):
        print(f"Labeling sample with {model_id} ...")
        results = await asyncio.gather(*(one(model_id, i) for i in issues))
        proposals[model_id] = {num: rec for num, rec in results}
    await client.close()

    json.dump({m: {str(k): v for k, v in d.items()} for m, d in proposals.items()},
              open(PROPOSALS_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"Cached proposals -> {PROPOSALS_PATH.name}")
    return proposals


# --- 3. resolve ------------------------------------------------------------
def resolve(numbers: list[int], proposals: dict[str, dict[int, dict]]) -> None:
    corpus = load_corpus_by_number()
    gold: list[dict] = []
    review: list[dict] = []
    source_counts: dict[str, int] = {}

    for num in numbers:
        issue = corpus[num]
        maint = map_maintainer(issue["maintainer_labels"])
        pa = proposals[LABELER_A][num]["label"]
        pb = proposals[LABELER_B][num]["label"]

        gold_label, source = None, "review"
        if maint is not None:
            if pa == pb == maint:
                gold_label, source = maint, "unanimous"
            elif maint in (pa, pb):
                gold_label, source = maint, "majority_human"
            else:
                source = "review"  # human vs models conflict
        else:
            if pa is not None and pa == pb and pa not in ("security", "other"):
                gold_label, source = pa, "model_consensus"
            else:
                source = "review"  # model split, or sensitive class -> human eyes

        entry = {
            "number": num,
            "maintainer_mapped": maint,
            "proposal_a": pa,
            "proposal_b": pb,
            "source": source,
        }
        if gold_label is not None:
            gold.append({"number": num, "label": gold_label, "source": source,
                         "maintainer_mapped": maint})
        else:
            entry["title"] = issue["title"]
            entry["body_excerpt"] = (issue.get("body") or "")[:800]
            entry["url"] = issue.get("url")
            entry["suggested"] = pa if pa == pb else (maint or pa or pb)
            review.append(entry)
        source_counts[source] = source_counts.get(source, 0) + 1

    with open(config.GOLD_PATH, "w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    with open(REVIEW_PATH, "w", encoding="utf-8") as f:
        for r in review:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nResolved {len(numbers)} sampled issues:")
    for src, n in sorted(source_counts.items()):
        print(f"  {src:16} {n}")
    print(f"\nAuto-resolved gold: {len(gold)} -> {config.GOLD_PATH.name}")
    print(f"Need manual review: {len(review)} -> {REVIEW_PATH.name}")


# --- 4. merge manual review decisions --------------------------------------
def merge_review(decisions_path: str) -> None:
    """decisions_path: jsonl of {number, label} adjudicated by hand. Appends/updates
    gold.jsonl with source='manual'."""
    decisions = {json.loads(l)["number"]: json.loads(l)["label"]
                 for l in open(decisions_path, encoding="utf-8")}
    corpus = load_corpus_by_number()
    existing = {json.loads(l)["number"]: json.loads(l)
                for l in open(config.GOLD_PATH, encoding="utf-8")}
    for num, label in decisions.items():
        assert label in config.LABEL_SET, f"bad label {label!r} for #{num}"
        existing[num] = {"number": num, "label": label, "source": "manual",
                         "maintainer_mapped": map_maintainer(corpus[num]["maintainer_labels"])}
    with open(config.GOLD_PATH, "w", encoding="utf-8") as f:
        for num in sorted(existing):
            f.write(json.dumps(existing[num], ensure_ascii=False) + "\n")
    print(f"Merged {len(decisions)} manual decisions. Gold now {len(existing)} issues.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge-review", metavar="DECISIONS.jsonl",
                    help="merge hand-adjudicated {number,label} decisions into gold.jsonl")
    args = ap.parse_args()

    if args.merge_review:
        merge_review(args.merge_review)
        return

    numbers = build_sample()
    proposals = asyncio.run(get_proposals(numbers))
    resolve(numbers, proposals)


if __name__ == "__main__":
    main()
