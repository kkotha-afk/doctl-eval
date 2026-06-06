"""Scoring and comparison — pure functions over result dicts and the gold set.

No I/O of model calls here; this turns persisted predictions into the numbers the
UI shows. Metrics are computed by hand (not sklearn) so every figure is auditable:
you can see exactly how TP/FP/FN produce precision, recall, and F1.
"""
from __future__ import annotations

import json

import config


# --- loading ---------------------------------------------------------------
def load_result(corpus_tag: str, model_id: str) -> dict:
    from runner import result_path
    with open(result_path(corpus_tag, model_id), encoding="utf-8") as f:
        return json.load(f)


def load_gold() -> dict[int, str]:
    """{issue_number: gold_label}."""
    out = {}
    with open(config.GOLD_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["number"]] = row["label"]
    return out


def records_by_number(result: dict) -> dict[int, dict]:
    return {r["number"]: r for r in result["records"]}


# --- scored metrics (vs ground truth) --------------------------------------
def score_against_gold(result: dict, gold: dict[int, str]) -> dict:
    """Accuracy + per-class P/R/F1 + confusion matrix over the gold subset.

    Per-class/confusion are computed over gold issues the model actually
    classified (pred not None); failed calls are reported separately as
    `n_unclassified` so a model can't hide misses by erroring out.
    """
    recs = records_by_number(result)
    labels = config.LABELS

    pairs = []           # (gold_label, pred_label) for classified gold issues
    n_unclassified = 0
    gold_cost = 0.0      # cost of ONLY the gold-subset calls (correct for full runs too)
    for num, g in gold.items():
        r = recs.get(num)
        if r is None:
            continue                      # model wasn't run on this issue
        gold_cost += r.get("cost", 0.0)
        if r["label"] is None:
            n_unclassified += 1           # call failed -> a miss, tracked apart
            continue
        pairs.append((g, r["label"]))

    n_classified = len(pairs)
    n_correct = sum(1 for g, p in pairs if g == p)
    accuracy = n_correct / n_classified if n_classified else 0.0

    # Confusion matrix: confusion[gold][pred] = count.
    confusion = {g: {p: 0 for p in labels} for g in labels}
    for g, p in pairs:
        confusion[g][p] += 1

    # Per-class precision / recall / F1 from the matrix.
    per_class = {}
    for c in labels:
        tp = confusion[c][c]
        fp = sum(confusion[g][c] for g in labels if g != c)
        fn = sum(confusion[c][p] for p in labels if p != c)
        support = sum(confusion[c].values())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": support}

    present = [c for c in labels if per_class[c]["support"] > 0]
    macro_f1 = sum(per_class[c]["f1"] for c in present) / len(present) if present else 0.0
    total_support = sum(per_class[c]["support"] for c in labels)
    weighted_f1 = (
        sum(per_class[c]["f1"] * per_class[c]["support"] for c in labels) / total_support
        if total_support else 0.0
    )

    # Cost per correct classification — the headline economic metric. Computed
    # from the gold-subset call costs so it's valid whether `result` is a gold
    # run or a full run that merely contains the gold issues.
    cost_per_correct = gold_cost / n_correct if n_correct else float("inf")

    return {
        "accuracy": accuracy,
        "n_gold": len(gold),
        "n_classified": n_classified,
        "n_correct": n_correct,
        "n_unclassified": n_unclassified,
        "per_class": per_class,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion": confusion,
        "gold_subset_cost_usd": gold_cost,
        "cost_per_correct_usd": cost_per_correct,
    }


# --- unscored comparison (no ground truth) ---------------------------------
def compare_unscored(result_a: dict, result_b: dict) -> dict:
    """Agreement rate, per-model label distribution, and the disagreement list
    over the issues both models classified."""
    a = records_by_number(result_a)
    b = records_by_number(result_b)
    common = sorted(set(a) & set(b))

    agree = 0
    n_compared = 0
    disagreements = []
    dist_a = {l: 0 for l in config.LABELS}
    dist_b = {l: 0 for l in config.LABELS}

    for num in common:
        la, lb = a[num]["label"], b[num]["label"]
        if la in dist_a:
            dist_a[la] += 1
        if lb in dist_b:
            dist_b[lb] += 1
        if la is None or lb is None:
            continue                      # can't compare if either failed
        n_compared += 1
        if la == lb:
            agree += 1
        else:
            disagreements.append({"number": num, "label_a": la, "label_b": lb})

    return {
        "agreement_rate": agree / n_compared if n_compared else 0.0,
        "n_compared": n_compared,
        "n_agree": agree,
        "n_disagree": len(disagreements),
        "dist_a": dist_a,
        "dist_b": dist_b,
        "disagreements": disagreements,
    }
