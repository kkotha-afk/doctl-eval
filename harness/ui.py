"""Streamlit eval dashboard — the viewer half of the harness.

Reads persisted result files (produced by runner.py) and renders:
  - Leaderboard: the broader evaluation across all candidate models (gold set).
  - Scored view: accuracy, per-class P/R/F1, confusion matrices, disagreement drill-down.
  - Unscored view: agreement rate, suggestion distributions, disagreement filter, raw output.
  - Operational metrics: cost, latency (with concurrency), throughput, wall-clock, errors.
  - Run/methodology: trigger a run with configurable concurrency; explain ground truth.

The viewer never calls a model itself except via the explicit "run" button, which
shells out to runner.py — so the numbers on screen always come from a persisted,
reproducible run.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

import config
import scoring
from runner import result_path, safe_model_name

st.set_page_config(page_title="doctl issue-classifier eval", layout="wide")

# Models we recommend out of the broader sweep (the side-by-side default).
RECOMMENDED = ["openai-gpt-oss-120b", "anthropic-claude-haiku-4.5"]


# --- data loading ----------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_result_cached(path_str: str, mtime: float) -> dict:
    with open(path_str, encoding="utf-8") as f:
        return json.load(f)


def load_result(corpus_tag: str, model_id: str) -> dict | None:
    p = result_path(corpus_tag, model_id)
    if not p.exists():
        return None
    return load_result_cached(str(p), p.stat().st_mtime)


@st.cache_data(show_spinner=False)
def load_gold_cached(mtime: float) -> dict[int, str]:
    return scoring.load_gold()


def gold() -> dict[int, str]:
    return load_gold_cached(config.GOLD_PATH.stat().st_mtime)


@st.cache_data(show_spinner=False)
def corpus_index() -> dict[int, dict]:
    return {json.loads(l)["number"]: json.loads(l)
            for l in open(config.CORPUS_PATH, encoding="utf-8")}


def available_models(corpus_tag: str) -> list[str]:
    """Model ids that have a result file for this corpus, ordered by the registry."""
    prefix = f"{corpus_tag}__"
    have = set()
    for p in config.RESULTS_DIR.glob(f"{prefix}*.json"):
        stem = p.name[len(prefix):-len(".json")]
        for m in config.CANDIDATES:
            if safe_model_name(m.id) == stem:
                have.add(m.id)
    ordered = [m.id for m in config.CANDIDATES if m.id in have]
    return ordered


def display_name(model_id: str) -> str:
    m = config.CANDIDATES_BY_ID.get(model_id)
    return m.label if m else model_id


# --- leaderboard -----------------------------------------------------------
def render_leaderboard(corpus_tag: str):
    st.subheader("Broader evaluation — all candidates on the gold set")
    st.caption(
        "The two recommended models are the **output** of this sweep, not an assumption. "
        "Each row is one inference run over the 142-issue gold set. "
        "Claude Opus 4.8 and GPT-5.4 co-defined ground truth (gold labelers), so their "
        "scores are partly definitional — read them as an upper anchor, not a fair contestant."
    )
    g = gold()
    rows = []
    for mid in available_models(corpus_tag):
        res = load_result(corpus_tag, mid)
        if not res:
            continue
        sc = scoring.score_against_gold(res, g)
        meta = res["meta"]
        m = config.CANDIDATES_BY_ID.get(mid)
        rows.append({
            "Model": display_name(mid),
            "Type": ("frontier" if m and not m.open_weight else "open-weight") + (" · reasoning" if m and m.reasoning else ""),
            "Accuracy": sc["accuracy"],
            "Macro-F1": sc["macro_f1"],
            "Cost/correct ($)": sc["cost_per_correct_usd"],
            "Gold cost ($)": sc["gold_subset_cost_usd"],
            "p95 (s)": meta["latency_p95_s"],
            "Throughput (rps)": meta["throughput_rps"],
            "Fails": sc["n_unclassified"],
            "_is_rec": mid in RECOMMENDED,
        })
    if not rows:
        st.info("No gold results yet. Run the sweep: `python harness/runner.py --all-candidates --corpus gold`")
        return
    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)

    def highlight_rec(row):
        return ["background-color: #14432a" if row["_is_rec"] else "" for _ in row]

    show = df.drop(columns=["_is_rec"])
    styler = (df.style.apply(highlight_rec, axis=1)
              .format({"Accuracy": "{:.1%}", "Macro-F1": "{:.3f}",
                       "Cost/correct ($)": "{:.5f}", "Gold cost ($)": "{:.4f}",
                       "p95 (s)": "{:.2f}", "Throughput (rps)": "{:.1f}"})
              .hide(axis="columns", subset=["_is_rec"]))
    st.dataframe(styler, width="stretch", hide_index=True)
    st.caption("Highlighted rows = the two models recommended for production. "
               "Cost/correct = dollars spent on the gold subset ÷ correct classifications.")


# --- scored view -----------------------------------------------------------
def per_class_table(sc: dict) -> pd.DataFrame:
    rows = []
    for c in config.LABELS:
        pc = sc["per_class"][c]
        rows.append({"class": c, "precision": pc["precision"], "recall": pc["recall"],
                     "f1": pc["f1"], "support": pc["support"]})
    return pd.DataFrame(rows)


def confusion_df(sc: dict) -> pd.DataFrame:
    conf = sc["confusion"]
    df = pd.DataFrame(conf).T  # rows = gold (true), cols = pred
    df = df.reindex(index=config.LABELS, columns=config.LABELS).fillna(0).astype(int)
    df.index.name = "true ↓ / pred →"
    return df


def style_confusion(df: pd.DataFrame):
    """Blue gradient by cell value — implemented by hand so we don't pull in
    matplotlib just for a heatmap."""
    mx = int(df.values.max()) or 1

    def color(val):
        if val <= 0:
            return ""
        a = 0.12 + 0.68 * (val / mx)
        return f"background-color: rgba(31,119,180,{a:.3f})"

    return df.style.map(color).format("{:d}")


def render_scored(corpus_tag: str, a: str, b: str):
    g = gold()
    ra, rb = load_result(corpus_tag, a), load_result(corpus_tag, b)
    if not (ra and rb):
        st.warning("Both models need a result file for this corpus.")
        return
    sca, scb = scoring.score_against_gold(ra, g), scoring.score_against_gold(rb, g)

    st.subheader("Scored view — against ground truth")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{display_name(a)} accuracy", f"{sca['accuracy']:.1%}")
    c2.metric(f"{display_name(b)} accuracy", f"{scb['accuracy']:.1%}",
              delta=f"{(scb['accuracy']-sca['accuracy'])*100:+.1f} pts")
    c3.metric(f"{display_name(a)} macro-F1", f"{sca['macro_f1']:.3f}")
    c4.metric(f"{display_name(b)} macro-F1", f"{scb['macro_f1']:.3f}",
              delta=f"{scb['macro_f1']-sca['macro_f1']:+.3f}")
    st.caption(f"Scored on {sca['n_classified']}/{sca['n_gold']} gold issues "
               f"(A unclassified: {sca['n_unclassified']}, B unclassified: {scb['n_unclassified']}). "
               "Macro-F1 weights every class equally, so it exposes rare-class weakness "
               "that accuracy (dominated by bug/enhancement) hides.")

    st.markdown("#### Per-class precision / recall / F1")
    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown(f"**{display_name(a)}**")
        st.dataframe(per_class_table(sca).style.format(
            {"precision": "{:.2f}", "recall": "{:.2f}", "f1": "{:.2f}"}),
            width="stretch", hide_index=True)
    with pc2:
        st.markdown(f"**{display_name(b)}**")
        st.dataframe(per_class_table(scb).style.format(
            {"precision": "{:.2f}", "recall": "{:.2f}", "f1": "{:.2f}"}),
            width="stretch", hide_index=True)
    st.caption("⚠️ security (support 6) and other (support 3) are statistically thin — "
               "one miss swings their F1 a lot. Treat as directional.")

    st.markdown("#### Confusion matrices (rows = ground truth, cols = prediction)")
    cm1, cm2 = st.columns(2)
    with cm1:
        st.markdown(f"**{display_name(a)}**")
        st.dataframe(style_confusion(confusion_df(sca)), width="stretch")
    with cm2:
        st.markdown(f"**{display_name(b)}**")
        st.dataframe(style_confusion(confusion_df(scb)), width="stretch")

    st.markdown("#### Drill-down: issues where the models disagree (with ground truth)")
    ax, bx = scoring.records_by_number(ra), scoring.records_by_number(rb)
    corpus = corpus_index()
    rows = []
    for num, gl in g.items():
        if num not in ax or num not in bx:
            continue
        la, lb = ax[num]["label"], bx[num]["label"]
        if la != lb:
            rows.append({
                "#": num, "ground truth": gl,
                f"{display_name(a)}": la, f"{display_name(b)}": lb,
                f"A✓": "✓" if la == gl else "", f"B✓": "✓" if lb == gl else "",
                "title": corpus.get(num, {}).get("title", "")[:80],
            })
    st.caption(f"{len(rows)} of {sca['n_classified']} scored issues are model disagreements.")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# --- unscored view ---------------------------------------------------------
def render_unscored(corpus_tag: str, a: str, b: str):
    ra, rb = load_result(corpus_tag, a), load_result(corpus_tag, b)
    if not (ra and rb):
        st.warning("Both models need a result file for this corpus.")
        return
    cmp = scoring.compare_unscored(ra, rb)

    st.subheader("Unscored view — full corpus, no ground truth")
    h1, h2, h3 = st.columns(3)
    h1.metric("Agreement rate", f"{cmp['agreement_rate']:.1%}",
              help="Share of jointly-classified issues where A and B chose the same label.")
    h2.metric("Issues compared", cmp["n_compared"])
    h3.metric("Disagreements", cmp["n_disagree"])

    st.markdown("#### Per-class distribution of suggestions")
    dist = pd.DataFrame({display_name(a): cmp["dist_a"], display_name(b): cmp["dist_b"]}).reindex(config.LABELS)
    st.bar_chart(dist)

    st.markdown("#### Disagreements")
    corpus = corpus_index()
    only_dis = st.checkbox("Show only disagreements", value=True)
    ax, bx = scoring.records_by_number(ra), scoring.records_by_number(rb)
    rows = []
    nums = [d["number"] for d in cmp["disagreements"]] if only_dis else sorted(set(ax) & set(bx))
    for num in nums:
        rows.append({
            "#": num,
            f"{display_name(a)}": ax[num]["label"],
            f"{display_name(b)}": bx[num]["label"],
            f"{display_name(a)} conf": ax[num].get("confidence"),
            f"{display_name(b)} conf": bx[num].get("confidence"),
            "title": corpus.get(num, {}).get("title", "")[:80],
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=320)

    st.markdown("#### Raw model output for an issue")
    pick = st.number_input("Issue number", min_value=0,
                           value=(rows[0]["#"] if rows else 0), step=1)
    if pick in ax and pick in bx:
        issue = corpus.get(pick, {})
        st.markdown(f"**#{pick} — {issue.get('title','')}**  ·  [open on GitHub]({issue.get('url','')})")
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown(f"**{display_name(a)}** → `{ax[pick]['label']}` (conf {ax[pick].get('confidence')})")
            st.code(ax[pick].get("raw", "") or "(empty)", language="json")
        with rc2:
            st.markdown(f"**{display_name(b)}** → `{bx[pick]['label']}` (conf {bx[pick].get('confidence')})")
            st.code(bx[pick].get("raw", "") or "(empty)", language="json")
        with st.expander("Issue body"):
            st.text((issue.get("body") or "(no body)")[:4000])


# --- operational metrics ---------------------------------------------------
def render_ops(corpus_tag: str, a: str, b: str):
    ra, rb = load_result(corpus_tag, a), load_result(corpus_tag, b)
    if not (ra and rb):
        st.warning("Both models need a result file for this corpus.")
        return
    st.subheader("Operational metrics (per run)")
    ma, mb = ra["meta"], rb["meta"]

    def block(name, m):
        st.markdown(f"**{display_name(name)}**  · concurrency={m['concurrency']} · prompt={m['prompt_version']}")
        c = st.columns(3)
        c[0].metric("Total cost", f"${m['total_cost_usd']:.4f}")
        c[1].metric("Cost / call", f"${m['cost_per_call_usd']*1000:.4f} /1k")
        c[2].metric("Wall-clock", f"{m['wall_clock_s']:.1f}s")
        c = st.columns(3)
        c[0].metric(f"p50 latency @c={m['concurrency']}", f"{m['latency_p50_s']:.2f}s")
        c[1].metric(f"p95 latency @c={m['concurrency']}", f"{m['latency_p95_s']:.2f}s")
        c[2].metric("Throughput", f"{m['throughput_rps']:.1f} rps")
        c = st.columns(3)
        c[0].metric("Tokens in / out", f"{m['tokens_in_total']:,} / {m['tokens_out_total']:,}")
        c[1].metric("Error rate", f"{m['error_rate']:.1%}")
        c[2].metric("Failed calls", m["n_failed"])
        if m["error_breakdown"]:
            st.caption("Errors by type: " + ", ".join(f"{k}={v}" for k, v in m["error_breakdown"].items()))
        else:
            st.caption("No errors.")
        st.caption(f"Pricing used: input ${m['pricing']['input']}/1M, output ${m['pricing']['output']}/1M "
                   "→ cost = prompt_tokens·in_rate + completion_tokens·out_rate (see harness/config.py).")

    col1, col2 = st.columns(2)
    with col1:
        block(a, ma)
    with col2:
        block(b, mb)


# --- run / methodology -----------------------------------------------------
def render_run():
    st.subheader("Run a classification")
    st.caption("Shells out to `harness/runner.py`. Concurrency is a runtime arg "
               "(also settable via the CONCURRENCY env var) — never baked into the image.")
    with st.form("run"):
        c1, c2, c3 = st.columns(3)
        model = c1.selectbox("Model", [m.id for m in config.CANDIDATES],
                             format_func=display_name)
        corpus = c2.selectbox("Corpus", ["gold", "full"])
        conc = c3.number_input("Concurrency", 1, 64, config.DEFAULT_CONCURRENCY)
        go = st.form_submit_button("Run")
    if go:
        cmd = [sys.executable, str(Path(config.ROOT) / "harness" / "runner.py"),
               "--model", model, "--corpus", corpus, "--concurrency", str(conc)]
        with st.spinner(f"Running {display_name(model)} on {corpus} @ c={conc} ..."):
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=config.ROOT)
        st.code((proc.stdout or "") + (proc.stderr or ""))
        load_result_cached.clear()
        st.success("Done. Switch tabs to see updated results.")

    st.markdown("---")
    st.subheader("Ground-truth methodology")
    st.markdown(
        "- **142-issue gold set**, stratified to give rare classes usable support.\n"
        "- Ground truth = **triangulation**: maintainer label (human) + two independent "
        "frontier labelers (Opus 4.8 + GPT-5.4); 15 contested cases hand-adjudicated.\n"
        "- **Maintainer labels are ~8.5% noisy** vs adjudicated gold (e.g. a token-to-stdout "
        "issue labeled `bug` is really `security`).\n"
        "- **`security` is low-diversity**: 26/27 maintainer security issues were automated "
        "CVE-scanner templates; downsampled to 5 + 1 human case. Open question for the customer: "
        "are bot CVE reports `security` or `other`?\n"
        "- Full corpus = 530 issues; gold ⊂ full. The unscored view uses the full corpus."
    )


# --- app shell -------------------------------------------------------------
def main():
    st.title("doctl issue classification — model evaluation")
    st.markdown(
        "**Recommendation:** run **gpt-oss-120b** (open-weight, ~37× cheaper *per correct "
        "classification* than the Claude Opus 4.8 baseline) as the primary classifier, and "
        "escalate to **Claude Haiku 4.5** on a **disagreement** signal — not on confidence "
        "(cheap models are confidently wrong). gpt-oss-120b and Haiku disagree on ~11% of "
        "issues, and that 11% holds ~71% of the errors. Always escalate `security`/`other` "
        "to a human. The leaderboard below is the evidence."
    )

    with st.sidebar:
        st.header("Compare")
        corpus_choices = [c for c in ("gold", "full") if available_models(c)]
        if not corpus_choices:
            st.error("No result files. Run a sweep first.")
            corpus_tag = "gold"
        else:
            corpus_tag = st.radio("Corpus", corpus_choices,
                                  help="gold = 142 labeled issues (scored); full = all 530 (unscored).")
        models = available_models(corpus_tag)
        defaults = [m for m in RECOMMENDED if m in models] or models[:2]
        a = st.selectbox("Model A", models, index=models.index(defaults[0]) if defaults else 0,
                         format_func=display_name) if models else None
        b_default = defaults[1] if len(defaults) > 1 else (models[-1] if models else None)
        b = st.selectbox("Model B", models,
                         index=models.index(b_default) if b_default in models else 0,
                         format_func=display_name) if models else None

    tabs = st.tabs(["🏆 Leaderboard", "🎯 Scored (A vs B)", "🔀 Unscored (A vs B)",
                    "⚙️ Operational", "📋 Run / methodology"])
    with tabs[0]:
        render_leaderboard("gold" if available_models("gold") else corpus_tag)
    with tabs[1]:
        if a and b:
            render_scored(corpus_tag, a, b)
    with tabs[2]:
        if a and b:
            render_unscored(corpus_tag, a, b)
    with tabs[3]:
        if a and b:
            render_ops(corpus_tag, a, b)
    with tabs[4]:
        render_run()


if __name__ == "__main__":
    main()
