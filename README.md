# doctl Issue Classification — Model Evaluation Harness

An eval harness that classifies GitHub issues from the [DigitalOcean `doctl`](https://github.com/digitalocean/doctl)
repository into a six-label schema (`bug`, `enhancement`, `question`, `documentation`,
`security`, `other`), compares language models on that workload, and produces a
production recommendation backed by accuracy, cost, latency, and throughput numbers.

All generation runs against the **DigitalOcean Serverless Inference** API
(OpenAI-compatible). Each issue is its own inference request — no batching — so every
call has its own cost, latency, retry, and fallback path.

> **Running application:** _<ADD YOUR DEPLOYED URL HERE>_

---

## TL;DR recommendation

A customer running this workload against **Claude Opus 4.8** (a frontier model) is
overpaying by roughly **30–50×** for accuracy they don't need.

- **Primary: `gpt-oss-120b`** (open-weight). ~88% accuracy at **~$0.0002 per correct
  classification** — about **37× cheaper per correct answer** than Opus 4.8, with
  competitive macro-F1.
- **Escalation: `Claude Haiku 4.5`**. ~93% accuracy at ~$0.0011 per correct — used as a
  fallback for low-confidence predictions and the high-stakes `security` / `other`
  classes, not for the whole corpus.

The operational pattern is a **cascade**: classify everything with the cheap model, and
route only the uncertain or sensitive cases to the stronger one (or a human). See
[Production handling](#production-handling).

These two were chosen out of an **11-model sweep** (see [Model selection](#model-selection));
the leaderboard in the app is the evidence.

---

## Architecture

The harness is split into a **runner** (does inference, writes results) and a **viewer**
(reads persisted results, renders the dashboard). Running ~500 issues × many models is
slow and costs money, so we never re-run inference on a page load — the viewer only reads
reproducible, persisted artifacts.

```
harness/
  config.py      # paths, env, label schema, pricing loader, model registry, cost math
  ingest.py      # GitHub API -> data/corpus.jsonl (stable snapshot, PRs filtered)
  build_gold.py  # construct the 142-issue ground-truth gold set (triangulated)
  classify.py    # prompt construction + robust response parsing
  client.py      # ONE inference call: retries, error categorization, latency/token capture
  runner.py      # concurrency, metric aggregation, persistence
  scoring.py     # accuracy, per-class P/R/F1, confusion matrix, agreement (hand-computed)
  ui.py          # Streamlit dashboard
data/
  corpus.jsonl   # 530 issues (stable corpus)
  gold.jsonl     # 142 ground-truth labels (the eval corpus)
  pricing.json   # per-token rates -> every $ figure is traceable to a rate
results/
  gold__<model>.json   # per-model run: meta (ops metrics) + per-issue records
  full__<model>.json
prompts/
  classify_v1.txt      # the classification prompt (versioned)
```

---

## Quick start (Docker)

```bash
# 1. Build
docker build -t doctl-eval .

# 2. Run (provide your SI key; optionally tune concurrency without rebuilding)
docker run --rm -p 8501:8501 \
  -e SI_API_KEY=doo_v1_your_key_here \
  -e CONCURRENCY=8 \
  doctl-eval

# 3. Open the dashboard
open http://localhost:8501
```

The image ships with the stable corpus, the gold set, and the persisted sweep results,
so the dashboard is populated immediately. To re-run inference yourself, use the
**Run / methodology** tab or the CLI below.

### Deploying (for the hosted link)

The container is self-contained and stateless, so any container host works. On
**DigitalOcean App Platform**: push this repo, create an App from the Dockerfile, set
`SI_API_KEY` as an encrypted env var, and set the HTTP port to `8501`. Put the resulting
URL at the top of this README. (Locally, `docker run` as above is enough to review it.)

### Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `SI_API_KEY` | **yes** | — | DigitalOcean Serverless Inference model access key (`doo_v1_...`). |
| `SI_BASE_URL` | no | `https://inference.do-ai.run/v1` | OpenAI-compatible endpoint. |
| `CONCURRENCY` | no | `8` | Parallel inference requests. Runtime arg — no rebuild needed. |
| `GITHUB_TOKEN` | no | — | Higher rate limits during ingestion only. |

---

## Running from source (without Docker)

```bash
python -m venv .venv && . .venv/Scripts/activate    # Windows; use bin/activate on *nix
pip install -r requirements.txt
cp .env.example .env        # then paste your SI_API_KEY

# (corpus + gold are already committed; these regenerate them)
python harness/ingest.py                                   # refresh corpus snapshot
python harness/build_gold.py                               # rebuild gold set

# Run the eval
python harness/runner.py --all-candidates --corpus gold --concurrency 8   # the sweep
python harness/runner.py --model openai-gpt-oss-120b --corpus full         # full corpus

# Launch the dashboard
streamlit run harness/ui.py
```

---

## What the dashboard shows

- **🏆 Leaderboard** — every candidate scored on the gold set: accuracy, macro-F1,
  cost-per-correct, p95, throughput. This is the broader evaluation behind the two-model
  recommendation.
- **🎯 Scored (A vs B)** — accuracy + per-class precision/recall/F1, side-by-side
  confusion matrices, and a drill-down of issues where the two models disagree with the
  ground-truth label visible.
- **🔀 Unscored (A vs B)** — full-corpus suggestions with raw model output, the headline
  agreement rate, per-class suggestion distributions, and a disagreement filter.
- **⚙️ Operational** — per-run cost (total + per-call), p50/p95 latency with the
  concurrency they were measured at, wall-clock, throughput, and error rate by type.
- **📋 Run / methodology** — trigger a run at any concurrency; read the ground-truth method.

---

## Methodology

### Data source & stable corpus
`ingest.py` pulls all open and closed issues via the GitHub public API, filtering out
pull requests (the issues endpoint returns them too). GitHub caps REST pagination at
~1000 items, so the fetcher windows forward with a `since` cursor and dedups by issue
number. The result is snapshotted to `data/corpus.jsonl` (**530 issues**) and committed,
so the corpus is identical across runs.

### Ground truth (the key methodology choice)
Maintainer labels are noisy and inconsistent — only a handful are category labels
(`bug`, `suggestion`, `security vulnerability`, `question`, `docs`), the rest are
process/area labels (`hacktoberfest`, `good first issue`, `snap`, …) that say nothing
about the category. So ground truth is a **triangulation**, not a copy of maintainer labels:

1. **Stratified sample** of 160 issues (fixed seed). Rare classes are oversampled so they
   have usable support instead of 2–3 examples from a naive random draw.
2. Each issue gets: the **maintainer label** (human, where present, mapped to schema) plus
   **two independent frontier proposals** (Claude Opus 4.8 + GPT-5.4).
3. **Auto-resolve** where human + both models agree (110 unanimous) or there's a clear
   majority; **hand-adjudicate** the 15 contested cases; add 3 hand-found `other` examples.

This yields **142 gold issues**. Two honest findings fell out of it:

- **Maintainer labels are ~8.5% wrong** vs adjudicated gold. Example: an issue reporting
  that `doctl auth init` prints the API token to stdout was labeled `bug` by maintainers —
  it's really `security`.
- **`security` is a trap.** 26 of 27 maintainer `security vulnerability` issues are
  automated CVE-scanner templates (`CVE-XXXX detected in github.com/docker/cli…`) for
  transitive dependencies; only one is a human-reported app-security issue. We downsampled
  the template to 5 representatives + the 1 human case, so `security` F1 measures real
  judgment, not template-matching. **Open question for the customer:** should bot CVE
  reports be `security` (schema-faithful) or `other` (automated noise)? It changes routing.

Because the gold labelers are Opus 4.8 and GPT-5.4, *their* scores are partly definitional
and shown only as an upper anchor — the production recommendation rests on the **cheap**
models, none of which graded their own homework.

### Scoring
Metrics are computed by hand in `scoring.py` (not sklearn) so every number is auditable.
We report **accuracy** (dominated by the common classes) and **macro-F1** (every class
weighted equally, which exposes rare-class weakness), per-class precision/recall/F1 with
support, and confusion matrices. Failed calls are counted separately as `n_unclassified`
so a model can't hide misses by erroring out.

### Cost
`config.call_cost()` is the single place cost is computed:
`prompt_tokens × input_rate + completion_tokens × output_rate`, with rates from
`data/pricing.json` (captured from DO's published pricing). Token counts come from the
API's `usage` field. **Cost-per-correct-classification** = gold-subset cost ÷ correct
answers — the metric that actually matters to the customer.

---

## Model selection

Evaluated **11 candidates** spanning the tradeoff axes (open-weight vs frontier, small vs
large, reasoning vs non-reasoning):

`Claude Opus 4.8` (baseline), `GPT-5.4`, `Claude Haiku 4.5`, `GPT-4o mini`,
`gpt-oss-120b`, `gpt-oss-20b`, `Qwen3-32B`, `Gemma 4 31B`, `Llama 3.3 70B`,
`Ministral 3 14B`, `o3-mini` (reasoning).

Gold-set results (142 issues), best to worst by accuracy:

| Model | Type | Accuracy | Macro-F1 | $/correct | p95 (s) | rps |
|---|---|---:|---:|---:|---:|---:|
| Claude Opus 4.8 † | frontier | 95.1% | 0.955 | $0.00713 | 2.92 | 4.0 |
| **Claude Haiku 4.5** | frontier | 93.0% | 0.927 | $0.00110 | 1.38 | 6.8 |
| GPT-5.4 † | frontier | 92.3% | 0.882 | $0.00245 | 1.57 | 6.0 |
| Gemma 4 31B | open-weight | 90.1% | 0.873 | $0.00018 | 2.74 | 5.5 |
| gpt-oss-20b | open-weight | 88.7% | 0.858 | $0.00013 | 3.29 | 4.7 |
| Ministral 3 14B | open-weight | 88.7% | 0.823 | $0.00018 | 0.89 | 12.7 |
| o3-mini | reasoning | 88.0% | 0.844 | $0.00242 | 3.96 | 3.4 |
| **gpt-oss-120b** | open-weight | 88.0% | 0.875 | $0.00019 | 2.12 | 6.1 |
| GPT-4o mini | frontier | 86.6% | 0.817 | $0.00015 | 1.30 | 8.6 |
| Qwen3-32B | open-weight | 85.9% | 0.839 | $0.00044 | 15.89 | 0.8 |
| Llama 3.3 70B | open-weight | 84.5% | 0.785 | $0.00059 | 4.13 | 2.5 |

† Opus 4.8 and GPT-5.4 co-defined the gold labels, so their scores are an upper anchor,
not fair contestants. **Bold** = recommended pair.

Things the sweep revealed:
- **Reasoning is overkill.** `o3-mini` matches `gpt-oss-120b`'s 88% accuracy at ~13× the
  cost and ~2× the latency — chain-of-thought buys nothing for single-label classification.
- **Latency disqualifies some.** `Qwen3-32B` posts a 15.9s p95 (0.8 rps) — unusable at volume.
- **`gpt-oss-120b` and `Gemma 4 31B` are statistically tied** at the cheap end (88–90%
  accuracy is within noise at n=142, same cost). I lead with `gpt-oss-120b` for better tail
  latency, throughput, and the best balanced macro-F1; Gemma is the equally-valid alternative.
- **All models score 1.00 on `security`** — but only because the gold `security` examples
  are mostly the CVE-bot template (see methodology). This is *not* evidence they'd catch a
  subtle human-reported vulnerability.

**Why these two:** `gpt-oss-120b` is the accuracy/cost knee — near-frontier-minus accuracy
at open-weight cost. `Haiku 4.5` is the cheapest model that closes most of the remaining
gap to the frontier, making it the right escalation target. They differ on the axes the
brief cares about (open-weight vs frontier-family; cost; capability), not minor variants.

**Why not a reasoning model:** for single-label classification, `o3-mini` adds latency and
output-token cost (reasoning tokens) for no accuracy win — included in the sweep to show it.

---

## Production handling

This is where the naive answer is wrong, and the data says so.

**Self-reported confidence does NOT reliably detect cheap-model errors.** `gpt-oss-120b`'s
*wrong* predictions have a median confidence of 0.96 — essentially the same as its correct
ones (0.97). A "route everything below 0.95" rule escalates 10% of the corpus but catches
only ~41% of the errors. Cheap models are confidently wrong. (Haiku is better calibrated:
~90% of its errors fall under a 0.9 threshold.)

**Model disagreement is the far better detector.** `gpt-oss-120b` and `Haiku 4.5` disagree
on only ~11% of issues, but that 11% contains **~71% of `gpt-oss-120b`'s errors**, and
escalating those disagreements to Haiku *fixes* most of them. Disagreement, not confidence,
is the routing signal for this workload.

So the recommended wrapper is a **disagreement-triggered cascade**, not a confidence gate:

1. Classify everything with **gpt-oss-120b** (the cheap primary).
2. Get a cheap second opinion (e.g., Gemma 4 31B, also ~$0.0002) and **escalate the ~11% of
   disagreements** to **Haiku 4.5**.
3. **Always escalate `security` and `other`** regardless — missing a real vulnerability is
   the expensive failure, and these classes are rare so it's cheap to always check.
4. **Malformed output is caught, not coerced** — `parse_response` recovers a label where it
   can and flags `parse_error` (→ review) where it can't; never a silent wrong label.
5. **Every call is independently retryable** — rate-limit/timeout/5xx retry with exponential
   backoff + jitter; `bad_request` fails fast. Errors are categorized and reported by type.

Net effect: roughly **Haiku-level accuracy (~93%) at a small fraction of frontier cost**,
because the expensive model only touches the ~11% the cheap models can't agree on.

**Caveat on `security`:** every model scored 1.00 on the gold `security` set, but that set
is dominated by an automated CVE template. Real human-reported security issues are rare and
may hide under `bug` (one did). I would *not* trust any of these models to be the only line
of defense on security — always route the `security` class to a human.

---

## What I cut, and what I'd do next

- **Cut:** multi-annotator agreement (single adjudicator); prompt-variant A/B testing;
  few-shot prompting; calibration curves for the confidence score; auth on the dashboard.
- **Next with more time:** validate confidence calibration (is conf<0.7 actually where
  errors concentrate?); hunt for security issues hiding under `bug` to measure true
  security recall; test prompt sensitivity; add a held-out test split distinct from the
  set the frontier models labeled, to remove the gold-labeler bias entirely.

---

## License / notes

Built for the DigitalOcean FDE evaluation exercise. The methodology is provider-agnostic —
point `SI_BASE_URL` at any OpenAI-compatible endpoint and update `data/pricing.json`.
