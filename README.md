# RAGNAROK: RAG Numeric Analytics, Root-cause & Outcome Kit

***Where bad RAG pipelines meet their Ragnarök***

<br>


**RAGNAROK is a сonfigurable modular LLM-Independent RAG Evaluation Framework**

RAGNAROK evaluates Retrieval-Augmented Generation (RAG) pipelines end-to-end — retrieval, generation,
and business impact — and reduces the result to a single **Business Score (0–100)**, while automatically
diagnosing **why** a RAG answer is bad and **at which pipeline stage** the problem sits. No LLM call is
required for the core metrics (an LLM judge is available as an optional add-on).

> Quick setup and usage examples: see [QuickStart.md](QuickStart.md).

---

## What is RAGNAROK?

- A Python framework that scores a single `(query, answer, contexts)` triple, or a batch of them.
- Combines **retrieval quality**, **generation quality**, and **business impact** (cost, latency, trust)
  into one configurable score.
- Goes beyond scoring: it runs **root-cause diagnostics**, telling you whether a bad answer stems from
  a chunking problem, a bad embedding model, a weak reranker, a hallucinating generator, or a business
  constraint (too slow / too expensive).
- Ships with pytest coverage, YAML-driven configuration, ready-made scenario presets, and both
  no-LLM and LLM-based reporting.

## Key Features & Benefits

- **LLM-independent core** — faithfulness, coverage, and density metrics run on local embedding/NLI
  models; no API key required for the base pipeline.
- **Single Business Score (0–100)** — a weighted, guardrail-aware aggregation of every metric you enable,
  tunable per use case (safety-critical vs. cost-sensitive vs. speed-critical).
- **5-stage retrieval pipeline diagnostics** with automatic **bottleneck detection** — the framework tells
  you *which* stage is failing instead of just "retrieval is bad".
- **Automatic fault classification** — every evaluation returns a fault type/subtype, a confidence score,
  a plain-language explanation, and a recommended fix.
- **Flexible business metrics** — token cost, self-hosted GPU-hour cost (with automatic GPU-hour
  estimation from token counts), latency grading, trust scoring, and a composite business score with
  user-defined dimension weights (speed/cost/accuracy/coverage).
- **Ready-made configuration presets** (safety-first, speed-first, balanced, cost-saving, medical,
  customer support, research) plus free-text and numeric-priority preset selection — no manual YAML
  editing required.
- **Two reporting modes** — a deterministic, no-API-key structured report (Markdown/HTML/dict), and an
  optional LLM-generated narrative report.
- **Optional integrations** — RAGAS metrics and LLM-as-a-judge (OpenAI or GigaChat), both fully optional
  and disabled by default.
- **Modular, typed, tested** — every metric is its own class with its own pytest suite; the orchestrator
  (`RAGEvaluator`) composes them.

## What You Can Do

- **Evaluate a single answer**: `evaluator.evaluate(query, answer, contexts, ...)` → Business Score,
  status, fault diagnosis, and every underlying metric.
- **Evaluate a batch**: `evaluator.evaluate_batch(examples)`.
- **Diagnose the retrieval pipeline**: get a stage-by-stage pass/fail report with a named bottleneck
  stage, automatically included whenever `contexts` are provided.
- **Run individual metric stages standalone**: `evaluator.chunking`, `evaluator.embedding_quality`,
  `evaluator.reranking_quality`, `evaluator.pipeline`, `evaluator.data_availability` — each callable on
  its own with richer inputs (paraphrases, semantic clusters, before/after-rerank scores, corpus).
- **Pick a scenario preset** by name, by free-text description, or by numeric priorities — or write your
  own YAML config from scratch.
- **Estimate self-hosted inference cost** from token counts and GPU type, without knowing GPU-hours
  up front.
- **Set your own priorities** for the composite business score (e.g. "accuracy matters most to me").
- **Generate a report** in Markdown, HTML, or dict form — with or without an LLM.
- **Turn on RAGAS or an LLM judge** (OpenAI/GigaChat) as additional, optional scoring signals.
- **Run the included 30-case synthetic test suite** to see the fault classifier's accuracy on known
  scenarios (hallucination, omission, redundancy, retrieval failure, multilingual pairs, etc.).

---

## How Evaluation Works

Evaluation runs in four conceptual phases, in this order:

### 1. Retrieval pipeline diagnostics (5 stages, with bottleneck detection)

Each stage is a self-contained metric; `RetrievalPipelineEvaluator` runs all five and reports the
**first failing stage** as the bottleneck:

- **Stage 0 — Data Availability**: is there any relevant data in the vector store at all? (empty-result
  rate, coverage gap, index freshness, TF-IDF-based missing-document detection)
- **Stage 1 — Chunking Quality**: are documents split sensibly? (boundary coherence, chunk-size
  variance, informative-token ratio, semantic cohesion within a chunk)
- **Stage 2 — Embedding Quality**: is the embedding model behaving well? (query-doc similarity spread,
  cluster compactness, intra-query stability across paraphrases, cross-encoder calibration)
- **Stage 3 — Retrieval Quality**: classic IR metrics — Hit Rate@k, Precision@k, Recall@k, NDCG@k, MRR@k.
- **Stage 4 — Reranking Quality**: did the reranker help? (MRR lift before/after, relevance-order
  accuracy, first-relevant-position shift)

### 2. Generation quality

- **Faithfulness** — NLI-entailment between the answer and the retrieved context (with a semantic-
  similarity fallback and multilingual-pair handling); detects hallucination and contradiction.
- **SFC (Semantic Footprint Coverage)** — clusters context claims and checks how much of that
  information the answer actually covers; detects omission.
- **IDS (Information Density Score)** — penalizes redundant/verbose answers relative to their novel
  content.

### 3. Business metrics

- **Cost** — token-based cost score (configurable pricing) plus optional GPU-hour cost, either supplied
  directly or estimated from token counts and GPU type.
- **Latency** — graded against a configurable target/max SLA.
- **Trust** — combines faithfulness, citation presence/accuracy, and consistency into a 0–1 trust score.
- **Coverage Efficiency** — how much of the retrieved context was actually useful (waste ratio).
- **User Satisfaction Proxy** — a weighted proxy for user-perceived quality (faithfulness, relevance,
  completeness, conciseness).
- **Composite Business Score** — the same four business signals (speed/cost/accuracy/coverage) combined
  with *your* weights, plus the single weakest dimension named explicitly.

### 4. Aggregation and diagnosis

- All enabled metrics are normalized, weight-averaged, and reduced through any configured **guardrails**
  (hard floors that zero out the score if violated), a **latency penalty**, and optional **interaction
  modifiers** (bonus/penalty rules combining multiple metrics) — producing the final **Business Score
  (0–100)** and a **PASSED / FAILED / GUARDRAIL_VIOLATION** status.
- The **Fault Classifier** independently inspects the same metrics and returns a fault type
  (`healthy`, `retrieval`, `chunking`, `generation`, `business`, `out_of_scope`, `unknown`), a subtype,
  a confidence score, a plain-language explanation, and a recommended fix.
- If contexts were provided, the Stage 0–4 pipeline report from phase 1 is attached alongside, so a low
  score can be traced back to a specific stage.

The full result dict (`evaluator.evaluate(...)`) carries every intermediate metric, the business
metrics, the fault diagnosis, and the retrieval pipeline report — nothing is hidden behind the final
score.

---

## Configuration

- **YAML config** (`config.yaml`) — per-metric weight/bounds/critical-threshold, guardrails, latency
  penalty/threshold, interaction modifiers, cost config, fault thresholds.
- **Presets** (`ragnarok/presets.py`) — `safety_first`, `speed_first`, `balanced`, `cost_saving`,
  `medical`, `customer_support`, `research`; selectable by name, by free-text description
  (`suggest_preset(...)`), or by numeric priorities (`create_custom_preset(...)`).
- **Business dimension weights** — pass `business_dimension_weights=BusinessDimensionWeights(...)` to
  `RAGEvaluator` to control the composite business score independently of the main metric weights.

## Reports

- **Structured report** (default, no API key) — `evaluator.generate_report(result)` →
  `.to_markdown()` / `.to_html()` / `.to_dict()`, with 7 sections and severity levels.
- **LLM report** (optional) — `evaluator.generate_report(result, use_llm=True)`, reusing the configured
  OpenAI/GigaChat provider.

## Optional Integrations

- **RAGAS** (`ragas_enabled: true` in config) — adds RAGAS's own metrics to the result.
- **LLM Judge** (`llm_judge_enabled: true`) — OpenAI or GigaChat rubric-based scoring (faithfulness,
  relevance, completeness, coherence, conciseness) as an additional signal.

---

## Project Structure

```
RAGNAROK/
├── ragnarok/                     # The package
│   ├── evaluator.py              # RAGEvaluator — main orchestrator
│   ├── config.py                 # EvaluationConfig, scoring aggregation
│   ├── presets.py                # Scenario presets + preset selection
│   ├── reporting.py              # Structured (no-LLM) and LLM report generators
│   ├── metrics/                  # One class per retrieval-pipeline stage + generation metrics
│   ├── business/                 # Business metrics + fault classifier
│   └── integrations/             # Optional LLM judge / RAGAS bridges
├── tests/                        # pytest suite, one file per module
├── config.yaml                   # Example YAML configuration
├── main.py                       # Runnable demo
├── synthetic_test_cases.py       # 30 labeled synthetic evaluation cases
└── run_synthetic.py              # Runs the synthetic case suite
```

## License

See [LICENSE](LICENSE).
