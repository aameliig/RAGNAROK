# QuickStart

Get RAGNAROK installed, verified, and running in a few minutes. For the full feature overview, see
[README.md](README.md).

## 1. Prerequisites

- Python >= 3.9
- ~500MB free disk space for the first model download (`all-MiniLM-L6-v2` and
  `cross-encoder/nli-deberta-v3-large`, downloaded automatically on first run)

## 2. Install

```bash
# Core only
pip install -e .

# + OpenAI LLM judge support
pip install -e ".[openai]"

# + RAGAS integration
pip install -e ".[ragas]"

# + dev tools (pytest)
pip install -e ".[dev]"

# everything
pip install -e ".[all]"
```

Core dependencies: `sentence-transformers`, `transformers`, `torch`, `scikit-learn`, `numpy`, `nltk`,
`pyyaml`, `requests`.

## 3. Verify it works

Run the demo (evaluates one example end-to-end and prints a formatted report):

```bash
python main.py
# equivalent: python -m ragnarok
```

Run the automated test suite (120+ tests, one file per module):

```bash
pip install -e ".[dev]"
pytest
```

Optionally, run the 30-case synthetic evaluation suite (checks fault-diagnosis accuracy against
labeled scenarios):

```bash
python run_synthetic.py
```

> First run downloads the embedding and NLI models — this can take a few minutes. Subsequent runs are fast.

## 4. Quick Usage Examples

### Basic evaluation

```python
from ragnarok import RAGEvaluator, EvaluationConfig

config = EvaluationConfig.from_yaml("config.yaml")
evaluator = RAGEvaluator(config)

result = evaluator.evaluate(
    query="When was the Eiffel Tower built?",
    answer="The Eiffel Tower was built in 1889.",
    contexts=["The Eiffel Tower was built in 1889.", "It is located in Paris."],
    relevance_scores=[1, 0],
    latency_ms=150,
    input_tokens=10,
    output_tokens=8,
    num_chunks_retrieved=2,
)

evaluator.print_results(result)
print(result["final_score"], result["status"], result["fault_type"])
```

### Using a built-in preset

```python
from ragnarok import RAGEvaluator

evaluator = RAGEvaluator(preset="medical")   # safety_first, speed_first, balanced,
                                              # cost_saving, medical, customer_support, research
```

### Picking a preset from free text or numeric priorities

```python
from ragnarok import RAGEvaluator, suggest_preset, create_custom_preset

preset_name = suggest_preset("I don't want the system to hallucinate, even if it's slower")
evaluator = RAGEvaluator(preset=preset_name)

# or specify priorities directly (speed/cost/accuracy/coverage)
config = create_custom_preset(speed=0.1, cost=0.1, accuracy=0.6, coverage=0.2)
evaluator = RAGEvaluator(config)
```

### Batch evaluation

```python
examples = [
    {"query": "Q1", "answer": "A1", "contexts": ["C1"], "relevance_scores": [1]},
    {"query": "Q2", "answer": "A2", "contexts": ["C2"], "relevance_scores": [1]},
]
results = evaluator.evaluate_batch(examples)
```

### Retrieval pipeline bottleneck diagnosis

Included automatically whenever `contexts` are provided:

```python
result = evaluator.evaluate(query=..., answer=..., contexts=[...], relevance_scores=[...])
report = result["retrieval_pipeline_report"]
print(report["bottleneck_stage"], report["bottleneck_explanation"])
```

### Reports

```python
report = evaluator.generate_report(result)              # structured, no LLM/API key needed
print(report.to_markdown())                              # or .to_html(), .to_dict()

narrative = evaluator.generate_report(result, use_llm=True)  # optional LLM narrative
```

### Self-hosted GPU cost estimate

```python
result = evaluator.evaluate(
    query=..., answer=..., contexts=...,
    input_tokens=1000, output_tokens=500,
    gpu_type="A100",   # gpu_hours and cost are estimated automatically
)
print(result["business_metrics"]["gpu_efficiency_score"])
```

## 5. Configuration Options

| Option | Where | Purpose |
|---|---|---|
| `metrics` | YAML / `EvaluationConfig` | Per-metric weight, bounds, critical threshold, enabled flag |
| `guardrails` | YAML / `EvaluationConfig` | Hard floor on a metric — violating it zeroes the score |
| `latency_penalty` / `latency_threshold_ms` | YAML / `EvaluationConfig` | Penalty applied above a latency threshold |
| `interaction_modifiers` / `bonus_penalty` | YAML / `EvaluationConfig` | Bonus/penalty rules combining multiple metrics |
| `cost_config` | YAML / `EvaluationConfig` | Custom token pricing and/or GPU-hour pricing |
| `business_dimensions` | YAML / `EvaluationConfig` | Retrieval/generation/business weighting (informational) |
| `fault_thresholds` | YAML / `EvaluationConfig` | Thresholds used by the fault classifier |
| `ragas_enabled` / `llm_judge_enabled` | YAML / `EvaluationConfig` | Toggle optional RAGAS / LLM-judge scoring |
| `preset` | `RAGEvaluator(preset=...)` | Load a ready-made scenario config by name |
| `business_dimension_weights` | `RAGEvaluator(business_dimension_weights=...)` | Weights for the composite business score (speed/cost/accuracy/coverage) |

Optional LLM judge providers need credentials:

```bash
export OPENAI_API_KEY="sk-..."
# or
export GIGACHAT_CLIENT_ID="..."
export GIGACHAT_CLIENT_SECRET="..."
```

Then in config: `llm_judge_enabled: true`, `llm_judge_provider: "openai"` (or `"gigachat"`).

## Next Steps

- Full feature list and evaluation-logic explanation: [README.md](README.md)
- Example config: [config.yaml](config.yaml)
- All presets: `ragnarok.presets.list_presets()`
