# RAGNAROK — LLM-Independent RAG Evaluation Framework

## Структура проекта

```
RAGNAROK/
├── ragnarok/                    # Пакет
│   ├── __init__.py              # Экспорты пакета
│   ├── __main__.py              # Демо: python -m ragnarok
│   ├── _libs.py                 # Базовые импорты
│   ├── config.py                # Конфигурация: метрики, веса, guardrails
│   ├── evaluator.py             # Главный оркестратор (RAGEvaluator)
│   ├── utils.py                 # Утилиты (measure_latency)
│   ├── presets.py                # Готовые конфиги-пресеты + подбор по описанию/приоритетам
│   ├── reporting.py               # Отчёты: без LLM (StructuredReport) и с LLM (нарратив)
│   ├── metrics/                  # Метрики этапов retrieval-пайплайна
│   │   ├── data_availability.py  # Этап 0: есть ли данные в базе
│   │   ├── chunking.py           # Этап 1: качество чанкинга
│   │   ├── embedding_quality.py  # Этап 2: качество эмбеддингов
│   │   ├── retrieval.py          # Этап 3: HitRate/Precision/Recall/NDCG/MRR
│   │   ├── reranking_quality.py  # Этап 4: эффективность реранкинга
│   │   ├── pipeline.py           # Все 5 этапов вместе + поиск bottleneck-этапа
│   │   ├── coverage.py           # SFC — Semantic Footprint Coverage
│   │   ├── density.py            # IDS — Information Density Score
│   │   └── faithfulness.py       # NLI-энтейлмент (faithfulness)
│   ├── business/                 # Бизнес-метрики и диагностика
│   │   ├── business_metrics.py   # Cost/GPU/Latency/Trust/Coverage Efficiency
│   │   └── fault_classifier.py   # Root-cause диагностика
│   └── integrations/             # Опциональные интеграции
│       ├── llm_judge.py          # LLM-as-a-Judge
│       └── ragas_bridge.py       # Интеграция с RAGAS
├── tests/                        # pytest-сюит (по одному файлу на модуль)
├── config.yaml                   # YAML-конфиг с весами метрик
├── main.py                       # Точка входа: python main.py
├── synthetic_test_cases.py       # 30 синтетических кейсов
├── run_synthetic.py              # Запуск 30 кейсов
├── pyproject.toml                # Зависимости с optional extras
├── requirements.txt              # Core-зависимости
└── requirements-dev.txt          # + pytest
```

## Начало работы

### Шаг 1: Установить зависимости

```bash
# Только core
pip install -e .

# С OpenAI Judge
pip install -e ".[openai]"

# С RAGAS
pip install -e ".[ragas]"

# Всё сразу
pip install -e ".[all]"
```

Core: sentence-transformers, transformers, torch, scikit-learn, numpy, nltk, pyyaml, requests.

Optional: openai>=1.0.0, ragas>=0.1.0, datasets>=2.14.0.

### Шаг 2: Запустить демо

```bash
python main.py
# то же самое: python -m ragnarok
```

**Ожидаемый вывод:**

```
============================================================
  RAGNAROK Demo Evaluation
============================================================

  📊 Business Score: 100.0/100  ✅ PASSED

  🔍 Fault: HEALTHY
     Confidence: 100%
     Explanation: All metrics within acceptable thresholds
     Action: No action needed

  📈 Core Metrics
     Faithfulness:   1.00
     SFC:            1.00
     IDS:            0.3389
     HitRate@3:      1.00
     NDCG@5:         0.9197

  💼 Business Metrics
     Cost Score:     100.0
     Latency Grade:  EXCELLENT
     Trust Score:    0.80
     Trust Level:    MEDIUM
     Coverage Eff.:  1.000
     Waste Ratio:    0.00
     User Sat. Proxy:0.62

  🧩 Chunking Quality: EXCELLENT (score: 0.98)

  🔌 Optional Modules
     RAGAS:          disabled
     LLM Judge:      disabled

============================================================
```

> При первом запуске скачаются модели (~500MB): all-MiniLM-L6-v2 (22MB) и cross-encoder/nli-deberta-v3-large (304MB). Занимает 2–5 минут.

### Шаг 2.5: Запустить полный набор тестов

```bash
pip install -e ".[dev]"   # или: pip install -r requirements-dev.txt
pytest
```

Тесты разбиты по модулям в `tests/` (`test_retrieval.py`, `test_coverage.py`, `test_chunking.py`, `test_data_availability.py`, `test_evaluator.py` и т.д.).

### Шаг 3: Запустить 30 синтетических кейсов

```bash
python run_synthetic.py
```

**Ожидаемый вывод (фрагмент):**

```
==============================================================
RAGNAROK Synthetic Test Suite — 30 Cases
==============================================================

--- Case 1/30: perfect_rag ---
Description: Идеальный RAG: полный, точный, релевантный ответ
  Expected fault: healthy
  Actual fault:   healthy ✅
  Business Score: 85.3
  Faithfulness:   1.00
  SFC:            0.50
  Trust Score:    0.76
  Cost Score:     100.0
  Latency Grade:  EXCELLENT

--- Case 2/30: hallucination_severe ---
Description: Серьёзная галлюцинация: полная противоположность фактам
  Expected fault: generation
  Actual fault:   generation ✅
  Business Score: 12.5
  Faithfulness:   0.00
  ...

==============================================================
SUMMARY
==============================================================

Total cases: 30
Correct diagnoses: 26/30 (86.7%)

Failed cases:
  ❌ multi_doc_contradiction: expected healthy, got generation
  ❌ conflicting_dates: expected healthy, got generation
  ...

Score distribution:
  PASSED (>=60): 19
  FAILED (<60):  11
```

> Некоторые кейсы с противоречивым контекстом могут давать `generation` вместо `healthy` — NLI-модель строгая. Для production настройте `threshold` в `faithfulness.py`.

---

## Использование в своём коде

### Базовый пример

```python
from ragnarok import RAGEvaluator, EvaluationConfig

config = EvaluationConfig.from_yaml("config.yaml")
evaluator = RAGEvaluator(config)

result = evaluator.evaluate(
    query="Когда построили Эйфелеву башню?",
    answer="Эйфелева башня была построена в 1889 году.",
    contexts=[
        "Эйфелева башня была построена в 1889 году.",
        "Её высота составляет 330 метров."
    ],
    relevance_scores=[1, 1],
    latency_ms=150,
    input_tokens=10,
    output_tokens=8,
    num_chunks_retrieved=2,
    has_citations=False
)

evaluator.print_results(result)
```

**Ожидаемый результат:**

```
============================================================
  RAG Evaluation Results
============================================================

  📊 Business Score: 85.3/100  ✅ PASSED

  🔍 Fault: HEALTHY
     Confidence: 100%
     Explanation: All metrics within acceptable thresholds
     Action: No action needed

  📈 Core Metrics
     Faithfulness:   1.00
     SFC:            0.50
     IDS:            0.0017
     HitRate@3:      1.00

  💼 Business Metrics
     Cost Score:     100.0
     Latency Grade:  EXCELLENT
     Trust Score:    0.76
     Trust Level:    HIGH
     Coverage Eff.:  0.250
     Waste Ratio:    0.50
     User Sat. Proxy:0.638

  🔌 Optional Modules
     RAGAS:          disabled
     LLM Judge:      disabled

============================================================
```

### Пакетная обработка

```python
examples = [
    {
        "query": "Вопрос 1",
        "answer": "Ответ 1",
        "contexts": ["Контекст 1"],
        "relevance_scores": [1],
        "latency_ms": 100,
        "input_tokens": 5,
        "output_tokens": 5,
        "num_chunks_retrieved": 1
    },
    {
        "query": "Вопрос 2",
        "answer": "Ответ 2",
        "contexts": ["Контекст 2"],
        "relevance_scores": [1],
        "latency_ms": 200,
        "input_tokens": 5,
        "output_tokens": 10,
        "num_chunks_retrieved": 1
    }
]

results = evaluator.evaluate_batch(examples)
for r in results:
    evaluator.print_results(r, title=f"Result: {r.get('fault_type', 'unknown')}")
```

---

## Метрики и формулы

### Retrieval Metrics

**Hit Rate @ k**

```
HR@k = 1  if Σ relevance[:k] > 0
       0  otherwise
```

**Precision @ k**

```
P@k = Σ relevance[:k] / k
```

**NDCG @ k**

```
DCG@k = Σ (rel_i / log₂(i + 2))   for i = 0..k-1
IDCG@k = DCG@k от идеального ранжирования
NDCG@k = DCG@k / IDCG@k
```

**MRR @ k**

```
MRR@k = 1 / (rank_first_relevant)   if relevant doc exists in top-k
        0                           otherwise
```

### Semantic Footprint Coverage (SFC)

Контекст разбивается на claims, кластеризуется (DBSCAN, cosine distance), затем проверяется покрытие кластеров ответом:

```
SFC = Σ (weight_i × covered_i) / Σ weight_i

weight_i = |cluster_i| / |total_claims|
covered_i = 1  if ∃ claim_ans : sim(claim_ans, cluster_i) ≥ threshold
            0  otherwise
```

### Information Density Score (IDS)

```
IDS = (Σ novelty_j / tokens_ans) × (1 - redundancy)

novelty_j = 1 - max(sim(claim_j, context_claims))
redundancy = (|raw_claims| - |unique_claims|) / |raw_claims|
```

Для коротких ответов (≤10 токенов) применяется базовый score 0.3 с бонусом за novelty.

### Faithfulness (NLI Entailment)

Для каждого claim в ответе:

```
faithful = 1  if P(entailment | context, claim) ≥ threshold
           0  if P(contradiction) ≥ 0.5
           1  fallback: sim_semantic(claim, context) ≥ 0.7
```

Для мультиязычных пар пороги мягче:
- `multilingual_threshold = 0.55` (вместо 0.7)
- entailment ≥ 0.15 + contradiction < 0.35 + sim > 0.45

### Business Score (0–100)

```
score = Σ (norm(metric_i) × weight_i) / Σ weight_i

norm(metric) = 100 × (raw - lower) / (upper - lower)
               0   if raw < critical_threshold

score_final = score - latency_penalty   if latency > threshold
```

Guardrails: если metric < min_value → score = fail_score (обычно 0).

### Cost Score

```
generation_cost = input_tokens / 1e6 × price_input + output_tokens / 1e6 × price_output
retrieval_cost  = num_chunks × 0.0001
total_cost      = generation_cost + retrieval_cost

cost_score = 100  if total ≤ 0
             95   if total ≤ 0.001
             80   if total ≤ 0.01
             60   if total ≤ 0.05
             40   if total ≤ 0.1
             20   if total ≤ 0.5
             0    otherwise
```

### Latency Grade

```
EXCELLENT          if latency < target × 0.5      → score = 100
GOOD               if latency < target            → score = 90
ACCEPTABLE         if latency < target × 2        → score = 70
BELOW_EXPECTATIONS if latency < max × 0.5         → score = 40
POOR               if latency < max               → score = 20
UNACCEPTABLE       otherwise                      → score = 0
```

### Trust Score

```
trust = faithfulness × 0.6
        + 0.2 × citation_accuracy   (if has_citations)
        + 0.1 × consistency_score
        + 0.1

trust_level = HIGH   if trust ≥ 0.85
              MEDIUM if trust ≥ 0.6
              LOW    otherwise
```

### Coverage Efficiency

```
efficiency = SFC / num_chunks_retrieved
waste_ratio = 1 - (chunks_used / chunks_retrieved)
```

### Embedding Quality

Диагностика этапа 2 ретривал-пайплайна — `EmbeddingQualityMetrics` (`ragnarok/metrics/embedding_quality.py`).
Отдельно от `evaluator.evaluate()` не запускается напрямую (нужны доп. данные — парафразы, кластеры,
оценки cross-encoder), но участвует в общей диагностике пайплайна (см. ниже "Retrieval Pipeline
Diagnostics"), если эти данные переданы в `evaluate()`. Для точечного вызова — `evaluator.embedding_quality.evaluate(...)`.

| Метрика | Что измеряет | Тревожный сигнал |
|---------|-------------|-------------------|
| Query-Doc Similarity Distribution | Разброс cosine similarity между запросом и топ-чанками | Большой std → нестабильная модель |
| Cluster Compactness | Насколько плотно эмбеддинги похожих документов группируются | Низкая компактность → плохое разделение тем |
| Intra-Query Stability | Вариативность эмбеддингов парафраз одного запроса | Низкая стабильность → чувствительность к формулировке |
| Cross-Encoder Calibration | Корреляция similarity-скора эмбеддингов с оценкой cross-encoder | Низкая корреляция → эмбеддинги плохо ранжируют |

```python
report = evaluator.embedding_quality.evaluate(
    query="When was the Eiffel Tower built?",
    doc_texts=["It was built in 1889.", "Paris is a city in France."],
    clusters=[["It was built in 1889.", "Constructed in 1889."]],
    query_paraphrases=["When was the Eiffel Tower built?", "In what year was it constructed?"],
    embedding_scores=[0.9, 0.4],
    cross_encoder_scores=[0.95, 0.3],
)
```

### Reranking Quality

Диагностика этапа 4 — `RerankingQualityMetrics` (`ragnarok/metrics/reranking_quality.py`).
Сравнивает релевантность до/после реранкинга для одного и того же запроса.

```
Reranking Lift            = MRR@k_after - MRR@k_before        (< 0.02 → реранкер бесполезен)
Relevance Order Accuracy  = доля пар, где порядок после реранкинга не противоречит релевантности
First Relevant Position Shift = позиция_до - позиция_после     (> 0 → реранкер помогает)
```

```python
report = evaluator.reranking_quality.evaluate(
    relevance_before=[0, 0, 1],
    relevance_after=[1, 0, 0],
    k=5,
)
```

### Retrieval Pipeline Diagnostics (bottleneck detection)

`RetrievalPipelineEvaluator` (`ragnarok/metrics/pipeline.py`) runs all 5 stages together and reports
the **first failing stage** as the bottleneck — so a diagnosis can say *"the problem is at stage 2"*
instead of a single opaque retrieval score. It's computed automatically inside `evaluator.evaluate()`
whenever `contexts` are given, and shows up in the result as `retrieval_pipeline_report`:

```python
result = evaluator.evaluate(
    query="When was the Eiffel Tower built?",
    answer="The Eiffel Tower was built in 1889.",
    contexts=["The Eiffel Tower was built in 1889.", "It is located in Paris."],
    relevance_scores=[1, 0],
    # optional extra signal per stage — pass only what you have:
    query_paraphrases=["When was the Eiffel Tower built?", "In what year was it built?"],
    clusters=[["The Eiffel Tower was built in 1889.", "Constructed in 1889."]],
    relevance_before_rerank=[0, 1],
)

report = result["retrieval_pipeline_report"]
print(report["bottleneck_stage"], report["bottleneck_explanation"])
```

Or run it standalone: `evaluator.pipeline.evaluate_pipeline(query=..., contexts=..., ...)`.

---

## Настройка через YAML

```yaml
metrics:
  faithfulness:
    weight: 0.25
    critical_threshold: 0.4
    enabled: true
  sfc:
    weight: 0.20
    enabled: true
  ids:
    weight: 0.15
    enabled: true
  cost_score:
    weight: 0.15
    enabled: true

guardrails:
  - metric: "faithfulness"
    min_value: 0.75
    fail_score: 0.0

interaction_modifiers:
  - condition: "faithfulness > 0.9 and sfc < 0.4"
    penalty: 10
```

**Изменить приоритеты:**

```yaml
# Для медицинского RAG
metrics:
  faithfulness:
    weight: 0.50
  sfc:
    weight: 0.15

# Для поисковика
metrics:
  latency_score:
    weight: 0.25
  cost_score:
    weight: 0.20
```

### Своя цена за токены и GPU-часы

По умолчанию `BusinessMetrics` берёт цену по имени модели из встроенной таблицы (OpenAI и т.п.).
Если у вас свой контракт с провайдером или self-hosted модель — задайте цену явно через `cost_config`
в `EvaluationConfig`, и `RAGEvaluator` подставит её автоматически:

```yaml
cost_config:
  input_token_price: 0.00015    # $ за 1K входных токенов
  output_token_price: 0.0006    # $ за 1K выходных токенов
  model: "my-self-hosted-model" # опционально, только для отображения
  gpu_price_per_hour: 2.50      # $ за GPU-час — используется, если evaluate() передаёт gpu_hours
```

```python
result = evaluator.evaluate(
    query=..., answer=..., contexts=...,
    gpu_hours=0.02,             # если инференс считается по GPU-часам, а не по цене за токен
    price_per_gpu_hour=2.50,    # необязательно — иначе берётся из cost_config
)
result["business_metrics"]["gpu_cost_usd"]
```

Можно также передать цену вручную, без YAML: `BusinessMetrics(model_name="local", custom_pricing={"input": 1.0, "output": 2.0})`
(цены — $ за 1M токенов).

**Не знаете GPU-часы, но знаете число токенов?** `compute_gpu_metrics()` оценивает их по throughput
GPU (A100/A10/T4/V100/RTX4090), либо просто передайте `gpu_type` в `evaluate()`:

```python
result = evaluator.evaluate(
    query=..., answer=..., contexts=...,
    input_tokens=1000, output_tokens=500,
    gpu_type="A100", num_queries=1,   # gpu_hours и его цена оцениваются автоматически
)
result["business_metrics"]["gpu_efficiency_score"]  # 0-100: токенов/GPU-час, нормировано
```

### Composite Business Score — гибкие веса измерений

Помимо взвешенной суммы метрик из `EvaluationConfig`, `BusinessMetrics.compute_composite_business_score()`
считает отдельный композитный скор по 4 бизнес-измерениям (speed/cost/accuracy/coverage) и называет
**самое слабое измерение** — конкретную точку роста, а не просто число. Веса задаются пользователем:

```python
from ragnarok import RAGEvaluator, BusinessDimensionWeights

evaluator = RAGEvaluator(business_dimension_weights=BusinessDimensionWeights(
    speed=0.1, cost=0.1, accuracy=0.6, coverage=0.2  # "точность важнее всего"
))
result = evaluator.evaluate(query=..., answer=..., contexts=...)
result["business_metrics"]["composite_business_score"]
result["business_metrics"]["weakest_dimension"]  # напр. "coverage" -> куда инвестировать в первую очередь
```

---

## Пресеты конфигурации

Готовые сценарии в `ragnarok/presets.py`, чтобы не настраивать веса метрик вручную:

| Пресет | Приоритет |
|--------|-----------|
| `safety_first` | Максимальная faithfulness, жёсткие guardrails — даже ценой скорости/цены |
| `speed_first` | Минимальная задержка — faithfulness и цена вторичны |
| `balanced` | Равномерный баланс faithfulness / retrieval / cost / latency |
| `cost_saving` | Максимальная экономия — допускает более низкую faithfulness и задержку |
| `medical` | Медицинский RAG — самый высокий порог безопасности, жёсткий отказ при риске галлюцинации |
| `customer_support` | Чат-боты поддержки — баланс скорости ответа и точности |
| `research` | Исследовательский/аналитический RAG — максимальная полнота охвата |

```python
from ragnarok import RAGEvaluator, load_preset, suggest_preset, list_presets

print(list_presets())
# {'safety_first': '...', 'speed_first': '...', 'balanced': '...', ...}

evaluator = RAGEvaluator(preset="medical")   # или: RAGEvaluator(load_preset("medical"))
```

**Подбор пресета по свободному описанию** (без YAML, без LLM — простой keyword-matcher):

```python
preset_name = suggest_preset("Мне важно, чтобы система не галлюцинировала, даже если будет медленнее")
# -> "safety_first"
evaluator = RAGEvaluator(preset=preset_name)
```

**Подбор по числовым приоритетам**, если словесного описания недостаточно:

```python
from ragnarok import create_custom_preset

config = create_custom_preset(speed=0.1, cost=0.1, accuracy=0.6, coverage=0.2, base="balanced")
evaluator = RAGEvaluator(config)
```

---

## Отчёты

Проще всего — через сам evaluator: `evaluator.generate_report(result)`. Есть два режима.

### Structured report (по умолчанию, без LLM и API-ключа)

`StructuredReportGenerator` (`ragnarok/reporting.py`) детерминированно строит `StructuredReport`
прямо из результата `evaluate()`: 7 типизированных секций (Executive Summary, Retrieval Pipeline
Diagnostics, Generation Quality, Business Metrics, Fault Analysis, Recommendations, Action Items) с
уровнями критичности и экспортом в Markdown/HTML/dict:

```python
result = evaluator.evaluate(query=..., answer=..., contexts=...)

report = evaluator.generate_report(result)   # use_llm=False по умолчанию
print(report.to_markdown())    # или .to_html(), .to_dict()
```

### LLM-отчёт (опционально)

`LLMReportGenerator` скармливает тот же результат в LLM (переиспользует провайдеров
`OpenAIJudge`/`GigaChatJudge` из `llm_judge.py`) и возвращает развёрнутый текстовый нарратив
поверх тех же данных, что и structured report:

```python
report_text = evaluator.generate_report(result, use_llm=True)  # берёт provider/model из config
# или явно:
report_text = evaluator.generate_report(result, use_llm=True, llm_provider="openai", llm_model="gpt-4o-mini")
```

---

## Опциональные модули

### LLM Judge (OpenAI)

```bash
export OPENAI_API_KEY="sk-..."
```

```yaml
llm_judge_enabled: true
llm_judge_provider: "openai"
llm_judge_model: "gpt-4o-mini"
```

### LLM Judge (GigaChat)

```bash
export GIGACHAT_CLIENT_ID="your-client-id"
export GIGACHAT_CLIENT_SECRET="your-secret"
```

```yaml
llm_judge_enabled: true
llm_judge_provider: "gigachat"
llm_judge_model: "GigaChat"
```

### RAGAS

```bash
pip install -e ".[ragas]"
```

```yaml
ragas_enabled: true
```

---

## Что получаете на выходе

### 1. Business Score (0–100)
Единая оценка качества RAG, взвешенная по бизнес-приоритетам.

### 2. Диагностика (Fault Classification)
- `healthy` — всё хорошо
- `retrieval` — проблема с поиском
- `chunking` — проблема с чанками
- `generation` — галлюцинации/пропуски
- `business` — слишком дорого/медленно
- `out_of_scope` — нет контекста

### 3. Технические метрики
- **Faithfulness** (0–1) — фактологичность
- **SFC** (0–1) — семантическое покрытие
- **IDS** (0–1) — плотность информации
- **HitRate/NDCG/Precision** — качество ретривала

### 4. Бизнес-метрики
- **Cost Score** (0–100) — экономичность
- **Latency Grade** — скорость
- **Trust Score** (0–1) — доверие
- **Coverage Efficiency** — эффективность чанков
- **User Satisfaction Proxy** — прокси NPS

---

## Ошибки и решения

| Ошибка | Решение |
|--------|---------|
| `ModuleNotFoundError: No module named 'nltk'` | `pip install nltk` |
| `LookupError: Resource punkt not found` | Запустите один раз — скачается автоматически |
| `OSError: Model not found` | При первом запуске скачиваются модели (~5 мин) |
| `CUDA out of memory` | Добавьте `device="cpu"` в faithfulness.py |
| `ValueError: Set OPENAI_API_KEY` | Задайте ключ или отключите `llm_judge_enabled: false` |
| `No module named 'openai'` | `pip install -e ".[openai]"` |
| `No module named 'ragas'` | `pip install -e ".[ragas]"` |
