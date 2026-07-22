from dataclasses import replace

from ._libs import Dict, List
from .config import EvaluationConfig, MetricConfig, GuardrailConfig

PRESETS: Dict[str, Dict] = {
    "safety_first": {
        "description": "Maximizes faithfulness and enforces strict guardrails, even at the cost of "
                        "speed or price.",
        "metrics": {
            "faithfulness": MetricConfig(weight=0.45, critical_threshold=0.6),
            "sfc": MetricConfig(weight=0.20, critical_threshold=0.4),
            "ids": MetricConfig(weight=0.10, critical_threshold=0.2),
            "trust_score": MetricConfig(weight=0.15, lower_bound=0, upper_bound=1),
            "cost_score": MetricConfig(weight=0.05, lower_bound=0, upper_bound=100),
            "latency_score": MetricConfig(weight=0.05, lower_bound=0, upper_bound=100),
        },
        "guardrails": [
            GuardrailConfig(metric="faithfulness", min_value=0.85, fail_score=0.0),
            GuardrailConfig(metric="trust_score", min_value=0.70, fail_score=0.0),
        ],
        "latency_penalty": 2.0,
        "latency_threshold_ms": 1500,
    },
    "speed_first": {
        "description": "Prioritizes low latency; faithfulness and cost matter less.",
        "metrics": {
            "latency_score": MetricConfig(weight=0.40, lower_bound=0, upper_bound=100),
            "faithfulness": MetricConfig(weight=0.25, critical_threshold=0.3),
            "sfc": MetricConfig(weight=0.15, critical_threshold=0.2),
            "cost_score": MetricConfig(weight=0.10, lower_bound=0, upper_bound=100),
            "ids": MetricConfig(weight=0.10, critical_threshold=0.15),
        },
        "guardrails": [],
        "latency_penalty": 15.0,
        "latency_threshold_ms": 150,
    },
    "balanced": {
        "description": "Even weighting across faithfulness, retrieval coverage, cost, and latency.",
        "metrics": {
            "faithfulness": MetricConfig(weight=0.25, critical_threshold=0.4),
            "sfc": MetricConfig(weight=0.20, critical_threshold=0.3),
            "ids": MetricConfig(weight=0.15, critical_threshold=0.2),
            "cost_score": MetricConfig(weight=0.15, lower_bound=0, upper_bound=100),
            "trust_score": MetricConfig(weight=0.10, lower_bound=0, upper_bound=1),
            "latency_score": MetricConfig(weight=0.15, lower_bound=0, upper_bound=100),
        },
        "guardrails": [
            GuardrailConfig(metric="faithfulness", min_value=0.60, fail_score=0.0),
        ],
        "latency_penalty": 5.0,
        "latency_threshold_ms": 500,
    },
    "cost_saving": {
        "description": "Maximizes cost efficiency; tolerates lower faithfulness and higher latency.",
        "metrics": {
            "cost_score": MetricConfig(weight=0.40, lower_bound=0, upper_bound=100),
            "faithfulness": MetricConfig(weight=0.25, critical_threshold=0.3),
            "sfc": MetricConfig(weight=0.15, critical_threshold=0.2),
            "latency_score": MetricConfig(weight=0.10, lower_bound=0, upper_bound=100),
            "ids": MetricConfig(weight=0.10, critical_threshold=0.15),
        },
        "guardrails": [],
        "latency_penalty": 2.0,
        "latency_threshold_ms": 2000,
    },
    "medical": {
        "description": "Medical RAG: highest safety bar of any preset, hard-fails on any hallucination risk.",
        "metrics": {
            "faithfulness": MetricConfig(weight=0.45, critical_threshold=0.8),
            "trust_score": MetricConfig(weight=0.25, lower_bound=0, upper_bound=1, critical_threshold=0.7),
            "sfc": MetricConfig(weight=0.10, critical_threshold=0.3),
            "ids": MetricConfig(weight=0.05, critical_threshold=0.2),
            "cost_score": MetricConfig(weight=0.05, lower_bound=0, upper_bound=100),
            "coverage_efficiency": MetricConfig(weight=0.05, lower_bound=0, upper_bound=1),
            "latency_score": MetricConfig(weight=0.05, lower_bound=0, upper_bound=100),
        },
        "guardrails": [
            GuardrailConfig(metric="faithfulness", min_value=0.90, fail_score=0.0),
            GuardrailConfig(metric="trust_score", min_value=0.85, fail_score=0.0),
            GuardrailConfig(metric="sfc", min_value=0.25, fail_score=0.0),
        ],
        "latency_penalty": 1.0,
        "latency_threshold_ms": 2000,
    },
    "customer_support": {
        "description": "Customer support chatbots: balances response speed with accuracy customers can trust.",
        "metrics": {
            "latency_score": MetricConfig(weight=0.25, lower_bound=0, upper_bound=100, critical_threshold=20),
            "faithfulness": MetricConfig(weight=0.20, critical_threshold=0.4),
            "trust_score": MetricConfig(weight=0.15, lower_bound=0, upper_bound=1, critical_threshold=0.4),
            "cost_score": MetricConfig(weight=0.15, lower_bound=0, upper_bound=100),
            "sfc": MetricConfig(weight=0.10, critical_threshold=0.2),
            "ids": MetricConfig(weight=0.10, critical_threshold=0.1),
            "coverage_efficiency": MetricConfig(weight=0.05, lower_bound=0, upper_bound=1),
        },
        "guardrails": [
            GuardrailConfig(metric="faithfulness", min_value=0.60, fail_score=0.0),
            GuardrailConfig(metric="latency_score", min_value=20, fail_score=0.0),
        ],
        "latency_penalty": 8.0,
        "latency_threshold_ms": 300,
    },
    "research": {
        "description": "Research/analytical RAG: maximizes coverage and completeness over speed or cost.",
        "metrics": {
            "sfc": MetricConfig(weight=0.30, critical_threshold=0.4),
            "faithfulness": MetricConfig(weight=0.25, critical_threshold=0.5),
            "ids": MetricConfig(weight=0.15, critical_threshold=0.2),
            "coverage_efficiency": MetricConfig(weight=0.10, lower_bound=0, upper_bound=1, critical_threshold=0.1),
            "trust_score": MetricConfig(weight=0.10, lower_bound=0, upper_bound=1),
            "cost_score": MetricConfig(weight=0.05, lower_bound=0, upper_bound=100),
            "latency_score": MetricConfig(weight=0.05, lower_bound=0, upper_bound=100),
        },
        "guardrails": [
            GuardrailConfig(metric="sfc", min_value=0.50, fail_score=0.0),
            GuardrailConfig(metric="faithfulness", min_value=0.70, fail_score=0.0),
        ],
        "latency_penalty": 2.0,
        "latency_threshold_ms": 2000,
    },
}

# The user's free text is matched to a preset by keywords —
# no LLM, so picking a preset doesn't require a network call or an API key.
_KEYWORD_MAP: Dict[str, List[str]] = {
    "safety_first": [
        "safety", "safe", "hallucin", "faithful", "trust", "accura", "risk", "compliance",
        "legal", "financial", "high stakes",
        "безопас", "галлюцинац", "довер", "точност", "риск", "юридич", "финанс",
    ],
    "speed_first": [
        "speed", "fast", "latency", "quick", "real-time", "realtime", "responsive",
        "скорост", "быстр", "задержк", "реальн",
    ],
    "cost_saving": [
        "cost", "cheap", "budget", "price", "expensive", "save money", "economical", "self-hosted",
        "дешев", "бюджет", "стоимост", "экономи", "цен",
    ],
    "balanced": [
        "balance", "balanced", "general", "default", "everything", "баланс", "обычн", "общ",
    ],
    "medical": [
        "medical", "doctor", "hospital", "diagnosis", "patient", "clinical", "healthcare", "pharma",
        "медицин", "врач", "диагноз", "пациент", "лечение", "клиник",
    ],
    "customer_support": [
        "support", "customer", "helpdesk", "chatbot", "ticket", "assistance",
        "поддержк", "клиент", "чат-бот", "оператор", "тикет",
    ],
    "research": [
        "research", "analysis", "academic", "science", "paper", "comprehensive", "thorough", "literature",
        "исследован", "анализ", "наука", "академ", "статья", "полнот", "всё о",
    ],
}


def list_presets() -> Dict[str, str]:
    """Preset names and their short description"""
    return {name: cfg["description"] for name, cfg in PRESETS.items()}


def load_preset(name: str) -> EvaluationConfig:
    """Builds an EvaluationConfig from a named preset"""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available presets: {', '.join(PRESETS)}")
    preset = PRESETS[name]
    return EvaluationConfig(
        metrics={name: replace(cfg) for name, cfg in preset["metrics"].items()},
        guardrails=[replace(g) for g in preset["guardrails"]],
        latency_penalty=preset["latency_penalty"],
        latency_threshold_ms=preset["latency_threshold_ms"],
    )


def suggest_preset(description: str) -> str:
    """
    Picks a preset from a free-text description of the user's priorities
    (e.g. "I don't want the system to hallucinate, even if it's slower").
    Returns a preset name from PRESETS; falls back to "balanced" if nothing matches.
    """
    text = description.lower()
    scores = {name: sum(1 for kw in keywords if kw in text) for name, keywords in _KEYWORD_MAP.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "balanced"


def create_custom_preset(speed: float = 0.25, cost: float = 0.25,
                          accuracy: float = 0.30, coverage: float = 0.20,
                          base: str = "balanced") -> EvaluationConfig:
    """
    Builds an EvaluationConfig from numeric priorities (not necessarily normalized to 1.0),
    rather than from keywords

    accuracy is split between faithfulness (70%) and trust_score (30%);
    coverage — between sfc (60%) and coverage_efficiency (40%);
    speed -> latency_score, cost -> cost_score.
    Guardrails/latency_penalty/latency_threshold_ms are taken from the `base` preset unchanged.
    """
    if base not in PRESETS:
        raise ValueError(f"Unknown base preset '{base}'. Available presets: {', '.join(PRESETS)}")

    total = speed + cost + accuracy + coverage
    if total <= 0:
        speed, cost, accuracy, coverage = 0.25, 0.25, 0.25, 0.25
        total = 1.0
    speed, cost, accuracy, coverage = (v / total for v in (speed, cost, accuracy, coverage))

    base_preset = PRESETS[base]
    metrics: Dict[str, MetricConfig] = {name: replace(cfg) for name, cfg in base_preset["metrics"].items()}

    def _set_weight(name: str, weight: float, **defaults):
        if name in metrics:
            metrics[name].weight = round(weight, 4)
        else:
            metrics[name] = MetricConfig(weight=round(weight, 4), **defaults)

    _set_weight("faithfulness", accuracy * 0.7)
    _set_weight("trust_score", accuracy * 0.3, lower_bound=0, upper_bound=1)
    _set_weight("sfc", coverage * 0.6)
    _set_weight("coverage_efficiency", coverage * 0.4, lower_bound=0, upper_bound=1)
    _set_weight("latency_score", speed, lower_bound=0, upper_bound=100)
    _set_weight("cost_score", cost, lower_bound=0, upper_bound=100)

    total_weight = sum(m.weight for m in metrics.values() if m.enabled)
    if total_weight > 0:
        for m in metrics.values():
            if m.enabled:
                m.weight = round(m.weight / total_weight, 4)

    return EvaluationConfig(
        metrics=metrics,
        guardrails=list(base_preset["guardrails"]),
        latency_penalty=base_preset["latency_penalty"],
        latency_threshold_ms=base_preset["latency_threshold_ms"],
    )
