from ._libs import dataclass, field, Dict, List, Any, Optional, Union, Tuple, Path, yaml, json, re


@dataclass
class MetricConfig:
    weight: float = 1.0
    lower_bound: float = 0.0
    upper_bound: float = 1.0
    critical_threshold: float = 0.4
    enabled: bool = True


@dataclass
class NormalizationRule:
    type: str = "linear"
    lower_bound: float = 0.0
    upper_bound: float = 1.0
    thresholds: Optional[Dict[float, float]] = None


@dataclass
class GuardrailConfig:
    metric: str
    min_value: float
    fail_score: float = 0.0


@dataclass
class EvaluationConfig:
    metrics: Dict[str, MetricConfig] = field(default_factory=dict)
    normalization: Dict[str, NormalizationRule] = field(default_factory=dict)
    guardrails: List[GuardrailConfig] = field(default_factory=list)
    latency_penalty: float = 5.0
    latency_threshold_ms: int = 300
    bonus_penalty: List[Dict[str, Any]] = field(default_factory=list)
    interaction_modifiers: List[Dict[str, Any]] = field(default_factory=list)
    ragas_enabled: bool = False
    llm_judge_enabled: bool = False
    llm_judge_provider: str = "openai"
    llm_judge_model: str = "gpt-4o-mini"
    fault_thresholds: Dict[str, float] = field(default_factory=dict)
    business_dimensions: Dict[str, float] = field(default_factory=lambda: {
        "retrieval": 0.25,
        "generation": 0.40,
        "business": 0.35
    })
    cost_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "EvaluationConfig":
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "EvaluationConfig":
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "EvaluationConfig":
        metrics = {}
        for name, cfg in data.get("metrics", {}).items():
            metrics[name] = MetricConfig(**cfg)

        normalization = {}
        for name, rule in data.get("normalization", {}).items():
            normalization[name] = NormalizationRule(**rule)

        guardrails = [GuardrailConfig(**g) for g in data.get("guardrails", [])]

        return cls(
            metrics=metrics,
            normalization=normalization,
            guardrails=guardrails,
            latency_penalty=data.get("latency_penalty", 5.0),
            latency_threshold_ms=data.get("latency_threshold_ms", 300),
            bonus_penalty=data.get("bonus_penalty", []),
            interaction_modifiers=data.get("interaction_modifiers", []),
            ragas_enabled=data.get("ragas_enabled", False),
            llm_judge_enabled=data.get("llm_judge_enabled", False),
            llm_judge_provider=data.get("llm_judge_provider", "openai"),
            llm_judge_model=data.get("llm_judge_model", "gpt-4o-mini"),
            fault_thresholds=data.get("fault_thresholds", {
                "hit_rate": 0.70, "ndcg": 0.65, "sfc": 0.60,
                "ids": 0.30, "faithfulness": 0.75,
                "cost_score": 50.0, "trust_score": 0.6,
                "coverage_efficiency": 0.15
            }),
            business_dimensions=data.get("business_dimensions", {
                "retrieval": 0.25, "generation": 0.40, "business": 0.35
            }),
            cost_config=data.get("cost_config", {})
        )

    def to_yaml(self, path: Union[str, Path]) -> None:
        from dataclasses import asdict
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, allow_unicode=True)

    def aggregate(self, scores: Dict[str, float], latency_ms: float) -> Tuple[float, float, str]:
        if not self.metrics:
            return 0.0, 0.0, "FAILED"

        for guardrail in self.guardrails:
            raw = scores.get(guardrail.metric, 1.0)
            if raw < guardrail.min_value:
                return guardrail.fail_score, 0.0, "GUARDRAIL_VIOLATION"

        total_w = sum(c.weight for c in self.metrics.values() if c.enabled)
        final = 0.0

        for m, cfg in self.metrics.items():
            if not cfg.enabled:
                continue
            raw = scores.get(m, 0.0)

            norm_rule = self.normalization.get(m)
            if norm_rule and norm_rule.type == "step_penalty":
                norm = self._apply_step_penalty(raw, norm_rule.thresholds)
            else:
                norm = 100 * ((raw - cfg.lower_bound) /
                             (cfg.upper_bound - cfg.lower_bound)) if cfg.upper_bound > cfg.lower_bound else 0.0

            norm = max(0.0, min(100.0, norm))
            if raw < cfg.critical_threshold:
                norm = 0.0
            final += norm * cfg.weight

        final = final / total_w if total_w > 0 else 0.0
        penalty = self.latency_penalty if latency_ms > self.latency_threshold_ms else 0.0
        final = max(0.0, final - penalty)

        for modifier in self.interaction_modifiers:
            if self._eval_condition(modifier.get("condition", ""), latency_ms, scores):
                final += modifier.get("bonus", 0)
                final -= modifier.get("penalty", 0)

        for modifier in self.bonus_penalty:
            if self._eval_condition(modifier.get("condition", ""), latency_ms, scores):
                final += modifier.get("bonus", 0)
                final -= modifier.get("penalty", 0)

        status = "PASSED" if final >= 60 else "FAILED"
        return final, penalty, status

    @staticmethod
    def _apply_step_penalty(raw: float, thresholds: Optional[Dict[float, float]]) -> float:
        if not thresholds:
            return raw * 100
        for thresh, score in sorted(thresholds.items()):
            if raw <= thresh:
                return score
        return 0.0

    def _eval_condition(self, condition: str, latency_ms: float, scores: Dict[str, float]) -> bool:
        if not condition or not condition.strip():
            return False

        env = {"latency": latency_ms, **scores}
        expr = condition.lower()

        import re
        for key, val in env.items():
            if isinstance(val, (int, float)):
                expr = re.sub(r'\b' + re.escape(key.lower()) + r'\b', str(val), expr)

        expr = expr.replace("and", " and ").replace("or", " or ")
        expr = expr.replace("<", " < ").replace(">", " > ").replace("=", " == ")
        expr = expr.replace("  ", " ").strip()

        try:
            allowed = {"__builtins__": None}
            return bool(eval(expr, allowed, {}))
        except Exception:
            if "latency" in condition and "<" in condition:
                try:
                    val = float(condition.split("<")[1].split()[0])
                    return latency_ms < val
                except:
                    pass
            return False
