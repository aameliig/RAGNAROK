from .._libs import List, Dict, Any, Optional, dataclass


@dataclass
class CostBreakdown:
    input_tokens: int
    output_tokens: int
    retrieval_cost: float
    generation_cost: float
    total_cost: float


@dataclass
class GPUMetrics:
    """GPU usage estimate for self-hosted models, computed from token counts"""
    gpu_hours: float
    gpu_hours_per_query: float
    tokens_per_gpu_hour: float
    cost_per_gpu_hour: float
    total_gpu_cost: float
    efficiency_score: float  # 0-100


@dataclass
class BusinessDimensionWeights:
    """
    Flexible business-dimension weights — the user decides what matters most:
    speed, cost, accuracy, or answer completeness.
    """
    speed: float = 0.25      # latency
    cost: float = 0.25       # cost_score
    accuracy: float = 0.30   # faithfulness / trust
    coverage: float = 0.20   # SFC / recall

    def normalize(self) -> "BusinessDimensionWeights":
        """Normalizes the weights so that their sum equals 1.0"""
        total = self.speed + self.cost + self.accuracy + self.coverage
        if total <= 0:
            return BusinessDimensionWeights()
        return BusinessDimensionWeights(
            speed=self.speed / total,
            cost=self.cost / total,
            accuracy=self.accuracy / total,
            coverage=self.coverage / total,
        )


class BusinessMetrics:
    """Business metrics for evaluating RAG cost, efficiency, and trust"""

    PRICING = {
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gigachat": {"input": 0.0, "output": 0.0},
        "all-MiniLM-L6-v2": {"input": 0, "output": 0},
        "cross-encoder/nli-deberta-v3-large": {"input": 0, "output": 0},
    }

    # Approximate throughput (tokens/sec) and price ($/GPU-hour) for self-hosted estimates —
    # ballpark values for a 7-8B model, use as a starting point rather than an exact calculation.
    GPU_THROUGHPUT_TOKENS_PER_SEC = {
        "A100": 150, "A10": 80, "T4": 40, "V100": 100, "RTX4090": 120,
    }
    GPU_PRICE_PER_HOUR = {
        "A100": 2.5, "A10": 1.2, "T4": 0.6, "V100": 2.0, "RTX4090": 0.8,
    }

    def __init__(self, model_name: str = "gpt-4o-mini", custom_pricing: Optional[Dict[str, float]] = None,
                 dimension_weights: Optional[BusinessDimensionWeights] = None):
        """
        custom_pricing: {"input": price_per_1M_tokens, "output": price_per_1M_tokens}.
        Overrides the built-in PRICING table

        dimension_weights: weights for compute_composite_business_score (defaults to
        speed=0.25, cost=0.25, accuracy=0.30, coverage=0.20).
        """
        self.model_name = model_name
        self.pricing = custom_pricing or self.PRICING.get(model_name, {"input": 0, "output": 0})
        self.dimension_weights = (dimension_weights or BusinessDimensionWeights()).normalize()

    def compute_cost_score(self,
                          input_tokens: int,
                          output_tokens: int,
                          num_retrieved_chunks: int = 0,
                          embedding_dim: int = 384,
                          gpu_hours: float = 0.0,
                          price_per_gpu_hour: float = 0.0) -> Dict[str, Any]:
        """
        gpu_hours / price_per_gpu_hour: for self-hosted inference, where cost is computed
        by GPU usage time rather than by price per token.
        """
        gen_cost = (input_tokens / 1e6 * self.pricing["input"] +
                   output_tokens / 1e6 * self.pricing["output"])
        retrieval_cost = num_retrieved_chunks * 0.0001 if self.pricing["input"] > 0 else 0.0
        gpu_cost = gpu_hours * price_per_gpu_hour
        total = gen_cost + retrieval_cost + gpu_cost

        if total <= 0:
            cost_score = 100.0
        elif total <= 0.001:
            cost_score = 95.0
        elif total <= 0.01:
            cost_score = 80.0
        elif total <= 0.05:
            cost_score = 60.0
        elif total <= 0.1:
            cost_score = 40.0
        elif total <= 0.5:
            cost_score = 20.0
        else:
            cost_score = 0.0

        return {
            "cost_score": cost_score,
            "generation_cost_usd": round(gen_cost, 6),
            "retrieval_cost_usd": round(retrieval_cost, 6),
            "gpu_cost_usd": round(gpu_cost, 6),
            "total_cost_usd": round(total, 6),
            "cost_efficiency": round(output_tokens / max(total, 0.000001), 2),
            "tokens_per_dollar": round(output_tokens / max(total, 0.000001), 2)
        }

    def compute_latency_grade(self, latency_ms: float, 
                             target_ms: float = 500,
                             max_acceptable_ms: float = 3000) -> Dict[str, Any]:
        if latency_ms < target_ms * 0.5:
            grade = "EXCELLENT"
            score = 100
        elif latency_ms < target_ms:
            grade = "GOOD"
            score = 90
        elif latency_ms < target_ms * 2:
            grade = "ACCEPTABLE"
            score = 70
        elif latency_ms < max_acceptable_ms * 0.5:
            grade = "BELOW_EXPECTATIONS"
            score = 40
        elif latency_ms < max_acceptable_ms:
            grade = "POOR"
            score = 20
        else:
            grade = "UNACCEPTABLE"
            score = 0

        sla_compliance = 1.0 if latency_ms < target_ms else 0.0

        return {
            "latency_grade": grade,
            "latency_score": score,
            "sla_compliance": sla_compliance,
            "latency_ms": latency_ms,
            "latency_vs_target": round(latency_ms / target_ms, 2)
        }

    def compute_trust_score(self, 
                           faithfulness: float,
                           has_citations: bool = False,
                           citation_accuracy: float = 0.0,
                           consistency_score: float = 1.0) -> Dict[str, Any]:
        trust = faithfulness * 0.6
        if has_citations:
            trust += 0.2 * citation_accuracy
        trust += 0.1 * consistency_score
        trust += 0.1

        trust_level = "HIGH" if trust >= 0.85 else "MEDIUM" if trust >= 0.6 else "LOW"

        return {
            "trust_score": round(min(trust, 1.0), 3),
            "trust_level": trust_level,
            "faithfulness_component": round(faithfulness * 0.6, 3),
            "citation_component": round(0.2 * citation_accuracy if has_citations else 0, 3),
            "consistency_component": round(0.1 * consistency_score, 3),
            "transparency_bonus": 0.1 if has_citations else 0,
            "recommendation": "Add citations" if not has_citations and faithfulness < 0.8 else "OK"
        }

    def compute_coverage_efficiency(self,
                                    sfc: float,
                                    num_chunks_retrieved: int,
                                    num_chunks_used: int = None) -> Dict[str, Any]:
        if num_chunks_retrieved == 0:
            return {
                "coverage_efficiency": 0.0, 
                "waste_ratio": 1.0,
                "optimal_chunks": 0,
                "efficiency_grade": "N/A"
            }

        efficiency = sfc / num_chunks_retrieved
        used = num_chunks_used or max(1, int(sfc * num_chunks_retrieved))
        waste = 1 - (used / num_chunks_retrieved)

        grade = "EXCELLENT" if efficiency >= 0.3 else "GOOD" if efficiency >= 0.2 else "FAIR" if efficiency >= 0.1 else "POOR"

        return {
            "coverage_efficiency": round(efficiency, 3),
            "waste_ratio": round(waste, 3),
            "optimal_chunks": max(1, int(sfc * 5)),
            "efficiency_grade": grade,
            "chunks_retrieved": num_chunks_retrieved,
            "chunks_used_estimate": used
        }

    def compute_user_satisfaction_proxy(self,
                                        faithfulness: float,
                                        relevance: float = 0.0,
                                        completeness: float = 0.0,
                                        conciseness: float = 0.0) -> Dict[str, Any]:
        satisfaction = (
            faithfulness * 0.35 +
            relevance * 0.25 +
            completeness * 0.25 +
            conciseness * 0.15
        )

        nps_proxy = "PROMOTER" if satisfaction >= 0.8 else "PASSIVE" if satisfaction >= 0.5 else "DETRACTOR"

        return {
            "user_satisfaction_proxy": round(satisfaction, 3),
            "nps_proxy": nps_proxy,
            "satisfaction_breakdown": {
                "trust": round(faithfulness * 0.35, 3),
                "relevance": round(relevance * 0.25, 3),
                "completeness": round(completeness * 0.25, 3),
                "conciseness": round(conciseness, 3)
            }
        }

    def compute_token_efficiency(self,
                                  answer_tokens: int,
                                  context_tokens: int,
                                  novel_claims: int = 0) -> Dict[str, Any]:
        if answer_tokens == 0:
            return {"token_efficiency": 0.0, "compression_ratio": 0.0}

        efficiency = novel_claims / answer_tokens
        compression = context_tokens / answer_tokens if answer_tokens > 0 else 0

        return {
            "token_efficiency": round(efficiency, 4),
            "compression_ratio": round(compression, 2),
            "novel_claims": novel_claims,
            "answer_tokens": answer_tokens,
            "context_tokens": context_tokens
        }

    def compute_gpu_metrics(self,
                             input_tokens: int,
                             output_tokens: int,
                             num_queries: int = 1,
                             gpu_type: str = "A100") -> GPUMetrics:
        """
        Estimates GPU-hours and their cost from token counts, without an explicit gpu_hours —
        useful when the user only knows the token volume, not the inference time.
        Throughput/price are ballpark figures (see GPU_THROUGHPUT_TOKENS_PER_SEC/GPU_PRICE_PER_HOUR);
        for an exact calculation pass gpu_hours directly to compute_cost_score.
        """
        throughput = self.GPU_THROUGHPUT_TOKENS_PER_SEC.get(gpu_type, 100)
        total_tokens = input_tokens + output_tokens

        gpu_hours = (total_tokens / max(throughput, 1)) / 3600
        gpu_hours_per_query = gpu_hours / max(num_queries, 1)
        tokens_per_gpu_hour = total_tokens / max(gpu_hours, 0.000001)

        cost_per_hour = self.GPU_PRICE_PER_HOUR.get(gpu_type, 2.0)
        total_gpu_cost = gpu_hours * cost_per_hour

        # 1M tokens/hour is normalized to score = 100
        efficiency = min(100.0, tokens_per_gpu_hour / 10000)

        return GPUMetrics(
            gpu_hours=round(gpu_hours, 6),
            gpu_hours_per_query=round(gpu_hours_per_query, 6),
            tokens_per_gpu_hour=round(tokens_per_gpu_hour, 2),
            cost_per_gpu_hour=cost_per_hour,
            total_gpu_cost=round(total_gpu_cost, 6),
            efficiency_score=round(efficiency, 2),
        )

    def compute_composite_business_score(self,
                                          latency_score: float,
                                          cost_score: float,
                                          trust_score: float,
                                          coverage_score: float,
                                          dimension_weights: Optional[BusinessDimensionWeights] = None
                                          ) -> Dict[str, Any]:
        """
        Business score with flexible dimension weights (speed/cost/accuracy/coverage).
        The user can set priorities ("speed matters most to me" -> high speed weight)
        via dimension_weights in __init__ or here.
        """
        weights = (dimension_weights or self.dimension_weights).normalize()

        norm_trust = trust_score * 100
        norm_coverage = coverage_score * 100

        dimension_scores = {
            "speed": latency_score,
            "cost": cost_score,
            "accuracy": norm_trust,
            "coverage": norm_coverage,
        }

        composite = (
            weights.speed * dimension_scores["speed"] +
            weights.cost * dimension_scores["cost"] +
            weights.accuracy * dimension_scores["accuracy"] +
            weights.coverage * dimension_scores["coverage"]
        )

        weakest = min(dimension_scores, key=dimension_scores.get)

        return {
            "composite_business_score": round(composite, 2),
            "dimension_weights": {
                "speed": round(weights.speed, 3),
                "cost": round(weights.cost, 3),
                "accuracy": round(weights.accuracy, 3),
                "coverage": round(weights.coverage, 3),
            },
            "dimension_scores": {k: round(v, 2) for k, v in dimension_scores.items()},
            "weakest_dimension": weakest,
            "weakest_score": round(dimension_scores[weakest], 2),
            "recommendation": f"Improving the '{weakest}' dimension would give the largest business-score gain.",
        }
