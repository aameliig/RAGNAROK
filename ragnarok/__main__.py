"""Runnable demo: evaluate a single RAG answer end-to-end.

Run via `python -m ragnarok` or the root-level `main.py`.
For the automated test suite, run `pytest` instead (see tests/).
"""
import time

from .evaluator import RAGEvaluator
from .config import EvaluationConfig, MetricConfig
from .utils import measure_latency


def main():
    config = EvaluationConfig(
        metrics={
            "faithfulness": MetricConfig(weight=0.7),
            "sfc": MetricConfig(weight=0.3),
            "cost_score": MetricConfig(weight=0.0, enabled=False),
            "trust_score": MetricConfig(weight=0.0, enabled=False),
            "coverage_efficiency": MetricConfig(weight=0.0, enabled=False),
            "latency_score": MetricConfig(weight=0.0, enabled=False),
        },
        latency_penalty=10.0,
        latency_threshold_ms=100,
    )
    evaluator = RAGEvaluator(config)

    @measure_latency
    def simulate_answer():
        time.sleep(0.05)
        return "The Eiffel Tower was built in 1889."

    answer, latency_ms = simulate_answer()
    result = evaluator.evaluate(
        query="When was the Eiffel Tower built?",
        answer=answer,
        contexts=["The Eiffel Tower was constructed in 1889 for the World's Fair in Paris."],
        relevance_scores=[1, 0, 1],
        latency_ms=latency_ms,
        input_tokens=10,
        output_tokens=8,
        num_chunks_retrieved=1,
    )

    RAGEvaluator.print_results(result, title="RAGNAROK Demo Evaluation")


if __name__ == "__main__":
    main()
