import time

from ragnarok import EvaluationConfig, MetricConfig, RAGEvaluator, measure_latency


def _build_evaluator() -> RAGEvaluator:
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
    return RAGEvaluator(config)


def test_evaluate_end_to_end_produces_a_business_score():
    evaluator = _build_evaluator()

    @measure_latency
    def simulate_answer():
        time.sleep(0.05)
        return "The Eiffel Tower was built in 1889."

    answer, latency_ms = simulate_answer()
    result = evaluator.evaluate(
        query="When was Eiffel built?",
        answer=answer,
        contexts=["It was built in 1889."],
        relevance_scores=[1, 0, 1],
        latency_ms=latency_ms,
        input_tokens=10,
        output_tokens=8,
        num_chunks_retrieved=1,
    )

    assert 0.0 <= result["final_score"] <= 100.0
    assert result["status"] in {"PASSED", "FAILED", "GUARDRAIL_VIOLATION"}
    assert "fault_type" in result


def test_evaluate_includes_data_availability_and_chunking_quality():
    evaluator = _build_evaluator()

    result = evaluator.evaluate(
        query="When was Eiffel built?",
        answer="The Eiffel Tower was built in 1889.",
        contexts=["It was built in 1889.", "It is located in Paris."],
    )

    assert result["data_availability"] is not None
    assert result["metrics"]["data_availability"] is result["data_availability"]

    assert result["chunking_quality"] is not None
    assert result["chunking_quality"]["overall_grade"] != "NO_DATA"
    assert result["metrics"]["chunking_quality"] is result["chunking_quality"]


def test_evaluate_without_contexts_skips_chunking_quality():
    evaluator = _build_evaluator()

    result = evaluator.evaluate(
        query="When was Eiffel built?",
        answer="I don't know.",
        contexts=[],
    )

    assert result["chunking_quality"] is None


def test_evaluate_includes_gpu_cost_when_gpu_hours_given():
    evaluator = _build_evaluator()

    result = evaluator.evaluate(
        query="When was Eiffel built?",
        answer="The Eiffel Tower was built in 1889.",
        contexts=["It was built in 1889."],
        gpu_hours=2.0,
        price_per_gpu_hour=1.5,
    )

    assert result["business_metrics"]["gpu_cost_usd"] == 3.0


def test_evaluator_uses_cost_config_for_custom_pricing():
    config = EvaluationConfig(
        metrics={"faithfulness": MetricConfig(weight=1.0)},
        cost_config={"input_token_price": 0.001, "output_token_price": 0.002, "model": "self-hosted"},
    )
    evaluator = RAGEvaluator(config)

    assert evaluator.business_metrics.pricing == {"input": 1.0, "output": 2.0}


def test_evaluate_without_contexts_skips_retrieval_pipeline_report():
    evaluator = _build_evaluator()
    result = evaluator.evaluate(query="q", answer="I don't know.", contexts=[])
    assert result["retrieval_pipeline_report"] is None


def test_evaluator_accepts_preset_by_name():
    evaluator = RAGEvaluator(preset="safety_first")
    assert "faithfulness" in evaluator.config.metrics
    assert evaluator.config.metrics["faithfulness"].weight == 0.45


def test_evaluate_includes_retrieval_pipeline_report_with_bottleneck():
    evaluator = _build_evaluator()
    result = evaluator.evaluate(
        query="When was Eiffel built?",
        answer="The Eiffel Tower was built in 1889.",
        contexts=["It was built in 1889.", "It is located in Paris."],
        relevance_scores=[1, 0],
    )

    report = result["retrieval_pipeline_report"]
    assert report is not None
    assert len(report["stages"]) == 5
    assert "overall_passed" in report


def test_evaluate_includes_composite_business_score():
    evaluator = _build_evaluator()
    result = evaluator.evaluate(
        query="When was Eiffel built?",
        answer="The Eiffel Tower was built in 1889.",
        contexts=["It was built in 1889."],
        num_chunks_retrieved=1,
    )
    biz = result["business_metrics"]
    assert "composite_business_score" in biz
    assert biz["weakest_dimension"] in {"speed", "cost", "accuracy", "coverage"}


def test_evaluate_gpu_type_estimates_gpu_cost_from_tokens():
    evaluator = _build_evaluator()
    result = evaluator.evaluate(
        query="q", answer="The Eiffel Tower was built in 1889.", contexts=["It was built in 1889."],
        input_tokens=150 * 3600, output_tokens=0, gpu_type="A100",
    )
    biz = result["business_metrics"]
    assert biz["gpu_cost_usd"] > 0
    assert "gpu_efficiency_score" in biz


def test_generate_report_default_is_structured_and_needs_no_llm():
    evaluator = _build_evaluator()
    result = evaluator.evaluate(
        query="q", answer="The Eiffel Tower was built in 1889.", contexts=["It was built in 1889."],
    )
    report = evaluator.generate_report(result)
    markdown = report.to_markdown()
    assert "RAGNAROK Evaluation Report" in markdown
