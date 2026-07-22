import pytest

from ragnarok import BusinessMetrics, BusinessDimensionWeights


def test_compute_cost_score():
    bm = BusinessMetrics(model_name="gpt-4o-mini")
    result = bm.compute_cost_score(input_tokens=100, output_tokens=50, num_retrieved_chunks=3)
    assert result["cost_score"] > 0
    assert result["total_cost_usd"] > 0


def test_compute_latency_grade_fast_response():
    bm = BusinessMetrics()
    result = bm.compute_latency_grade(latency_ms=150)
    assert result["latency_grade"] == "EXCELLENT"
    assert result["sla_compliance"] == 1.0


def test_compute_trust_score_with_citations():
    bm = BusinessMetrics()
    result = bm.compute_trust_score(faithfulness=0.9, has_citations=True, citation_accuracy=0.95)
    assert result["trust_level"] == "HIGH"
    assert 0.0 <= result["trust_score"] <= 1.0


def test_compute_coverage_efficiency_zero_chunks():
    bm = BusinessMetrics()
    result = bm.compute_coverage_efficiency(sfc=0.5, num_chunks_retrieved=0)
    assert result["coverage_efficiency"] == 0.0
    assert result["waste_ratio"] == 1.0


def test_compute_user_satisfaction_proxy():
    bm = BusinessMetrics()
    result = bm.compute_user_satisfaction_proxy(
        faithfulness=0.9, relevance=0.8, completeness=0.7, conciseness=0.6
    )
    assert result["nps_proxy"] in {"PROMOTER", "PASSIVE", "DETRACTOR"}


def test_custom_pricing_overrides_builtin_table():
    bm = BusinessMetrics(model_name="my-self-hosted-model", custom_pricing={"input": 1.0, "output": 2.0})
    assert bm.pricing == {"input": 1.0, "output": 2.0}
    result = bm.compute_cost_score(input_tokens=1_000_000, output_tokens=0)
    assert result["generation_cost_usd"] == 1.0


def test_compute_cost_score_includes_gpu_hours():
    bm = BusinessMetrics(model_name="local")
    result = bm.compute_cost_score(
        input_tokens=0, output_tokens=0, gpu_hours=2.0, price_per_gpu_hour=1.5
    )
    assert result["gpu_cost_usd"] == 3.0
    assert result["total_cost_usd"] == 3.0


def test_compute_cost_score_defaults_to_no_gpu_cost():
    bm = BusinessMetrics(model_name="gpt-4o-mini")
    result = bm.compute_cost_score(input_tokens=100, output_tokens=50)
    assert result["gpu_cost_usd"] == 0.0


def test_business_dimension_weights_normalize():
    weights = BusinessDimensionWeights(speed=1.0, cost=1.0, accuracy=1.0, coverage=1.0).normalize()
    assert weights.speed == pytest.approx(0.25)
    assert weights.speed + weights.cost + weights.accuracy + weights.coverage == pytest.approx(1.0)


def test_business_dimension_weights_normalize_handles_zero_total():
    weights = BusinessDimensionWeights(0, 0, 0, 0).normalize()
    assert weights == BusinessDimensionWeights()


def test_compute_gpu_metrics_estimates_cost_from_tokens():
    bm = BusinessMetrics()
    gpu = bm.compute_gpu_metrics(input_tokens=150 * 3600, output_tokens=0, num_queries=1, gpu_type="A100")
    assert gpu.gpu_hours == pytest.approx(1.0, rel=1e-3)
    assert gpu.total_gpu_cost == pytest.approx(gpu.cost_per_gpu_hour, rel=1e-3)


def test_compute_gpu_metrics_unknown_gpu_type_uses_default():
    bm = BusinessMetrics()
    gpu = bm.compute_gpu_metrics(input_tokens=1000, output_tokens=0, gpu_type="unknown-gpu")
    assert gpu.cost_per_gpu_hour == 2.0


def test_compute_composite_business_score_identifies_weakest_dimension():
    bm = BusinessMetrics(dimension_weights=BusinessDimensionWeights(0.25, 0.25, 0.25, 0.25))
    result = bm.compute_composite_business_score(
        latency_score=90, cost_score=20, trust_score=0.9, coverage_score=0.9
    )
    assert result["weakest_dimension"] == "cost"
    assert result["composite_business_score"] == pytest.approx(
        0.25 * 90 + 0.25 * 20 + 0.25 * 90 + 0.25 * 90, rel=1e-3
    )


def test_compute_composite_business_score_uses_default_weights_when_none_given():
    bm = BusinessMetrics(dimension_weights=BusinessDimensionWeights(speed=0.7, cost=0.1, accuracy=0.1, coverage=0.1))
    result = bm.compute_composite_business_score(
        latency_score=100, cost_score=0, trust_score=0.0, coverage_score=0.0
    )
    assert result["dimension_weights"]["speed"] == pytest.approx(0.7)
