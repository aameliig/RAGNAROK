from ragnarok import RerankingQualityMetrics


def test_reranking_lift_positive_when_ranking_improves():
    result = RerankingQualityMetrics.reranking_lift([0, 1, 0], [1, 0, 0], k=3)
    assert result["mrr_before"] == 0.5
    assert result["mrr_after"] == 1.0
    assert result["lift"] == 0.5


def test_reranking_lift_negative_when_ranking_worsens():
    result = RerankingQualityMetrics.reranking_lift([1, 0, 0], [0, 1, 0], k=3)
    assert result["lift"] < 0
    assert "negligible" in result["diagnostic"] or "disabling" in result["diagnostic"]


def test_relevance_order_accuracy_perfect_order():
    result = RerankingQualityMetrics.relevance_order_accuracy([2, 1, 0])
    assert result["accuracy"] == 1.0


def test_relevance_order_accuracy_reversed_order():
    result = RerankingQualityMetrics.relevance_order_accuracy([0, 1, 2])
    assert result["accuracy"] == 0.0
    assert "unreliable" in result["diagnostic"]


def test_relevance_order_accuracy_too_few_documents():
    result = RerankingQualityMetrics.relevance_order_accuracy([1])
    assert result["accuracy"] == 1.0
    assert result["total_pairs"] == 0


def test_first_relevant_position_shift_moves_toward_top():
    result = RerankingQualityMetrics.first_relevant_position_shift([0, 0, 1], [1, 0, 0])
    assert result["shift"] == 2
    assert result["position_before"] == 2
    assert result["position_after"] == 0


def test_first_relevant_position_shift_no_relevant_documents():
    result = RerankingQualityMetrics.first_relevant_position_shift([0, 0, 0], [0, 0, 0])
    assert result["position_before"] is None
    assert result["shift"] == 0


def test_evaluate_reports_effective_reranker():
    rq = RerankingQualityMetrics()
    result = rq.evaluate([0, 0, 1], [1, 0, 0], k=3)
    assert result["overall_grade"] == "EFFECTIVE"
    assert result["diagnostics"]
    assert result["recommendations"] == []


def test_evaluate_reports_ineffective_reranker():
    rq = RerankingQualityMetrics()
    result = rq.evaluate([1, 0, 0], [0, 0, 1], k=3)
    assert result["overall_grade"] == "INEFFECTIVE"
    assert result["recommendations"]
