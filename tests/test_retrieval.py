import pytest

from ragnarok import RetrievalMetrics

REL = [1, 0, 1, 0, 1]


def test_hit_rate_at_k():
    assert RetrievalMetrics.hit_rate_at_k(REL, 3) == 1.0


def test_hit_rate_at_k_no_hits():
    assert RetrievalMetrics.hit_rate_at_k([0, 0, 0], 3) == 0.0


def test_precision_at_k():
    assert RetrievalMetrics.precision_at_k(REL, 2) == 0.5


def test_recall_at_k():
    assert RetrievalMetrics.recall_at_k(REL, total_relevant=3, k=5) == pytest.approx(1.0)


def test_recall_at_k_no_relevant_documents():
    assert RetrievalMetrics.recall_at_k(REL, total_relevant=0, k=5) == 0.0


def test_ndcg_at_k_perfect_ranking():
    assert RetrievalMetrics.ndcg_at_k([1, 1, 1], 3) == pytest.approx(1.0)


def test_mrr_at_k_first_hit_at_top():
    assert RetrievalMetrics.mrr_at_k(REL, 5) == 1.0


def test_mrr_at_k_no_relevant():
    assert RetrievalMetrics.mrr_at_k([0, 0, 0], 3) == 0.0
