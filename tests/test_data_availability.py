from datetime import datetime, timedelta

import pytest

from ragnarok import DataAvailabilityMetrics

CORPUS = [
    "The Eiffel Tower is in Paris.",
    "Paris is the capital of France.",
    "In 1889, the World's Fair was held in Paris.",
    "The height of the Eiffel Tower is 330 meters.",
]
QUERIES = [
    "Where is the Eiffel Tower?",
    "What is the capital of France?",
    "When was Big Ben built?",
    "How tall is the Eiffel Tower?",
]


def test_empty_result_rate():
    dam = DataAvailabilityMetrics()
    assert dam.empty_result_rate([5, 0, 3, 0]) == 0.5


def test_empty_result_rate_no_queries():
    assert DataAvailabilityMetrics.empty_result_rate([]) == 0.0


def test_coverage_gap():
    dam = DataAvailabilityMetrics()
    retrieved = [["docA", "docB"], ["docC"], ["docD"], ["docE", "docF"]]
    relevant = [["docA", "docX"], ["docC"], ["docY"], ["docE", "docF"]]
    assert dam.coverage_gap(retrieved, relevant) == 0.5


def test_coverage_gap_requires_matching_lengths():
    dam = DataAvailabilityMetrics()
    with pytest.raises(ValueError):
        dam.coverage_gap([["a"]], [["a"], ["b"]])


def test_index_freshness_reports_stale_ratio():
    dam = DataAvailabilityMetrics()
    now = datetime.now()
    timestamps = [now - timedelta(days=i * 10) for i in range(10)]
    freshness = dam.index_freshness(timestamps, max_age_days=30)
    assert 0.0 <= freshness["stale_ratio"] <= 1.0
    assert 0.0 <= freshness["freshness_score"] <= 100.0


def test_index_freshness_empty_is_fully_fresh():
    freshness = DataAvailabilityMetrics.index_freshness([])
    assert freshness["freshness_score"] == 100.0


def test_detect_missing_documents_flags_uncovered_query():
    dam = DataAvailabilityMetrics()
    missing = dam.detect_missing_documents(CORPUS, QUERIES, threshold=0.1)
    assert missing["missing_queries"] == ["When was Big Ben built?"]
    assert 0.0 < missing["missing_ratio"] < 1.0


def test_detect_missing_documents_empty_corpus():
    missing = DataAvailabilityMetrics.detect_missing_documents([], QUERIES)
    assert missing["missing_ratio"] == 0.0
    assert missing["missing_queries"] == []


def test_evaluate_aggregates_all_submetrics():
    dam = DataAvailabilityMetrics()
    now = datetime.now()
    timestamps = [now - timedelta(days=i * 10) for i in range(10)]

    report = dam.evaluate(
        queries=QUERIES,
        num_chunks_found=[2, 1, 0, 1],
        retrieved_doc_ids=[["doc1", "doc2"], ["doc3"], [], ["doc4"]],
        relevant_doc_ids=[["doc1", "doc2"], ["doc3"], ["doc5"], ["doc4"]],
        document_timestamps=timestamps,
        max_age_days=30,
        corpus=CORPUS,
        tfidf_threshold=0.1,
    )

    assert report["empty_result_rate"] == pytest.approx(0.25)
    assert report["coverage_gap"] == pytest.approx(0.25)
    assert report["index_freshness"] is not None
    assert report["missing_docs_tfidf"] is not None
    assert report["diagnostic"]
