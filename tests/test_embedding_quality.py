import pytest

from ragnarok import EmbeddingQualityMetrics


def test_query_doc_similarity_distribution_returns_stats():
    eq = EmbeddingQualityMetrics()
    result = eq.query_doc_similarity_distribution(
        "When was the Eiffel Tower built?",
        ["The Eiffel Tower was built in 1889.", "Bananas are a good source of potassium."],
    )
    assert 0.0 <= result["mean_similarity"] <= 1.0
    assert len(result["similarities"]) == 2
    assert result["requires_embedding"] is False


def test_query_doc_similarity_distribution_empty_inputs():
    eq = EmbeddingQualityMetrics()
    result = eq.query_doc_similarity_distribution("", [])
    assert result["diagnostic"] == "No query or documents to analyze"


def test_query_doc_similarity_distribution_without_embedder():
    eq = EmbeddingQualityMetrics(embedding_model=None)
    result = eq.query_doc_similarity_distribution("q", ["d"])
    assert result["requires_embedding"] is True


def test_cluster_compactness_single_item_clusters_are_fully_compact():
    eq = EmbeddingQualityMetrics()
    result = eq.cluster_compactness([["only one sentence here"]])
    assert result["per_cluster_compactness"] == [1.0]


def test_cluster_compactness_similar_texts_score_higher_than_dissimilar():
    eq = EmbeddingQualityMetrics()
    similar_cluster = [
        "The Eiffel Tower is located in Paris.",
        "Paris is home to the Eiffel Tower.",
    ]
    dissimilar_cluster = [
        "The Eiffel Tower is located in Paris.",
        "Bananas are rich in potassium.",
    ]
    similar_result = eq.cluster_compactness([similar_cluster])
    dissimilar_result = eq.cluster_compactness([dissimilar_cluster])
    assert similar_result["mean_compactness"] > dissimilar_result["mean_compactness"]


def test_cluster_compactness_empty():
    eq = EmbeddingQualityMetrics()
    result = eq.cluster_compactness([])
    assert result["diagnostic"] == "No clusters to analyze"


def test_intra_query_stability_needs_at_least_two_paraphrases():
    eq = EmbeddingQualityMetrics()
    result = eq.intra_query_stability(["only one query"])
    assert result["stability_score"] == 1.0


def test_intra_query_stability_similar_paraphrases_are_stable():
    eq = EmbeddingQualityMetrics()
    result = eq.intra_query_stability([
        "When was the Eiffel Tower built?",
        "In what year was the Eiffel Tower constructed?",
    ])
    assert result["stability_score"] > 0.5


def test_cross_encoder_calibration_perfect_correlation():
    result = EmbeddingQualityMetrics.cross_encoder_calibration([0.1, 0.5, 0.9], [0.2, 0.6, 0.95])
    assert result["correlation"] > 0.9


def test_cross_encoder_calibration_length_mismatch_raises():
    with pytest.raises(ValueError):
        EmbeddingQualityMetrics.cross_encoder_calibration([0.1, 0.2], [0.1])


def test_cross_encoder_calibration_no_variance():
    result = EmbeddingQualityMetrics.cross_encoder_calibration([0.5, 0.5], [0.1, 0.9])
    assert result["diagnostic"] == "No variance in scores — correlation is undefined"


def test_evaluate_combines_only_provided_blocks():
    eq = EmbeddingQualityMetrics()
    result = eq.evaluate(query="q", doc_texts=["a document about paris"])
    assert result["similarity_distribution"] is not None
    assert result["cluster_compactness"] is None
    assert result["intra_query_stability"] is None
    assert result["cross_encoder_calibration"] is None
    assert result["diagnostic"]


def test_evaluate_with_nothing_reports_healthy():
    eq = EmbeddingQualityMetrics()
    result = eq.evaluate()
    assert result["diagnostic"] == ["Embedding quality appears healthy across the evaluated dimensions."]
