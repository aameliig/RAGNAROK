from ragnarok import ChunkingQualityMetrics

GOOD_CHUNKS = [
    "In 1889, the World's Fair was held in Paris. The Eiffel Tower was built for this event.",
    "The Eiffel Tower was originally meant to be temporary. However, it became a symbol of Paris.",
    "The tower is 330 meters tall. This makes it the tallest structure in Paris.",
]

POOR_CHUNKS = [
    "In 1889, the World's Fair",
    "was held in Paris. For this event",
    "the Eiffel Tower was built. The Eiffel Tower was originally meant to be",
    "temporary. However, it became a symbol of Paris. The tower is 330",
    "meters tall. This makes it the tallest structure in Paris.",
]


def test_boundary_coherence_good_chunks():
    result = ChunkingQualityMetrics.boundary_coherence(GOOD_CHUNKS)
    assert result["coherent_fraction"] == 1.0


def test_boundary_coherence_empty_is_vacuously_coherent():
    result = ChunkingQualityMetrics.boundary_coherence([])
    assert result["coherent_fraction"] == 1.0


def test_boundary_coherence_poor_chunks_score_lower():
    good = ChunkingQualityMetrics.boundary_coherence(GOOD_CHUNKS)
    poor = ChunkingQualityMetrics.boundary_coherence(POOR_CHUNKS)
    assert poor["coherent_fraction"] < good["coherent_fraction"]


def test_chunk_size_variance_empty():
    result = ChunkingQualityMetrics.chunk_size_variance([])
    assert result["diagnostic"] == "No chunks to analyze"


def test_chunk_size_variance_stable_sizes():
    chunks = ["one two three four five"] * 5
    result = ChunkingQualityMetrics.chunk_size_variance(chunks)
    assert result["cv"] == 0.0
    assert result["diagnostic"] == "Good chunk size stability (low variability)"


def test_informative_token_ratio_empty():
    result = ChunkingQualityMetrics.informative_token_ratio([])
    assert result == {'mean_itr': 0.0, 'per_chunk_itr': [], 'diagnostic': "No chunks to analyze"}


def test_informative_token_ratio_returns_fraction_per_chunk():
    result = ChunkingQualityMetrics.informative_token_ratio(GOOD_CHUNKS)
    assert 0.0 <= result["mean_itr"] <= 1.0
    assert len(result["per_chunk_itr"]) == len(GOOD_CHUNKS)


def test_semantic_cohesion_without_embedder_reports_requires_embedding():
    metrics = ChunkingQualityMetrics(embedding_model=None)
    result = metrics.semantic_cohesion(GOOD_CHUNKS)
    assert result["requires_embedding"] is True
    assert result["diagnostic"] == "Embedding model not loaded"


def test_semantic_cohesion_empty_chunks():
    metrics = ChunkingQualityMetrics()
    result = metrics.semantic_cohesion([])
    assert result["diagnostic"] == "No chunks to analyze"


def test_semantic_cohesion_with_embedder():
    metrics = ChunkingQualityMetrics()
    result = metrics.semantic_cohesion(GOOD_CHUNKS)
    assert result["requires_embedding"] is False
    assert 0.0 <= result["mean_cohesion"] <= 1.0


def test_evaluate_empty_chunks_reports_no_data():
    metrics = ChunkingQualityMetrics()
    result = metrics.evaluate([])
    assert result["overall_grade"] == "NO_DATA"


def test_evaluate_good_chunks_score_higher_than_poor_chunks():
    metrics = ChunkingQualityMetrics()
    good = metrics.evaluate(GOOD_CHUNKS)
    poor = metrics.evaluate(POOR_CHUNKS)

    assert good["overall_score"] > poor["overall_score"]
    assert set(good.keys()) >= {
        "boundary", "variance", "informative_ratio", "cohesion",
        "overall_score", "overall_grade", "diagnostics", "recommendations",
    }
    assert good["diagnostics"]
    assert good["recommendations"]
