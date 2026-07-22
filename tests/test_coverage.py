from ragnarok import SemanticFootprintCoverage


def test_sfc_score_partial_coverage():
    sfc = SemanticFootprintCoverage(coverage_threshold=0.55)
    answer = "The Eiffel Tower was built in 1889."
    contexts = ["It was constructed in 1889.", "Its height is 330m."]

    result = sfc.score(answer, contexts)

    assert 0.0 <= result["sfc_score"] <= 1.0
    assert result["total_clusters"] > 0


def test_sfc_score_empty_inputs():
    sfc = SemanticFootprintCoverage()
    assert sfc.score("", [])["sfc_score"] == 0.0
