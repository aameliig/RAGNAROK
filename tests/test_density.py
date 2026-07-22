from ragnarok import InformationDensityMetric


def test_ids_flags_redundancy_in_repetitive_answer():
    ids = InformationDensityMetric()
    verbose_answer = "It was built in 1889. It was built in 1889. Yes, 1889."
    contexts = ["It was constructed in 1889.", "Its height is 330m."]

    result = ids.score(verbose_answer, contexts)

    assert result["redundancy_ratio"] > 0.0
    assert 0.0 <= result["ids_score"] <= 1.0


def test_ids_empty_answer():
    ids = InformationDensityMetric()
    result = ids.score("", ["some context"])
    assert result["ids_score"] == 0.0
