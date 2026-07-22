from ragnarok import NLIEntailment


def test_faithful_answer_scores_higher_than_hallucinated_answer():
    nli = NLIEntailment(threshold=0.5)
    contexts = ["The Eiffel Tower is in Paris.", "It was built in 1889."]
    faithful_answer = "The Eiffel Tower is in Paris. The Eiffel Tower was built in 1889."
    hallucinated_answer = "The Eiffel Tower is in London."

    faithful_score = nli.score(faithful_answer, contexts)
    hallucinated_score = nli.score(hallucinated_answer, contexts)

    assert faithful_score > hallucinated_score
    assert faithful_score >= 0.5
