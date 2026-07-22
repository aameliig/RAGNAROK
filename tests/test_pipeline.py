from ragnarok import RetrievalPipelineEvaluator, PipelineStage

GOOD_CONTEXTS = [
    "The Eiffel Tower was built in 1889 for the World's Fair in Paris.",
    "It was designed by the engineer Gustave Eiffel.",
]


def test_pipeline_all_stages_pass_with_healthy_data():
    pe = RetrievalPipelineEvaluator()
    report = pe.evaluate_pipeline(
        query="When was the Eiffel Tower built?",
        contexts=GOOD_CONTEXTS,
        relevance_scores=[1, 1],
    )
    assert len(report.stages) == 5
    assert {s.stage for s in report.stages} == set(PipelineStage)


def test_pipeline_reports_no_bottleneck_when_all_pass():
    pe = RetrievalPipelineEvaluator(max_empty_result_rate=1.0, min_hit_rate=0.0, min_ndcg=0.0)
    report = pe.evaluate_pipeline(
        query="When was the Eiffel Tower built?",
        contexts=GOOD_CONTEXTS,
        relevance_scores=[1, 1],
    )
    assert report.overall_passed is True
    assert report.bottleneck_stage is None
    assert report.bottleneck_explanation == "All stages passed."


def test_pipeline_flags_empty_contexts_as_data_availability_bottleneck():
    pe = RetrievalPipelineEvaluator()
    report = pe.evaluate_pipeline(query="anything?", contexts=[], relevance_scores=None)
    assert report.overall_passed is False
    assert report.bottleneck_stage == PipelineStage.DATA_AVAILABILITY


def test_pipeline_stage_3_is_marked_not_evaluated_without_relevance_scores():
    pe = RetrievalPipelineEvaluator()
    report = pe.evaluate_pipeline(query="q", contexts=GOOD_CONTEXTS, relevance_scores=None)
    stage_3 = next(s for s in report.stages if s.stage == PipelineStage.RETRIEVAL_QUALITY)
    assert stage_3.passed is True
    assert "not evaluated" in stage_3.summary


def test_pipeline_stage_4_is_marked_not_evaluated_without_before_rerank_data():
    pe = RetrievalPipelineEvaluator()
    report = pe.evaluate_pipeline(
        query="q", contexts=GOOD_CONTEXTS, relevance_scores=[1, 0], relevance_before_rerank=None
    )
    stage_4 = next(s for s in report.stages if s.stage == PipelineStage.RERANKING_QUALITY)
    assert stage_4.passed is True
    assert "not evaluated" in stage_4.summary


def test_pipeline_report_to_dict_is_json_serializable():
    import json

    pe = RetrievalPipelineEvaluator()
    report = pe.evaluate_pipeline(query="q", contexts=GOOD_CONTEXTS, relevance_scores=[1, 1])
    serialized = json.dumps(report.to_dict())
    assert "bottleneck_stage" in serialized
