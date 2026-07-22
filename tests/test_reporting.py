import pytest

from ragnarok import BaseLLM, LLMReportGenerator, StructuredReportGenerator


class FakeLLM(BaseLLM):
    """Stub LLM so report generation can be tested without a real API key or network call."""

    def __init__(self):
        self.last_prompt = None

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        self.last_prompt = prompt
        return "## Summary\nEverything looks fine."

    def name(self) -> str:
        return "fake-llm"


def test_generate_feeds_summarized_metrics_to_the_llm():
    llm = FakeLLM()
    generator = LLMReportGenerator(llm)

    result = {
        "final_score": 85.3,
        "status": "PASSED",
        "fault_type": "healthy",
        "fault_subtype": None,
        "fault_explanation": "All metrics within acceptable thresholds",
        "metrics": {"faithfulness": 0.9, "sfc": 0.5, "chunking_quality": {"overall_grade": "GOOD"}},
        "data_availability": {"diagnostic": ["The database appears sufficient and fresh."]},
        "chunking_quality": {"diagnostics": ["Chunking is performing well."]},
        "business_metrics": {"cost_score": 100.0, "trust_score": 0.8},
    }

    report = generator.generate(result)

    assert report == "## Summary\nEverything looks fine."
    assert "85.3" in llm.last_prompt
    assert "healthy" in llm.last_prompt
    # non-numeric metric values (e.g. the nested chunking_quality dict) must be filtered out
    assert "overall_grade" not in llm.last_prompt


def test_create_raises_for_unknown_provider():
    with pytest.raises(ValueError):
        LLMReportGenerator.create("not-a-real-provider", "some-model")


FAILING_RESULT = {
    "final_score": 45.0,
    "status": "FAILED",
    "fault_type": "generation",
    "fault_subtype": "hallucination",
    "fault_confidence": 0.9,
    "fault_explanation": "Severe hallucination detected",
    "recommended_action": "Strengthen prompt grounding",
    "metrics": {"faithfulness": 0.2, "sfc": 0.3, "ids": 0.1},
    "business_metrics": {"cost_score": 90, "latency_grade": "GOOD", "trust_score": 0.4},
}

PASSING_RESULT = {
    "final_score": 92.0,
    "status": "PASSED",
    "fault_type": "healthy",
    "fault_subtype": None,
    "fault_confidence": 1.0,
    "fault_explanation": "All metrics within acceptable thresholds",
    "recommended_action": "No action needed",
    "metrics": {"faithfulness": 0.95, "sfc": 0.9, "ids": 0.6},
    "business_metrics": {"cost_score": 95, "latency_grade": "EXCELLENT", "trust_score": 0.9},
}


def test_structured_report_requires_no_llm_and_reflects_failure():
    report = StructuredReportGenerator().generate(FAILING_RESULT)

    assert report.overall_score == 45.0
    assert report.status == "FAILED"
    assert report.fault_type == "generation"
    titles = [s.title for s in report.sections]
    assert "Executive Summary" in titles
    assert "Action Items" in titles

    executive_summary = next(s for s in report.sections if s.title == "Executive Summary")
    assert executive_summary.severity == "critical"


def test_structured_report_passing_result_has_info_severity_summary():
    report = StructuredReportGenerator().generate(PASSING_RESULT)
    executive_summary = next(s for s in report.sections if s.title == "Executive Summary")
    assert executive_summary.severity == "info"


def test_structured_report_includes_retrieval_pipeline_section_when_given():
    pipeline_report = {
        "overall_passed": False,
        "bottleneck_stage": "embedding_quality",
        "bottleneck_explanation": "Bottleneck at 'Stage 2: Embedding quality': unstable similarity.",
        "stages": [
            {"stage_name": "Stage 0: Data availability", "passed": True, "summary": "OK"},
            {"stage_name": "Stage 2: Embedding quality", "passed": False, "summary": "Unstable"},
        ],
    }
    report = StructuredReportGenerator().generate(FAILING_RESULT, retrieval_pipeline_report=pipeline_report)
    titles = [s.title for s in report.sections]
    assert "Retrieval Pipeline Diagnostics" in titles

    retrieval_section = next(s for s in report.sections if s.title == "Retrieval Pipeline Diagnostics")
    assert retrieval_section.severity == "critical"
    assert retrieval_section.metrics["bottleneck"] == "embedding_quality"


def test_structured_report_to_dict_roundtrips_all_sections():
    report = StructuredReportGenerator().generate(FAILING_RESULT)
    as_dict = report.to_dict()
    assert as_dict["overall_score"] == 45.0
    assert len(as_dict["sections"]) == len(report.sections)


def test_structured_report_to_markdown_contains_key_facts():
    md = StructuredReportGenerator().generate(FAILING_RESULT).to_markdown()
    assert "45.0/100" in md
    assert "GENERATION" in md
    assert "# RAGNAROK Evaluation Report" in md


def test_structured_report_to_html_contains_key_facts():
    html = StructuredReportGenerator().generate(FAILING_RESULT).to_html()
    assert "45.0/100" in html
    assert "<table>" in html
