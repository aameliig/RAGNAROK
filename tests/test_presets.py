import pytest

from ragnarok import EvaluationConfig, PRESETS, list_presets, load_preset, suggest_preset, create_custom_preset


def test_list_presets_returns_all_names_with_descriptions():
    presets = list_presets()
    assert set(presets) == set(PRESETS)
    assert all(isinstance(desc, str) and desc for desc in presets.values())


@pytest.mark.parametrize("name", list(PRESETS))
def test_load_preset_returns_usable_config(name):
    config = load_preset(name)
    assert isinstance(config, EvaluationConfig)
    assert config.metrics

    final_score, penalty, status = config.aggregate(
        {m: 0.8 for m in config.metrics}, latency_ms=100
    )
    assert 0.0 <= final_score <= 100.0
    assert status in {"PASSED", "FAILED", "GUARDRAIL_VIOLATION"}


def test_load_preset_unknown_name_raises():
    with pytest.raises(ValueError):
        load_preset("does_not_exist")


def test_suggest_preset_safety_keywords():
    assert suggest_preset("I don't want the model to hallucinate, even if it's slower") == "safety_first"


def test_suggest_preset_speed_keywords():
    assert suggest_preset("We need very low latency, real-time responses") == "speed_first"


def test_suggest_preset_cost_keywords():
    assert suggest_preset("Keep it as cheap and budget-friendly as possible") == "cost_saving"


def test_suggest_preset_falls_back_to_balanced_when_no_keywords_match():
    assert suggest_preset("xyzzy plugh qwerty") == "balanced"


def test_suggest_preset_medical_keywords():
    assert suggest_preset("This is for a hospital patient diagnosis assistant") == "medical"


def test_suggest_preset_customer_support_keywords():
    assert suggest_preset("A support chatbot for our helpdesk tickets") == "customer_support"


def test_suggest_preset_research_keywords():
    assert suggest_preset("An academic research literature review tool") == "research"


def test_load_preset_returns_independent_copies():
    """Mutating a loaded config must not corrupt the shared PRESETS singleton."""
    cfg1 = load_preset("balanced")
    cfg1.metrics["sfc"].weight = 0.99

    cfg2 = load_preset("balanced")
    assert cfg2.metrics["sfc"].weight != 0.99


def test_create_custom_preset_normalizes_weights_to_one():
    config = create_custom_preset(speed=0.5, cost=0.1, accuracy=0.3, coverage=0.1)
    total = sum(m.weight for m in config.metrics.values() if m.enabled)
    assert total == pytest.approx(1.0, abs=0.01)


def test_create_custom_preset_speed_priority_dominates_latency_weight():
    speed_heavy = create_custom_preset(speed=0.9, cost=0.03, accuracy=0.04, coverage=0.03)
    accuracy_heavy = create_custom_preset(speed=0.03, cost=0.03, accuracy=0.9, coverage=0.04)
    assert speed_heavy.metrics["latency_score"].weight > accuracy_heavy.metrics["latency_score"].weight
    assert accuracy_heavy.metrics["faithfulness"].weight > speed_heavy.metrics["faithfulness"].weight


def test_create_custom_preset_unknown_base_raises():
    with pytest.raises(ValueError):
        create_custom_preset(base="does_not_exist")


def test_create_custom_preset_zero_priorities_falls_back_to_equal_split():
    config = create_custom_preset(speed=0, cost=0, accuracy=0, coverage=0)
    assert config.metrics["latency_score"].weight > 0
