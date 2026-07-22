from ragnarok import FaultClassifier, FaultType


def test_diagnose_healthy_metrics():
    fc = FaultClassifier()
    metrics = {"sfc": 0.9, "ids": 0.5, "faithfulness": 0.9}
    diagnosis = fc.diagnose("Where is the Eiffel tower?", "a", ["Something about Paris."], None, metrics)
    assert diagnosis.fault_type == FaultType.HEALTHY


def test_diagnose_hallucination_from_low_faithfulness():
    fc = FaultClassifier()
    metrics = {"sfc": 0.2, "ids": 0.5, "faithfulness": 0.2}
    diagnosis = fc.diagnose("Where is the Eiffel tower?", "a", ["Something about Paris."], None, metrics)
    assert diagnosis.fault_type == FaultType.GENERATION


def test_diagnose_empty_context_is_out_of_scope():
    fc = FaultClassifier()
    metrics = {"sfc": 0.9, "ids": 0.5, "faithfulness": 0.9}
    diagnosis = fc.diagnose("Where is the Eiffel tower?", "a", [], None, metrics)
    assert diagnosis.fault_type == FaultType.OUT_OF_SCOPE


def test_diagnose_empty_query_is_unknown():
    fc = FaultClassifier()
    diagnosis = fc.diagnose("", "a", ["ctx"], None, {})
    assert diagnosis.fault_type == FaultType.UNKNOWN
