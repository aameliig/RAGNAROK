"""
Run the synthetic test suite
Usage: python run_synthetic.py
"""
from ragnarok import RAGEvaluator, EvaluationConfig
from synthetic_test_cases import SYNTHETIC_CASES


def run_synthetic_tests():
    print("\n" + "=" * 70)
    print("RAGNAROK Synthetic Test Suite — 30 Cases")
    print("=" * 70 + "\n")

    # Load the config
    config = EvaluationConfig.from_yaml("config.yaml")
    evaluator = RAGEvaluator(config)

    results_summary = []

    for i, case in enumerate(SYNTHETIC_CASES, 1):
        print(f"\n--- Case {i}/30: {case['name']} ---")
        print(f"Description: {case['description']}")

        result = evaluator.evaluate(
            query=case["query"],
            answer=case["answer"],
            contexts=case["contexts"],
            relevance_scores=case.get("relevance_scores"),
            latency_ms=case.get("latency_ms", 0),
            input_tokens=case.get("input_tokens", 0),
            output_tokens=case.get("output_tokens", 0),
            num_chunks_retrieved=case.get("num_chunks_retrieved", 0),
            has_citations=case.get("has_citations", False)
        )

        expected = case.get("expected_fault", "unknown")
        actual = result["fault_type"]
        match = "✅" if expected == actual else "❌"

        print(f"  Expected fault: {expected}")
        print(f"  Actual fault:   {actual} {match}")
        print(f"  Business Score: {result['final_score']:.1f}")
        print(f"  Faithfulness:   {result['metrics'].get('faithfulness', 0):.2f}")
        print(f"  SFC:            {result['metrics'].get('sfc', 0):.2f}")
        print(f"  IDS:            {result['metrics'].get('ids', 0):.4f}")
        print(f"  Trust Score:    {result['business_metrics'].get('trust_score', 0):.2f}")
        print(f"  Cost Score:     {result['business_metrics'].get('cost_score', 0):.1f}")
        print(f"  Latency Grade:  {result['business_metrics'].get('latency_grade', 'N/A')}")

        results_summary.append({
            "name": case["name"],
            "expected": expected,
            "actual": actual,
            "match": expected == actual,
            "score": result["final_score"],
            "faithfulness": result["metrics"].get("faithfulness", 0),
            "sfc": result["metrics"].get("sfc", 0)
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total = len(results_summary)
    correct = sum(1 for r in results_summary if r["match"])
    print(f"\nTotal cases: {total}")
    print(f"Correct diagnoses: {correct}/{total} ({correct/total*100:.1f}%)")

    print("\nFailed cases:")
    for r in results_summary:
        if not r["match"]:
            print(f"  ❌ {r['name']}: expected {r['expected']}, got {r['actual']}")

    print("\nScore distribution:")
    passed = sum(1 for r in results_summary if r["score"] >= 60)
    failed = total - passed
    print(f"  PASSED (>=60): {passed}")
    print(f"  FAILED (<60):  {failed}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_synthetic_tests()
