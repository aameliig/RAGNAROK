from .._libs import Dict, List

try:
    from ragas.metrics import context_precision, answer_relevancy
    from ragas import evaluate as ragas_evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


class RAGASBridge:
    def __init__(self):
        if not RAGAS_AVAILABLE:
            raise ImportError(
                "RAGAS not installed. Run: pip install ragas. "
                "Set ragas_enabled=false in config to skip."
            )
        self._metrics = {
            "context_precision": context_precision,
            "answer_relevancy": answer_relevancy,
        }

    def score(self, query: str, answer: str, contexts: List[str]) -> Dict[str, float]:
        from datasets import Dataset
        data = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
        }
        dataset = Dataset.from_dict(data)
        metrics_to_run = [self._metrics[m] for m in self._metrics.keys()]
        result = ragas_evaluate(dataset=dataset, metrics=metrics_to_run)

        scores = {}
        for metric_name in self._metrics.keys():
            scores[f"ragas_{metric_name}"] = result[metric_name][0]
        return scores

    @staticmethod
    def is_available() -> bool:
        return RAGAS_AVAILABLE
