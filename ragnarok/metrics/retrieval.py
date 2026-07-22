from .._libs import List, np


class RetrievalMetrics:
    @staticmethod
    def hit_rate_at_k(relevance_scores: List[int], k: int) -> float:
        return 1.0 if sum(relevance_scores[:k]) > 0 else 0.0

    @staticmethod
    def precision_at_k(relevance_scores: List[int], k: int) -> float:
        return sum(relevance_scores[:k]) / k if k > 0 else 0.0

    @staticmethod
    def recall_at_k(relevance_scores: List[int], total_relevant: int, k: int) -> float:
        return sum(relevance_scores[:k]) / total_relevant if total_relevant > 0 else 0.0

    @staticmethod
    def ndcg_at_k(relevance_scores: List[int], k: int) -> float:
        dcg = sum((rel / np.log2(i + 2)) for i, rel in enumerate(relevance_scores[:k]))
        ideal = sorted(relevance_scores, reverse=True)[:k]
        idcg = sum((rel / np.log2(i + 2)) for i, rel in enumerate(ideal))
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def mrr_at_k(relevance_scores: List[int], k: int) -> float:
        for i, rel in enumerate(relevance_scores[:k]):
            if rel > 0:
                return 1.0 / (i + 1)
        return 0.0
