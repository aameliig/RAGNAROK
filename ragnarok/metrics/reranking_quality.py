from typing import List, Dict, Any, Optional

from .retrieval import RetrievalMetrics


class RerankingQualityMetrics:
    """
    Metrics for evaluating reranking effectiveness — comparing ranking before/after
    """
    @staticmethod
    def reranking_lift(relevance_before: List[int], relevance_after: List[int], k: int = 5) -> Dict[str, Any]:
        """
        MRR@k gain after reranking < 0.02 — the reranker is useless or harmful.
        """
        mrr_before = RetrievalMetrics.mrr_at_k(relevance_before, k)
        mrr_after = RetrievalMetrics.mrr_at_k(relevance_after, k)
        lift = mrr_after - mrr_before

        if lift < 0.02:
            diagnostic = "Reranker provides negligible or negative lift — consider disabling it"
        elif lift < 0.1:
            diagnostic = "Reranker provides a modest improvement in ranking"
        else:
            diagnostic = "Reranker meaningfully improves ranking quality"

        return {
            'mrr_before': round(mrr_before, 4),
            'mrr_after': round(mrr_after, 4),
            'lift': round(lift, 4),
            'diagnostic': diagnostic
        }

    @staticmethod
    def relevance_order_accuracy(relevance_scores: List[int]) -> Dict[str, Any]:
        """
        Fraction of document pairs whose final order (after reranking)
        does not contradict their relevance. < 70% — the reranker is unreliable.
        """
        n = len(relevance_scores)
        if n < 2:
            return {
                'accuracy': 1.0, 'correct_pairs': 0, 'total_pairs': 0,
                'diagnostic': "Not enough documents to evaluate pairwise order"
            }

        correct = 0
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += 1
                # position i is ranked above position j — its relevance should not be lower
                if relevance_scores[i] >= relevance_scores[j]:
                    correct += 1

        accuracy = correct / total if total > 0 else 1.0

        if accuracy < 0.7:
            diagnostic = "Low relevance order accuracy — reranker output is unreliable"
        elif accuracy < 0.9:
            diagnostic = "Moderate relevance order accuracy"
        else:
            diagnostic = "High relevance order accuracy — ranking respects relevance grades"

        return {
            'accuracy': round(accuracy, 4),
            'correct_pairs': correct,
            'total_pairs': total,
            'diagnostic': diagnostic
        }

    @staticmethod
    def _first_relevant_index(scores: List[int]) -> Optional[int]:
        for i, s in enumerate(scores):
            if s > 0:
                return i
        return None

    @classmethod
    def first_relevant_position_shift(cls, relevance_before: List[int],
                                       relevance_after: List[int]) -> Dict[str, Any]:
        """
        Shift in the position of the first relevant document after reranking.
        A positive shift (toward the top of the list) means the reranker helps.
        """
        pos_before = cls._first_relevant_index(relevance_before)
        pos_after = cls._first_relevant_index(relevance_after)

        if pos_before is None or pos_after is None:
            return {
                'shift': 0, 'position_before': pos_before, 'position_after': pos_after,
                'diagnostic': "No relevant document found before or after reranking"
            }

        shift = pos_before - pos_after

        if shift > 0:
            diagnostic = f"Reranker moved the first relevant document {shift} position(s) closer to the top"
        elif shift < 0:
            diagnostic = f"Reranker pushed the first relevant document {-shift} position(s) further down"
        else:
            diagnostic = "Reranker did not change the position of the first relevant document"

        return {
            'shift': shift,
            'position_before': pos_before,
            'position_after': pos_after,
            'diagnostic': diagnostic
        }

    def evaluate(self, relevance_before: List[int], relevance_after: List[int], k: int = 5) -> Dict[str, Any]:
        """
        Summary of all metrics
        """
        lift_res = self.reranking_lift(relevance_before, relevance_after, k)
        order_res = self.relevance_order_accuracy(relevance_after)
        shift_res = self.first_relevant_position_shift(relevance_before, relevance_after)

        diagnostics = []
        recommendations = []

        if lift_res['lift'] < 0.02:
            diagnostics.append(f"Reranking lift is negligible (Δ MRR={lift_res['lift']:.3f}).")
            recommendations.append("Consider disabling the reranker or fine-tuning it on your domain.")
        else:
            diagnostics.append(f"Reranking lift is positive (Δ MRR={lift_res['lift']:.3f}).")

        if order_res['accuracy'] < 0.7:
            diagnostics.append(f"Low relevance order accuracy ({order_res['accuracy']*100:.1f}%).")
            recommendations.append("Reranker output is unreliable — verify the cross-encoder or its training data.")
        else:
            diagnostics.append(f"Relevance order accuracy is {order_res['accuracy']*100:.1f}%.")

        diagnostics.append(shift_res['diagnostic'])

        if lift_res['lift'] >= 0.02 and order_res['accuracy'] >= 0.7:
            overall = "EFFECTIVE"
        elif lift_res['lift'] > 0 or order_res['accuracy'] >= 0.5:
            overall = "MARGINAL"
        else:
            overall = "INEFFECTIVE"

        if overall == "INEFFECTIVE":
            recommendations.append("Reranking pipeline is not adding value — investigate or remove it.")

        return {
            'lift': lift_res,
            'order_accuracy': order_res,
            'position_shift': shift_res,
            'overall_grade': overall,
            'diagnostics': diagnostics,
            'recommendations': recommendations
        }
