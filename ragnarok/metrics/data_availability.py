import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional
from datetime import datetime

class DataAvailabilityMetrics:
    """
    Metrics for evaluating data availability and freshness in the vector store
    """
    def __init__(self):
        pass

    @staticmethod
    def empty_result_rate(num_chunks_found: List[int]) -> float:
        """
        Fraction of queries for which no chunk was found
        """
        if not num_chunks_found:
            return 0.0
        empty_count = sum(1 for n in num_chunks_found if n == 0)
        return empty_count / len(num_chunks_found)

    @staticmethod
    def coverage_gap(
        retrieved_doc_ids: List[List[str]],
        relevant_doc_ids: List[List[str]]
    ) -> float:
        """
        Fraction of queries for which not all relevant documents were found
        """
        if len(retrieved_doc_ids) != len(relevant_doc_ids):
            raise ValueError("retrieved_doc_ids and relevant_doc_ids must have the same length")

        missing_count = 0
        total = len(relevant_doc_ids)

        for ret_docs, rel_docs in zip(retrieved_doc_ids, relevant_doc_ids):
            if not rel_docs:
                continue
            all_present = all(doc_id in ret_docs for doc_id in rel_docs)
            if not all_present:
                missing_count += 1

        return missing_count / total if total > 0 else 0.0

    @staticmethod
    def index_freshness(
        document_timestamps: List[datetime],
        max_age_days: int = 30
    ) -> Dict[str, Any]:
        """
        Average document age and fraction of stale documents
        """
        if not document_timestamps:
            return {
                'average_age_days': 0.0,
                'max_age_days': 0.0,
                'min_age_days': 0.0,
                'stale_ratio': 0.0,
                'freshness_score': 100.0
            }

        now = datetime.now()
        ages = [(now - ts).total_seconds() / 86400 for ts in document_timestamps]

        avg_age = np.mean(ages)
        max_age = np.max(ages)
        min_age = np.min(ages)
        stale_count = sum(1 for age in ages if age > max_age_days)
        stale_ratio = stale_count / len(ages)

        if avg_age <= max_age_days:
            freshness = 100.0
        elif avg_age >= 2 * max_age_days:
            freshness = 0.0
        else:
            freshness = 100.0 * (1 - (avg_age - max_age_days) / max_age_days)

        return {
            'average_age_days': round(avg_age, 2),
            'max_age_days': round(max_age, 2),
            'min_age_days': round(min_age, 2),
            'stale_ratio': round(stale_ratio, 4),
            'freshness_score': round(freshness, 2)
        }

    @staticmethod
    def detect_missing_documents(
        corpus: List[str],
        queries: List[str],
        threshold: float = 0.1,
        use_idf: bool = True
    ) -> Dict[str, Any]:
        """
        Queries for which the corpus has no relevant documents (TF-IDF)
        """
        if not corpus or not queries:
            return {
                'missing_ratio': 0.0,
                'missing_queries': [],
                'max_similarities': [],
                'avg_similarity': 0.0,
                'diagnostic': "Corpus or queries are empty"
            }

        # use_idf - whether to use IDF (affects the weight of rare words)
        vectorizer = TfidfVectorizer(stop_words='english', use_idf=use_idf)
        corpus_tfidf = vectorizer.fit_transform(corpus)
        query_tfidf = vectorizer.transform(queries)

        similarities = cosine_similarity(query_tfidf, corpus_tfidf)

        max_sims = np.max(similarities, axis=1).flatten().tolist()
        missing_indices = [i for i, sim in enumerate(max_sims) if sim < threshold]
        missing_queries = [queries[i] for i in missing_indices]

        missing_ratio = len(missing_queries) / len(queries) if queries else 0.0
        avg_sim = np.mean(max_sims) if max_sims else 0.0

        if missing_ratio == 0:
            diagnostic = "All queries have at least one relevant document in the corpus."
        elif missing_ratio < 0.1:
            diagnostic = f"A small number of queries ({missing_ratio*100:.1f}%) lack relevant documents."
        elif missing_ratio < 0.3:
            diagnostic = f"A significant proportion of queries ({missing_ratio*100:.1f}%) are not covered by the corpus."
        else:
            diagnostic = f"The majority of queries ({missing_ratio*100:.1f}%) have no relevant documents — the corpus is incomplete."
        
        return {
            'missing_ratio': round(missing_ratio, 4),
            'missing_queries': missing_queries,
            'max_similarities': [round(s, 4) for s in max_sims],
            'avg_similarity': round(avg_sim, 4),
            'diagnostic': diagnostic
        }

    def evaluate(
        self,
        queries: List[str],
        num_chunks_found: List[int],
        retrieved_doc_ids: Optional[List[List[str]]] = None,
        relevant_doc_ids: Optional[List[List[str]]] = None,
        document_timestamps: Optional[List[datetime]] = None,
        max_age_days: int = 30,
        corpus: Optional[List[str]] = None,
        tfidf_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Summary of all metrics
        """
        result = {
            'empty_result_rate': self.empty_result_rate(num_chunks_found),
            'coverage_gap': None,
            'index_freshness': None,
            'missing_docs_tfidf': None,
            'diagnostic': []
        }

        # Empty results
        if result['empty_result_rate'] > 0.05:
            result['diagnostic'].append(
                f"High proportion of empty results: {result['empty_result_rate']*100:.1f}% "
                "of queries find no chunks."
            )

        # Coverage Gap (ground truth)
        if retrieved_doc_ids is not None and relevant_doc_ids is not None:
            result['coverage_gap'] = self.coverage_gap(
                retrieved_doc_ids, relevant_doc_ids
            )
            if result['coverage_gap'] > 0.1:
                result['diagnostic'].append(
                    f"Insufficient database coverage: {result['coverage_gap']*100:.1f}% "
                    "of queries lack relevant documents in the index."
                )

        # Index freshness
        if document_timestamps:
            freshness = self.index_freshness(document_timestamps, max_age_days)
            result['index_freshness'] = freshness
            if freshness['stale_ratio'] > 0.2:
                result['diagnostic'].append(
                    f"Index is stale: {freshness['stale_ratio']*100:.1f}% of documents are older "
                    f"than {max_age_days} days. Average age: {freshness['average_age_days']} days."
                )

        # Missing document detection (TF‑IDF)
        if corpus is not None:
            missing = self.detect_missing_documents(
                corpus, queries, threshold=tfidf_threshold
            )
            result['missing_docs_tfidf'] = missing
            if missing['missing_ratio'] > 0.1:
                result['diagnostic'].append(
                    f"Auto-detection: {missing['missing_ratio']*100:.1f}% of queries have no "
                    f"relevant documents in the corpus (average similarity: {missing['avg_similarity']:.2f})."
                )

        # If no diagnostic messages, everything is fine
        if not result['diagnostic']:
            result['diagnostic'].append("The database appears sufficient and fresh.")

        return result
