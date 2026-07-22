import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingQualityMetrics:
    """
    Metrics for evaluating the quality of embeddings used in retrieval
    """
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedding_model) if embedding_model else None

    def query_doc_similarity_distribution(self, query: str, doc_texts: List[str]) -> Dict[str, Any]:
        """
        Distribution of cosine similarity between the query and the top chunks.
        A large spread (std) may indicate an unstable embedding model.
        """
        if self.embedder is None:
            return {
                'mean_similarity': 0.0, 'std_similarity': 0.0, 'similarities': [],
                'diagnostic': "Embedding model not loaded", 'requires_embedding': True
            }
        if not query or not doc_texts:
            return {
                'mean_similarity': 0.0, 'std_similarity': 0.0, 'similarities': [],
                'diagnostic': "No query or documents to analyze", 'requires_embedding': False
            }

        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        doc_embs = self.embedder.encode(doc_texts, convert_to_numpy=True)
        sims = cosine_similarity(query_emb, doc_embs)[0]

        mean_sim = float(np.mean(sims))
        std_sim = float(np.std(sims))

        if std_sim > 0.25:
            diagnostic = "High similarity spread — embedding model may be unstable for this query"
        elif std_sim > 0.12:
            diagnostic = "Moderate similarity spread across retrieved documents"
        else:
            diagnostic = "Low similarity spread — consistent relevance ranking"

        return {
            'mean_similarity': round(mean_sim, 4),
            'std_similarity': round(std_sim, 4),
            'min_similarity': round(float(np.min(sims)), 4),
            'max_similarity': round(float(np.max(sims)), 4),
            'similarities': [round(float(s), 4) for s in sims],
            'diagnostic': diagnostic,
            'requires_embedding': False
        }

    def cluster_compactness(self, clusters: List[List[str]]) -> Dict[str, Any]:
        """
        How close together the embeddings are within groups of semantically similar documents.
        Low compactness = embeddings poorly separate topics.
        """
        if self.embedder is None:
            return {
                'mean_compactness': 0.0, 'per_cluster_compactness': [],
                'diagnostic': "Embedding model not loaded", 'requires_embedding': True
            }

        clusters = [c for c in clusters if c]
        if not clusters:
            return {
                'mean_compactness': 0.0, 'per_cluster_compactness': [],
                'diagnostic': "No clusters to analyze", 'requires_embedding': False
            }

        compactness_scores = []
        for cluster in clusters:
            if len(cluster) < 2:
                compactness_scores.append(1.0)
                continue
            embs = self.embedder.encode(cluster, convert_to_numpy=True)
            sims = cosine_similarity(embs)
            n = len(cluster)
            off_diagonal_mean = (sims.sum() - np.trace(sims)) / (n * (n - 1))
            compactness_scores.append(float(off_diagonal_mean))

        mean_compactness = float(np.mean(compactness_scores))

        if mean_compactness > 0.7:
            diagnostic = "High cluster compactness — embeddings separate topics well"
        elif mean_compactness > 0.45:
            diagnostic = "Moderate cluster compactness"
        else:
            diagnostic = "Low cluster compactness — embeddings poorly separate semantically similar documents"

        return {
            'mean_compactness': round(mean_compactness, 4),
            'per_cluster_compactness': [round(s, 4) for s in compactness_scores],
            'diagnostic': diagnostic,
            'requires_embedding': False
        }

    def intra_query_stability(self, query_paraphrases: List[str]) -> Dict[str, Any]:
        """
        Variability of embeddings for paraphrases of the same query.
        High variability = embeddings are not invariant to query phrasing.
        """
        if self.embedder is None:
            return {
                'stability_score': 0.0, 'mean_pairwise_similarity': 0.0,
                'diagnostic': "Embedding model not loaded", 'requires_embedding': True
            }
        if len(query_paraphrases) < 2:
            return {
                'stability_score': 1.0, 'mean_pairwise_similarity': 1.0,
                'diagnostic': "Need at least 2 paraphrases to measure stability", 'requires_embedding': False
            }

        embs = self.embedder.encode(query_paraphrases, convert_to_numpy=True)
        sims = cosine_similarity(embs)
        n = len(query_paraphrases)
        off_diagonal_mean = (sims.sum() - np.trace(sims)) / (n * (n - 1))
        stability = float(max(0.0, min(1.0, off_diagonal_mean)))

        if stability > 0.85:
            diagnostic = "High intra-query stability — paraphrases map to similar embeddings"
        elif stability > 0.65:
            diagnostic = "Moderate intra-query stability"
        else:
            diagnostic = "Low intra-query stability — embeddings are sensitive to query phrasing"

        return {
            'stability_score': round(stability, 4),
            'mean_pairwise_similarity': round(stability, 4),
            'diagnostic': diagnostic,
            'requires_embedding': False
        }

    @staticmethod
    def cross_encoder_calibration(embedding_scores: List[float],
                                   cross_encoder_scores: List[float]) -> Dict[str, Any]:
        """
        Correlation between embedding similarity scores and cross-encoder scores.
        Low correlation = embeddings rank documents poorly relative to the more accurate model.
        """
        if len(embedding_scores) != len(cross_encoder_scores):
            raise ValueError("embedding_scores and cross_encoder_scores must have the same length")
        if len(embedding_scores) < 2:
            return {'correlation': 0.0, 'diagnostic': "Need at least 2 scored documents to compute correlation"}

        emb = np.array(embedding_scores, dtype=float)
        cross = np.array(cross_encoder_scores, dtype=float)
        if np.std(emb) == 0 or np.std(cross) == 0:
            return {'correlation': 0.0, 'diagnostic': "No variance in scores — correlation is undefined"}

        correlation = float(np.corrcoef(emb, cross)[0, 1])

        if correlation > 0.7:
            diagnostic = "High calibration — embedding similarity agrees with cross-encoder ranking"
        elif correlation > 0.4:
            diagnostic = "Moderate calibration"
        else:
            diagnostic = "Low calibration — embeddings rank documents poorly compared to the cross-encoder"

        return {'correlation': round(correlation, 4), 'diagnostic': diagnostic}

    def evaluate(self,
                 query: Optional[str] = None,
                 doc_texts: Optional[List[str]] = None,
                 clusters: Optional[List[List[str]]] = None,
                 query_paraphrases: Optional[List[str]] = None,
                 embedding_scores: Optional[List[float]] = None,
                 cross_encoder_scores: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Summary of all metrics — each block is computed only if the required data is provided
        """
        result: Dict[str, Any] = {
            'similarity_distribution': None,
            'cluster_compactness': None,
            'intra_query_stability': None,
            'cross_encoder_calibration': None,
            'diagnostic': []
        }

        if query and doc_texts:
            dist = self.query_doc_similarity_distribution(query, doc_texts)
            result['similarity_distribution'] = dist
            if dist.get('std_similarity', 0) > 0.25:
                result['diagnostic'].append(
                    f"Unstable query-document similarity (std={dist['std_similarity']:.2f})."
                )

        if clusters:
            compactness = self.cluster_compactness(clusters)
            result['cluster_compactness'] = compactness
            if compactness.get('mean_compactness', 1.0) < 0.45:
                result['diagnostic'].append(
                    f"Low cluster compactness ({compactness['mean_compactness']:.2f}) — "
                    "embeddings poorly separate topics."
                )

        if query_paraphrases and len(query_paraphrases) >= 2:
            stability = self.intra_query_stability(query_paraphrases)
            result['intra_query_stability'] = stability
            if stability.get('stability_score', 1.0) < 0.65:
                result['diagnostic'].append(
                    f"Low intra-query stability ({stability['stability_score']:.2f}) — "
                    "consider a more robust embedding model."
                )

        if embedding_scores and cross_encoder_scores:
            calibration = self.cross_encoder_calibration(embedding_scores, cross_encoder_scores)
            result['cross_encoder_calibration'] = calibration
            if calibration.get('correlation', 1.0) < 0.4:
                result['diagnostic'].append(
                    f"Low cross-encoder calibration (r={calibration['correlation']:.2f}) — "
                    "consider adding a reranker."
                )

        if not result['diagnostic']:
            result['diagnostic'].append("Embedding quality appears healthy across the evaluated dimensions.")

        return result
