from .._libs import re, List, Dict, SentenceTransformer, DBSCAN, cosine_similarity


class SemanticFootprintCoverage:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2",
                 cluster_eps: float = 0.3, min_samples: int = 1,
                 coverage_threshold: float = 0.55):
        self.embedder = SentenceTransformer(embedding_model)
        self.cluster_eps = cluster_eps
        self.min_samples = min_samples
        self.coverage_threshold = coverage_threshold

    def _extract_claims(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if len(s) > 10]

    def _cluster_claims(self, claims: List[str]) -> List[List[str]]:
        if len(claims) < 2:
            return [claims] if claims else []
        embeddings = self.embedder.encode(claims)
        labels = DBSCAN(eps=self.cluster_eps, min_samples=self.min_samples,
                        metric='cosine').fit_predict(embeddings)
        clusters = {}
        for i, (claim, label) in enumerate(zip(claims, labels)):
            cluster_key = f'noise_{i}' if label == -1 else label
            clusters.setdefault(cluster_key, []).append(claim)
        return list(clusters.values())

    def _claim_in_cluster(self, claim: str, cluster_claims: List[str]) -> bool:
        claim_emb = self.embedder.encode([claim])
        cluster_embs = self.embedder.encode(cluster_claims)
        similarities = cosine_similarity(claim_emb, cluster_embs)
        return similarities[0].max() >= self.coverage_threshold

    def score(self, answer: str, contexts: List[str]) -> Dict:
        if not answer or not contexts:
            return {'sfc_score': 0.0, 'total_clusters': 0, 'covered_clusters': 0, 'weighted_coverage': 0.0}
        context_claims = self._extract_claims(" ".join(contexts))
        answer_claims = self._extract_claims(answer)
        if not context_claims:
            return {'sfc_score': 1.0, 'total_clusters': 0, 'covered_clusters': 0, 'weighted_coverage': 0.0}
        if not answer_claims:
            return {'sfc_score': 0.0, 'total_clusters': 0, 'covered_clusters': 0, 'weighted_coverage': 0.0}

        clusters = self._cluster_claims(context_claims)

        # weighted coverage by cluster size
        # larger clusters (more information) matter more than smaller ones
        total_claims = len(context_claims)
        cluster_weights = [len(c) / total_claims for c in clusters] if total_claims > 0 else [1.0 / len(
            clusters)] * len(clusters)

        covered = 0
        weighted_covered = 0.0
        for i, c in enumerate(clusters):
            is_covered = any(self._claim_in_cluster(a, c) for a in answer_claims)
            if is_covered:
                covered += 1
                weighted_covered += cluster_weights[i]

        # use weighted coverage instead of a simple count
        total_weight = sum(cluster_weights)
        sfc = weighted_covered / total_weight if total_weight > 0 else 0.0

        # keep score within 0.0-1.0
        sfc = max(0.0, min(1.0, sfc))

        # if weighted coverage is too optimistic,
        # take the minimum between weighted and simple coverage
        simple_sfc = covered / len(clusters) if clusters else 0.0
        sfc = min(sfc, simple_sfc * 1.2)  # no more than 20% above simple coverage

        return {'sfc_score': sfc,
                'total_clusters': len(clusters),
                'covered_clusters': covered,
                'weighted_coverage': weighted_covered}