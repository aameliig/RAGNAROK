from .._libs import List, Optional, Dict, Tuple, Enum, dataclass


class FaultType(Enum):
    RETRIEVAL = "retrieval"
    CHUNKING = "chunking"
    GENERATION = "generation"
    OUT_OF_SCOPE = "out_of_scope"
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    BUSINESS = "business"


class FaultSubtype(Enum):
    EMBEDDING_QUALITY = "embedding_quality"
    CHUNKING_STRATEGY = "chunking_strategy"
    RANKER_QUALITY = "ranker_quality"
    QUERY_UNDERSTANDING = "query_understanding"
    HALLUCINATION = "hallucination"
    CONTRADICTION = "contradiction"
    OMISSION = "omission"
    REDUNDANCY = "redundancy"
    COST_INEFFICIENT = "cost_inefficient"
    LATENCY_CRITICAL = "latency_critical"
    LOW_TRUST = "low_trust"
    SCOPE_MISMATCH = "scope_mismatch"


@dataclass
class FaultDiagnosis:
    fault_type: FaultType
    fault_subtype: Optional[FaultSubtype]
    confidence: float
    explanation: str
    recommended_action: str


class FaultClassifier:
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or {
            'hit_rate': 0.70, 'ndcg': 0.65, 'sfc': 0.60,
            'ids': 0.30, 'faithfulness': 0.70,
            'cost_score': 50.0, 'trust_score': 0.65,
            'coverage_efficiency': 0.15, 'latency_score': 40.0
        }

    def diagnose(self, query: str, answer: str, contexts: List[str],
                 relevance_scores: Optional[List[int]],
                 metrics: Dict[str, float],
                 business_metrics: Optional[Dict[str, float]] = None) -> FaultDiagnosis:

        # check business metrics first
        if business_metrics:
            if business_metrics.get("trust_score", 1.0) < self.thresholds.get('trust_score', 0.65):
                return FaultDiagnosis(
                    FaultType.GENERATION, FaultSubtype.LOW_TRUST, 0.85,
                    "Low trust score indicates potentially harmful or unfaithful content",
                    "Add citations, improve faithfulness, or add confidence scores"
                )
            if business_metrics.get("cost_score", 100) < self.thresholds.get('cost_score', 50):
                return FaultDiagnosis(
                    FaultType.BUSINESS, FaultSubtype.COST_INEFFICIENT, 0.75,
                    "RAG pipeline is too expensive for the business value delivered",
                    "Use smaller LLM, enable caching, or reduce retrieved chunks"
                )
            if business_metrics.get("latency_score", 100) < self.thresholds.get('latency_score', 40):
                return FaultDiagnosis(
                    FaultType.BUSINESS, FaultSubtype.LATENCY_CRITICAL, 0.8,
                    "Response latency exceeds business SLA requirements",
                    "Enable caching, use faster model, or optimize retrieval"
                )
            coverage_eff = business_metrics.get("coverage_efficiency", 1.0)
            waste_ratio = business_metrics.get("waste_ratio", 0.0)
            if coverage_eff < self.thresholds.get('coverage_efficiency', 0.15) and waste_ratio > 0.7:
                return FaultDiagnosis(
                    FaultType.RETRIEVAL, FaultSubtype.RANKER_QUALITY, 0.75,
                    "High waste ratio: too many chunks retrieved but few used",
                    "Improve retrieval precision or reduce number of retrieved chunks"
                )

        if not query or not query.strip():
            return FaultDiagnosis(FaultType.UNKNOWN, None, 1.0,
                                  "Empty query", "Check input pipeline")

        # no context AND no relevant documents — out_of_scope
        if not contexts or len(contexts) == 0:
            # if relevance_scores are empty or all zero — definitely out_of_scope
            if not relevance_scores or sum(relevance_scores) == 0:
                return FaultDiagnosis(FaultType.OUT_OF_SCOPE, FaultSubtype.SCOPE_MISMATCH, 0.9,
                                      "No relevant context retrieved and no documents found",
                                      "Expand knowledge base or improve retrieval")
            # if the context is empty but relevance_scores exist — this is also out_of_scope
            return FaultDiagnosis(FaultType.OUT_OF_SCOPE, FaultSubtype.SCOPE_MISMATCH, 0.85,
                                  "No context provided despite retrieval attempt",
                                  "Check retrieval pipeline or expand knowledge base")

        # if relevance_scores exist and are all zero — a retrieval problem
        if relevance_scores and sum(relevance_scores) == 0:
            return self._diagnose_retrieval(metrics)

        sfc = metrics.get("sfc", 0.0)
        ids_score = metrics.get("ids", 0.0)
        faith = metrics.get("faithfulness", 0.0)
        hit_rate = metrics.get("hit_rate@3", 1.0)
        ndcg = metrics.get("ndcg@5", 1.0)
        answer_tokens = len(answer.split()) if answer else 0
        context_tokens = len(" ".join(contexts).split()) if contexts else 0

        # determine whether the query is broad (requires a full answer)
        is_broad_query = self._is_broad_query(query)

        # for multilingual pairs — a softer faithfulness check
        # (faithfulness already accounts for multilinguality in NLIEntailment)
        if faith < self.thresholds['faithfulness']:
            return self._diagnose_generation(faith, sfc, ids_score, answer_tokens, context_tokens, is_broad_query)

        if hit_rate < self.thresholds['hit_rate'] or ndcg < self.thresholds['ndcg']:
            return self._diagnose_retrieval(metrics)

        if sfc < self.thresholds['sfc']:
            # more precise chunking vs generation diagnosis
            if ids_score > self.thresholds['ids'] or answer_tokens <= 10:
                # if the answer is short but faithful — it's chunking (context didn't fit in the answer)
                if faith >= self.thresholds['faithfulness'] and answer_tokens <= 15:
                    return FaultDiagnosis(FaultType.CHUNKING, FaultSubtype.CHUNKING_STRATEGY, 0.8,
                                          "Answer is faithful but too short for the query scope; likely chunking or token limit issue",
                                          "Try smaller chunks, increase max tokens, or improve prompt for completeness")
                return FaultDiagnosis(FaultType.CHUNKING, FaultSubtype.CHUNKING_STRATEGY, 0.75,
                                      "Answer is faithful but incomplete; high density or short answer suggests chunking issue",
                                      "Try smaller chunks or overlap")
            return FaultDiagnosis(FaultType.GENERATION, FaultSubtype.OMISSION, 0.7,
                                  "Answer misses key information from context",
                                  "Improve prompt engineering or increase max tokens")

        # for short answers to complex/broad queries — chunking, not generation
        if ids_score < self.thresholds['ids'] and answer_tokens > 15:
            return FaultDiagnosis(FaultType.CHUNKING, FaultSubtype.REDUNDANCY, 0.8,
                                  "Answer is verbose or redundant",
                                  "Add redundancy penalty in generation or improve prompt")

        # if the answer is too short relative to the context and the query is broad
        if answer_tokens <= 10 and context_tokens > 30 and is_broad_query:
            return FaultDiagnosis(FaultType.CHUNKING, FaultSubtype.OMISSION, 0.75,
                                  "Answer is too brief for a broad query given the available context",
                                  "Increase max tokens or improve prompt for completeness")

        # if the answer is too short relative to the context (any query)
        if answer_tokens <= 10 and context_tokens > 30 and len(contexts) > 1:
            return FaultDiagnosis(FaultType.CHUNKING, FaultSubtype.OMISSION, 0.7,
                                  "Answer is too brief given the available context from multiple sources",
                                  "Increase max tokens or improve prompt for completeness")

        return FaultDiagnosis(FaultType.HEALTHY, None, 1.0,
                              "All metrics within acceptable thresholds", "No action needed")

    def _is_broad_query(self, query: str) -> bool:
        """Determines whether the query is broad (requires a full answer)"""
        broad_keywords = [
            'всё о', 'все о', 'расскажи', 'tell me about', 'everything about',
            'почему', 'why', 'как', 'how to', 'сравни', 'compare',
            'опиши', 'describe', 'объясни', 'explain'
        ]
        query_lower = query.lower()
        return any(kw in query_lower for kw in broad_keywords)

    def _diagnose_retrieval(self, metrics: Dict[str, float]) -> FaultDiagnosis:
        hit_rate = metrics.get("hit_rate@3", 0.0)
        ndcg = metrics.get("ndcg@5", 0.0)
        precision = metrics.get("precision@2", 0.0)

        if hit_rate < 0.5:
            return FaultDiagnosis(FaultType.RETRIEVAL, FaultSubtype.EMBEDDING_QUALITY, 0.8,
                                  "Very low hit rate: retrieval fails to find any relevant documents",
                                  "Check embedding model quality or reindex with better chunking")

        if ndcg < self.thresholds.get('ndcg', 0.65) and precision < 0.5:
            return FaultDiagnosis(FaultType.RETRIEVAL, FaultSubtype.RANKER_QUALITY, 0.75,
                                  "Low NDCG and precision: relevant documents are poorly ranked",
                                  "Add reranking (cross-encoder) or fine-tune retriever")

        if ndcg < self.thresholds.get('ndcg', 0.65):
            return FaultDiagnosis(FaultType.RETRIEVAL, FaultSubtype.RANKER_QUALITY, 0.75,
                                  "Low NDCG: relevant documents are poorly ranked",
                                  "Add reranking (cross-encoder) or fine-tune retriever")

        return FaultDiagnosis(FaultType.RETRIEVAL, FaultSubtype.QUERY_UNDERSTANDING, 0.7,
                              "Retrieval partially works but could be improved",
                              "Analyze query expansion or hybrid search")

    def _diagnose_generation(self, faith: float, sfc: float, ids: float,
                             answer_tokens: int = 0, context_tokens: int = 0,
                             is_broad_query: bool = False) -> FaultDiagnosis:
        if faith < 0.3:
            return FaultDiagnosis(FaultType.GENERATION, FaultSubtype.HALLUCINATION, 0.9,
                                  "Severe hallucination detected",
                                  "Strengthen prompt grounding or add citation enforcement")

        if sfc < 0.3 and ids > 0.1:
            return FaultDiagnosis(FaultType.GENERATION, FaultSubtype.CONTRADICTION, 0.75,
                                  "Answer contradicts context or itself",
                                  "Add self-consistency check or contradiction detection")

        # for broad queries with low faithfulness — check whether this is actually omission
        if is_broad_query and faith < self.thresholds.get('faithfulness', 0.70) and faith >= 0.3:
            if answer_tokens <= 15 and context_tokens > 30:
                return FaultDiagnosis(FaultType.CHUNKING, FaultSubtype.OMISSION, 0.7,
                                      "Broad query but answer is too brief; likely generation missed key points",
                                      "Improve prompt for completeness or increase max tokens")

        if sfc > 0.5 and ids < 0.2 and answer_tokens > 15:
            return FaultDiagnosis(FaultType.GENERATION, FaultSubtype.REDUNDANCY, 0.7,
                                  "Answer is faithful but overly verbose with low information density",
                                  "Add conciseness constraints to prompt or post-process")

        return FaultDiagnosis(FaultType.GENERATION, FaultSubtype.OMISSION, 0.7,
                              "Low faithfulness: answer may contain unverified claims",
                              "Add retrieval augmentation or reduce temperature in generation")