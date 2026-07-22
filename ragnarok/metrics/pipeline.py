from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime

from .data_availability import DataAvailabilityMetrics
from .chunking import ChunkingQualityMetrics
from .embedding_quality import EmbeddingQualityMetrics
from .retrieval import RetrievalMetrics
from .reranking_quality import RerankingQualityMetrics


class PipelineStage(Enum):
    DATA_AVAILABILITY = "data_availability"    # Stage 0
    CHUNKING_QUALITY = "chunking_quality"      # Stage 1
    EMBEDDING_QUALITY = "embedding_quality"    # Stage 2
    RETRIEVAL_QUALITY = "retrieval_quality"    # Stage 3
    RERANKING_QUALITY = "reranking_quality"    # Stage 4


@dataclass
class StageResult:
    stage: PipelineStage
    stage_name: str
    passed: bool
    summary: str
    recommendation: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalPipelineReport:
    overall_passed: bool
    stages: List[StageResult]
    bottleneck_stage: Optional[PipelineStage]
    bottleneck_explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall_passed': self.overall_passed,
            'bottleneck_stage': self.bottleneck_stage.value if self.bottleneck_stage else None,
            'bottleneck_explanation': self.bottleneck_explanation,
            'stages': [
                {
                    'stage': s.stage.value,
                    'stage_name': s.stage_name,
                    'passed': s.passed,
                    'summary': s.summary,
                    'recommendation': s.recommendation,
                    'details': s.details,
                }
                for s in self.stages
            ]
        }


class RetrievalPipelineEvaluator:
    """
    Runs all 5 retrieval-pipeline stages together (data availability, chunking,
    embeddings, retrieval, reranking) and identifies the *first* failing stage as
    the bottleneck — so a diagnosis can say "the problem is at stage X" instead of
    a single opaque retrieval score.
    """
    def __init__(self,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 max_empty_result_rate: float = 0.05,
                 min_hit_rate: float = 0.70,
                 min_ndcg: float = 0.65):
        self.data_availability = DataAvailabilityMetrics()
        self.chunking = ChunkingQualityMetrics(embedding_model=embedding_model)
        self.embedding_quality = EmbeddingQualityMetrics(embedding_model=embedding_model)
        self.retrieval = RetrievalMetrics()
        self.reranking_quality = RerankingQualityMetrics()

        self.max_empty_result_rate = max_empty_result_rate
        self.min_hit_rate = min_hit_rate
        self.min_ndcg = min_ndcg

    def _stage_data_availability(self, query: str, contexts: List[str], **kwargs) -> StageResult:
        report = self.data_availability.evaluate(
            queries=[query], num_chunks_found=[len(contexts)], **kwargs
        )
        passed = report['empty_result_rate'] <= self.max_empty_result_rate and (
            report['coverage_gap'] is None or report['coverage_gap'] <= 0.1
        )
        return StageResult(
            stage=PipelineStage.DATA_AVAILABILITY,
            stage_name="Stage 0: Data availability",
            passed=passed,
            summary=report['diagnostic'][0] if report['diagnostic'] else "",
            recommendation=("No action needed" if passed else
                             "Populate the knowledge base with relevant documents and verify indexing."),
            details=report,
        )

    def _stage_chunking_quality(self, chunks: List[str]) -> StageResult:
        report = self.chunking.evaluate(chunks)
        passed = report['overall_grade'] in {"EXCELLENT", "GOOD"}
        return StageResult(
            stage=PipelineStage.CHUNKING_QUALITY,
            stage_name="Stage 1: Chunking quality",
            passed=passed,
            summary=f"Overall grade: {report['overall_grade']}",
            recommendation=(report['recommendations'][-1] if not passed and report.get('recommendations')
                             else "No action needed"),
            details=report,
        )

    def _stage_embedding_quality(self, query: Optional[str], **kwargs) -> StageResult:
        report = self.embedding_quality.evaluate(query=query, **kwargs)
        healthy_message = "Embedding quality appears healthy across the evaluated dimensions."
        passed = report['diagnostic'] == [healthy_message]
        return StageResult(
            stage=PipelineStage.EMBEDDING_QUALITY,
            stage_name="Stage 2: Embedding quality",
            passed=passed,
            summary=report['diagnostic'][0] if report['diagnostic'] else "",
            recommendation=("No action needed" if passed else
                             "Consider a different embedding model or fine-tuning on domain data."),
            details=report,
        )

    def _stage_retrieval_quality(self, relevance_scores: Optional[List[int]]) -> StageResult:
        if not relevance_scores:
            return StageResult(
                stage=PipelineStage.RETRIEVAL_QUALITY,
                stage_name="Stage 3: Retrieval quality",
                passed=True,
                summary="No relevance scores provided — retrieval ranking was not evaluated.",
                recommendation="Provide relevance_scores to diagnose ranking quality.",
                details={},
            )

        hit_rate = float(self.retrieval.hit_rate_at_k(relevance_scores, 3))
        ndcg = float(self.retrieval.ndcg_at_k(relevance_scores, 5))
        passed = hit_rate >= self.min_hit_rate and ndcg >= self.min_ndcg

        if hit_rate < self.min_hit_rate:
            summary = f"Low hit rate ({hit_rate:.2f}) — relevant chunks are often missing from the top results."
            recommendation = "Check embedding quality or expand the knowledge base."
        elif ndcg < self.min_ndcg:
            summary = f"Hit rate is fine but NDCG is low ({ndcg:.3f}) — relevant chunks are poorly ranked."
            recommendation = "Add a reranker or improve the retrieval/embedding model."
        else:
            summary = f"Retrieval finds and ranks relevant chunks well (hit_rate={hit_rate:.2f}, ndcg={ndcg:.3f})."
            recommendation = "No action needed"

        return StageResult(
            stage=PipelineStage.RETRIEVAL_QUALITY,
            stage_name="Stage 3: Retrieval quality",
            passed=passed,
            summary=summary,
            recommendation=recommendation,
            details={"hit_rate@3": hit_rate, "ndcg@5": ndcg},
        )

    def _stage_reranking_quality(self, relevance_before: Optional[List[int]],
                                  relevance_after: Optional[List[int]]) -> StageResult:
        if not relevance_before or not relevance_after:
            return StageResult(
                stage=PipelineStage.RERANKING_QUALITY,
                stage_name="Stage 4: Reranking quality",
                passed=True,
                summary="No reranker data provided — reranking was not evaluated.",
                recommendation="Provide before/after relevance scores to diagnose the reranker.",
                details={},
            )

        report = self.reranking_quality.evaluate(relevance_before, relevance_after)
        passed = report['overall_grade'] == "EFFECTIVE"
        return StageResult(
            stage=PipelineStage.RERANKING_QUALITY,
            stage_name="Stage 4: Reranking quality",
            passed=passed,
            summary=f"Overall grade: {report['overall_grade']}",
            recommendation=(report['recommendations'][-1] if not passed and report.get('recommendations')
                             else "No action needed"),
            details=report,
        )

    def evaluate_pipeline(self,
                           query: str,
                           contexts: List[str],
                           relevance_scores: Optional[List[int]] = None,
                           chunks_for_chunking_eval: Optional[List[str]] = None,
                           query_paraphrases: Optional[List[str]] = None,
                           clusters: Optional[List[List[str]]] = None,
                           embedding_scores: Optional[List[float]] = None,
                           cross_encoder_scores: Optional[List[float]] = None,
                           relevance_before_rerank: Optional[List[int]] = None,
                           corpus: Optional[List[str]] = None,
                           data_availability_kwargs: Optional[Dict[str, Any]] = None) -> RetrievalPipelineReport:
        """
        Runs Stage 0 through Stage 4 and returns a single report with a bottleneck diagnosis.
        Every stage beyond 0/1 is optional — pass only the data you have; stages without
        enough data are marked passed=True with a note that they weren't evaluated.
        """
        da_kwargs = dict(data_availability_kwargs or {})
        if corpus is not None:
            da_kwargs.setdefault('corpus', corpus)

        stages = [
            self._stage_data_availability(query, contexts, **da_kwargs),
            self._stage_chunking_quality(chunks_for_chunking_eval or contexts),
            self._stage_embedding_quality(
                query=query,
                doc_texts=contexts if contexts else None,
                clusters=clusters,
                query_paraphrases=query_paraphrases,
                embedding_scores=embedding_scores,
                cross_encoder_scores=cross_encoder_scores,
            ),
            self._stage_retrieval_quality(relevance_scores),
            self._stage_reranking_quality(relevance_before_rerank, relevance_scores),
        ]

        bottleneck = None
        bottleneck_explanation = "All stages passed."
        for stage in stages:
            if not stage.passed:
                bottleneck = stage.stage
                bottleneck_explanation = (
                    f"Bottleneck at '{stage.stage_name}': {stage.summary} Recommendation: {stage.recommendation}"
                )
                break

        return RetrievalPipelineReport(
            overall_passed=all(s.passed for s in stages),
            stages=stages,
            bottleneck_stage=bottleneck,
            bottleneck_explanation=bottleneck_explanation,
        )
