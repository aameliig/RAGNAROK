from ._libs import List, Dict, Any, Optional, Path
from .config import EvaluationConfig
from .metrics.retrieval import RetrievalMetrics
from .metrics.coverage import SemanticFootprintCoverage
from .metrics.density import InformationDensityMetric
from .metrics.faithfulness import NLIEntailment
from .metrics.chunking import ChunkingQualityMetrics
from .metrics.data_availability import DataAvailabilityMetrics
from .metrics.embedding_quality import EmbeddingQualityMetrics
from .metrics.reranking_quality import RerankingQualityMetrics
from .metrics.pipeline import RetrievalPipelineEvaluator
from .business.fault_classifier import FaultClassifier, FaultDiagnosis
from .business.business_metrics import BusinessMetrics, BusinessDimensionWeights
from .presets import load_preset


class RAGEvaluator:
    """Orchestrates retrieval, generation, and business metrics into a single Business Score."""
    def __init__(self, config: Optional[EvaluationConfig] = None,
                 config_path: Optional[str] = None,
                 preset: Optional[str] = None,
                 business_dimension_weights: Optional[BusinessDimensionWeights] = None):
        """
        preset: name of a built-in config preset (see ragnarok.presets.list_presets()) —
        takes priority over `config`/`config_path` if given.
        business_dimension_weights: weights for the composite business score
        (speed/cost/accuracy/coverage) — see BusinessMetrics.compute_composite_business_score.
        """
        if preset:
            self.config = load_preset(preset)
        elif config:
            self.config = config
        elif config_path:
            path = Path(config_path)
            if path.suffix in ('.yaml', '.yml'):
                self.config = EvaluationConfig.from_yaml(path)
            else:
                self.config = EvaluationConfig.from_json(path)
        else:
            self.config = EvaluationConfig(metrics={})

        self._business_dimension_weights = business_dimension_weights

        self._sfc = None
        self._ids = None
        self._nli = None
        self._retrieval = None
        self._chunking = None
        self._data_availability = DataAvailabilityMetrics()
        self._embedding_quality = None
        self._reranking_quality = None
        self._pipeline = None
        self._ragas = None
        self._llm_judge = None
        self._fault_classifier = None
        self._business_metrics = None

    @property
    def sfc(self):
        if self._sfc is None:
            self._sfc = SemanticFootprintCoverage()
        return self._sfc

    @property
    def ids_metric(self):
        if self._ids is None:
            self._ids = InformationDensityMetric()
        return self._ids

    @property
    def nli(self):
        if self._nli is None:
            self._nli = NLIEntailment()
        return self._nli

    @property
    def data_availability(self):
        if self._data_availability is None:
            self._data_availability = DataAvailabilityMetrics()
        return self._data_availability

    @property
    def retrieval(self):
        if self._retrieval is None:
            self._retrieval = RetrievalMetrics()
        return self._retrieval

    @property
    def chunking(self):
        if self._chunking is None:
            self._chunking = ChunkingQualityMetrics()
        return self._chunking

    @property
    def embedding_quality(self):
        """Query-doc similarity, cluster compactness, intra-query stability, cross-encoder calibration.

        Not part of `evaluate()` — these need richer optional inputs (query paraphrases,
        semantic clusters, cross-encoder scores) that a single query/answer/contexts call
        doesn't carry. Call `evaluator.embedding_quality.evaluate(...)` directly instead.
        """
        if self._embedding_quality is None:
            self._embedding_quality = EmbeddingQualityMetrics()
        return self._embedding_quality

    @property
    def reranking_quality(self):
        """Reranking lift, relevance order accuracy, first-relevant-position shift.

        Not part of `evaluate()` — needs relevance grades from before and after reranking
        for the same query. Call `evaluator.reranking_quality.evaluate(...)` directly instead.
        """
        if self._reranking_quality is None:
            self._reranking_quality = RerankingQualityMetrics()
        return self._reranking_quality

    @property
    def pipeline(self):
        """Runs all 5 retrieval-pipeline stages together with bottleneck detection.

        Also computed automatically inside `evaluate()` (as `retrieval_pipeline_report`
        in the result) whenever contexts are given — call this property directly only if
        you want to run the pipeline diagnostic standalone, outside of a full evaluate().
        """
        if self._pipeline is None:
            self._pipeline = RetrievalPipelineEvaluator()
        return self._pipeline

    @property
    def fault_classifier(self):
        if self._fault_classifier is None:
            self._fault_classifier = FaultClassifier(self.config.fault_thresholds)
        return self._fault_classifier

    @property
    def business_metrics(self):
        if self._business_metrics is None:
            cost_cfg = self.config.cost_config or {}
            model = cost_cfg.get("model") or (
                self.config.llm_judge_model if self.config.llm_judge_enabled else "local"
            )
            custom_pricing = None
            if "input_token_price" in cost_cfg or "output_token_price" in cost_cfg:
                # cost_config prices are $ per 1K tokens (OpenAI convention);
                # BusinessMetrics.PRICING is $ per 1M tokens.
                custom_pricing = {
                    "input": cost_cfg.get("input_token_price", 0.0) * 1000,
                    "output": cost_cfg.get("output_token_price", 0.0) * 1000,
                }
            self._business_metrics = BusinessMetrics(
                model, custom_pricing=custom_pricing, dimension_weights=self._business_dimension_weights
            )
        return self._business_metrics

    def _init_ragas(self):
        if self._ragas is None and self.config.ragas_enabled:
            from .integrations.ragas_bridge import RAGASBridge
            self._ragas = RAGASBridge()
        return self._ragas

    def _init_llm_judge(self):
        if self._llm_judge is None and self.config.llm_judge_enabled:
            from .integrations.llm_judge import LLMJudge
            self._llm_judge = LLMJudge.create(
                self.config.llm_judge_provider,
                self.config.llm_judge_model
            )
        return self._llm_judge

    # Metric groups — each returns the pieces `evaluate()` assembles

    def _score_retrieval(self, query: str, contexts: List[str],
                          relevance_scores: Optional[List[int]]) -> Dict[str, Any]:
        """Data availability, ranking metrics (if ground truth given), and chunking quality."""
        scores: Dict[str, Any] = {
            "data_availability": self.data_availability.evaluate(
                queries=[query],
                num_chunks_found=[len(contexts)],
            )
        }

        if relevance_scores:
            scores.update({
                "hit_rate@3": self.retrieval.hit_rate_at_k(relevance_scores, 3),
                "precision@2": self.retrieval.precision_at_k(relevance_scores, 2),
                "ndcg@5": self.retrieval.ndcg_at_k(relevance_scores, 5),
                "recall@5": self.retrieval.recall_at_k(relevance_scores, sum(relevance_scores), 5),
            })

        if contexts:
            scores["chunking_quality"] = self.chunking.evaluate(contexts)

        return scores

    def _score_generation(self, answer: str, contexts: List[str]) -> Dict[str, Any]:
        """SFC, IDS, and NLI faithfulness, plus the raw sub-results later stages need."""
        sfc_result = self.sfc.score(answer, contexts)
        ids_result = self.ids_metric.score(answer, contexts)
        faith = self.nli.score(answer, contexts)

        return {
            "scores": {
                "sfc": sfc_result["sfc_score"],
                "ids": ids_result["ids_score"],
                "faithfulness": faith,
            },
            "ids_result": ids_result,
            "faithfulness": faith,
        }

    def _score_optional_modules(self, query: str, answer: str,
                                 contexts: List[str]) -> Dict[str, Dict[str, Any]]:
        """RAGAS and LLM-judge scores, only computed if enabled in config."""
        ragas_scores: Dict[str, Any] = {}
        if self.config.ragas_enabled:
            ragas = self._init_ragas()
            if ragas:
                ragas_scores = ragas.score(query, answer, contexts)

        llm_scores: Dict[str, Any] = {}
        if self.config.llm_judge_enabled:
            judge = self._init_llm_judge()
            if judge:
                llm_results = judge.evaluate(query, answer, contexts)
                llm_scores = {f"llm_{k}": v.score for k, v in llm_results.items()}

        return {"ragas": ragas_scores, "llm_judge": llm_scores}

    def _score_business(self, generation: Dict[str, Any], llm_scores: Dict[str, float],
                         latency_ms: float, input_tokens: int, output_tokens: int,
                         num_chunks_retrieved: int, has_citations: bool,
                         sfc_score: float, gpu_hours: float = 0.0,
                         price_per_gpu_hour: Optional[float] = None,
                         gpu_type: Optional[str] = None, num_queries: int = 1) -> Dict[str, Any]:
        """Cost, latency, trust, coverage efficiency, satisfaction-proxy, and composite scores."""
        business_scores: Dict[str, Any] = {}

        if price_per_gpu_hour is None:
            price_per_gpu_hour = (self.config.cost_config or {}).get("gpu_price_per_hour", 0.0)

        # If a GPU type is given but no explicit gpu_hours/price, estimate both from token counts.
        if gpu_type and gpu_hours <= 0:
            gpu_estimate = self.business_metrics.compute_gpu_metrics(
                input_tokens, output_tokens, num_queries, gpu_type
            )
            gpu_hours = gpu_estimate.gpu_hours
            if not price_per_gpu_hour:
                price_per_gpu_hour = gpu_estimate.cost_per_gpu_hour
            business_scores["gpu_efficiency_score"] = gpu_estimate.efficiency_score
            business_scores["tokens_per_gpu_hour"] = gpu_estimate.tokens_per_gpu_hour

        if input_tokens > 0 or output_tokens > 0 or gpu_hours > 0:
            business_scores.update(self.business_metrics.compute_cost_score(
                input_tokens, output_tokens, num_chunks_retrieved,
                gpu_hours=gpu_hours, price_per_gpu_hour=price_per_gpu_hour
            ))

        business_scores.update(self.business_metrics.compute_latency_grade(latency_ms))

        business_scores.update(self.business_metrics.compute_trust_score(
            generation["faithfulness"], has_citations=has_citations
        ))

        if num_chunks_retrieved > 0:
            business_scores.update(self.business_metrics.compute_coverage_efficiency(
                sfc_score, num_chunks_retrieved
            ))

        business_scores.update(self.business_metrics.compute_user_satisfaction_proxy(
            faithfulness=generation["faithfulness"],
            relevance=llm_scores.get("llm_relevance", 0.0),
            completeness=sfc_score,
            conciseness=generation["ids_result"].get("avg_novelty", 0.0),
        ))

        composite = self.business_metrics.compute_composite_business_score(
            latency_score=business_scores.get("latency_score", 0.0),
            cost_score=business_scores.get("cost_score", 100.0),
            trust_score=business_scores.get("trust_score", 0.0),
            coverage_score=sfc_score,
        )
        business_scores["composite_business_score"] = composite["composite_business_score"]
        business_scores["dimension_scores"] = composite["dimension_scores"]
        business_scores["weakest_dimension"] = composite["weakest_dimension"]

        return business_scores

    # Public API

    def evaluate(self, query: str, answer: str, contexts: List[str],
                 relevance_scores: Optional[List[int]] = None,
                 latency_ms: float = 0.0,
                 input_tokens: int = 0,
                 output_tokens: int = 0,
                 num_chunks_retrieved: int = 0,
                 has_citations: bool = False,
                 gpu_hours: float = 0.0,
                 price_per_gpu_hour: Optional[float] = None,
                 gpu_type: Optional[str] = None,
                 num_queries: int = 1,
                 # Extra optional inputs for the retrieval-pipeline bottleneck diagnosis —
                 # only the stages you provide data for get evaluated with real signal.
                 chunks_for_chunking_eval: Optional[List[str]] = None,
                 query_paraphrases: Optional[List[str]] = None,
                 clusters: Optional[List[List[str]]] = None,
                 embedding_scores: Optional[List[float]] = None,
                 cross_encoder_scores: Optional[List[float]] = None,
                 relevance_before_rerank: Optional[List[int]] = None,
                 corpus: Optional[List[str]] = None) -> Dict[str, Any]:
        retrieval_scores = self._score_retrieval(query, contexts, relevance_scores)
        generation = self._score_generation(answer, contexts)
        optional = self._score_optional_modules(query, answer, contexts)

        scores: Dict[str, Any] = {}
        scores.update(retrieval_scores)
        scores.update(generation["scores"])
        scores.update(optional["ragas"])
        scores.update(optional["llm_judge"])

        business_scores = self._score_business(
            generation, optional["llm_judge"], latency_ms, input_tokens, output_tokens,
            num_chunks_retrieved, has_citations, scores.get("sfc", 0.0),
            gpu_hours, price_per_gpu_hour, gpu_type, num_queries
        )
        scores.update(business_scores)

        final_score, penalty, status = self.config.aggregate(scores, latency_ms)

        diagnosis = self.fault_classifier.diagnose(
            query, answer, contexts, relevance_scores, scores, business_scores
        )

        pipeline_report = None
        if contexts:
            pipeline_report = self.pipeline.evaluate_pipeline(
                query=query, contexts=contexts, relevance_scores=relevance_scores,
                chunks_for_chunking_eval=chunks_for_chunking_eval,
                query_paraphrases=query_paraphrases, clusters=clusters,
                embedding_scores=embedding_scores, cross_encoder_scores=cross_encoder_scores,
                relevance_before_rerank=relevance_before_rerank, corpus=corpus,
            ).to_dict()

        return {
            "final_score": final_score,
            "status": status,
            "fault_type": diagnosis.fault_type.value,
            "fault_subtype": diagnosis.fault_subtype.value if diagnosis.fault_subtype else None,
            "fault_confidence": diagnosis.confidence,
            "fault_explanation": diagnosis.explanation,
            "recommended_action": diagnosis.recommended_action,
            "penalty": penalty,
            "latency_ms": latency_ms,
            "metrics": scores,
            "business_metrics": business_scores,
            "ragas_metrics": optional["ragas"],
            "llm_judge_metrics": optional["llm_judge"],
            "data_availability": retrieval_scores["data_availability"],
            "chunking_quality": retrieval_scores.get("chunking_quality"),
            "retrieval_pipeline_report": pipeline_report,
            "config_used": {
                "ragas_enabled": self.config.ragas_enabled,
                "llm_judge_enabled": self.config.llm_judge_enabled,
            }
        }

    def evaluate_batch(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.evaluate(**ex) for ex in examples]

    def generate_report(self, result: Dict[str, Any], use_llm: bool = False,
                         llm_provider: Optional[str] = None, llm_model: Optional[str] = None) -> Any:
        """
        Builds a report from an evaluate() result.
        use_llm=False (default): deterministic StructuredReport, no API key needed —
            call .to_markdown() / .to_html() / .to_dict() on the result.
        use_llm=True: narrative text from an LLM (defaults to the configured llm_judge
            provider/model, or pass llm_provider/llm_model explicitly).
        """
        if use_llm:
            from .reporting import LLMReportGenerator
            provider = llm_provider or self.config.llm_judge_provider
            model = llm_model or self.config.llm_judge_model
            return LLMReportGenerator.create(provider, model).generate(result)

        from .reporting import StructuredReportGenerator
        return StructuredReportGenerator().generate(
            result, retrieval_pipeline_report=result.get("retrieval_pipeline_report")
        )

    @staticmethod
    def print_results(result: Dict[str, Any], title: str = "RAG Evaluation Results") -> None:
        from textwrap import fill

        width = 60
        print("\n" + "=" * width)
        print(f"  {title}")
        print("=" * width)

        score = result.get("final_score", 0.0)
        status = result.get("status", "UNKNOWN")
        status_icon = "✅" if status == "PASSED" else "❌" if status == "FAILED" else "⚠️"
        print(f"\n  📊 Business Score: {score:.1f}/100  {status_icon} {status}")

        fault = result.get("fault_type", "unknown")
        subtype = result.get("fault_subtype")
        confidence = result.get("fault_confidence", 0.0)
        explanation = result.get("fault_explanation", "")
        action = result.get("recommended_action", "")

        print(f"\n  🔍 Fault: {fault.upper()}")
        if subtype:
            print(f"     Subtype: {subtype}")
        print(f"     Confidence: {confidence:.0%}")
        if explanation:
            print(f"     Explanation: {fill(explanation, width=width-8, initial_indent=' ' * 8, subsequent_indent=' ' * 8).lstrip()}")
        if action:
            print(f"     Action: {fill(action, width=width-8, initial_indent=' ' * 8, subsequent_indent=' ' * 8).lstrip()}")

        metrics = result.get("metrics", {})
        print(f"\n  📈 Core Metrics")
        print(f"     Faithfulness:   {metrics.get('faithfulness', 0):.2f}")
        print(f"     SFC:            {metrics.get('sfc', 0):.2f}")
        print(f"     IDS:            {metrics.get('ids', 0):.4f}")
        if "hit_rate@3" in metrics:
            print(f"     HitRate@3:      {metrics.get('hit_rate@3', 0):.2f}")
        if "ndcg@5" in metrics:
            print(f"     NDCG@5:         {metrics.get('ndcg@5', 0):.4f}")

        biz = result.get("business_metrics", {})
        if biz:
            print(f"\n  💼 Business Metrics")
            if "cost_score" in biz:
                print(f"     Cost Score:     {biz.get('cost_score', 0):.1f}")
            if "latency_grade" in biz:
                print(f"     Latency Grade:  {biz.get('latency_grade', 'N/A')}")
            if "trust_score" in biz:
                print(f"     Trust Score:    {biz.get('trust_score', 0):.2f}")
            if "trust_level" in biz:
                print(f"     Trust Level:    {biz.get('trust_level', 'N/A')}")
            if "coverage_efficiency" in biz:
                print(f"     Coverage Eff.:  {biz.get('coverage_efficiency', 0):.3f}")
            if "waste_ratio" in biz:
                print(f"     Waste Ratio:    {biz.get('waste_ratio', 0):.2f}")
            if "user_satisfaction_proxy" in biz:
                print(f"     User Sat. Proxy:{biz.get('user_satisfaction_proxy', 0):.2f}")
            if "composite_business_score" in biz:
                print(f"     Composite Biz:  {biz.get('composite_business_score', 0):.1f} "
                      f"(weakest: {biz.get('weakest_dimension', 'N/A')})")

        chunking = result.get("chunking_quality")
        if chunking and "overall_grade" in chunking:
            print(f"\n  🧩 Chunking Quality: {chunking['overall_grade']} (score: {chunking.get('overall_score', 0):.2f})")

        pipeline_report = result.get("retrieval_pipeline_report")
        if pipeline_report:
            print(f"\n  🔎 Retrieval Pipeline (5 Stages)")
            print(f"     Overall: {'✅ PASSED' if pipeline_report.get('overall_passed') else '❌ FAILED'}")
            if pipeline_report.get("bottleneck_stage"):
                print(f"     🚨 Bottleneck: {pipeline_report['bottleneck_stage']}")
            for stage in pipeline_report.get("stages", []):
                icon = "✅" if stage.get("passed") else "❌"
                print(f"     {icon} {stage.get('stage_name', 'Unknown')}")

        config_used = result.get("config_used", {})
        ragas = result.get("ragas_metrics", {})
        llm = result.get("llm_judge_metrics", {})
        if ragas or llm or config_used:
            print(f"\n  🔌 Optional Modules")
            print(f"     RAGAS:          {'enabled' if config_used.get('ragas_enabled') else 'disabled'}")
            print(f"     LLM Judge:      {'enabled' if config_used.get('llm_judge_enabled') else 'disabled'}")
            if ragas:
                for k, v in ragas.items():
                    print(f"     {k}: {v:.3f}")
            if llm:
                for k, v in llm.items():
                    print(f"     {k}: {v:.2f}")

        print("\n" + "=" * width + "\n")
