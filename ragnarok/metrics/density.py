from .._libs import re, List, Dict, Tuple, np, sent_tokenize, SentenceTransformer, cosine_similarity


class InformationDensityMetric:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2",
                 dedup_threshold: float = 0.85, min_claim_length: int = 10):
        self.embedder = SentenceTransformer(embedding_model)
        self.dedup_threshold = dedup_threshold
        self.min_claim_length = min_claim_length
        self.cache = {}

    def _extract_claims(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        claims = []
        for sent in sent_tokenize(text):
            for part in re.split(r';\s*|\s+and\s+|\s+but\s+|\s+or\s+', sent):
                part = part.strip()
                if len(part) > self.min_claim_length:
                    claims.append(part)
        return claims if claims else [s for s in sent_tokenize(text) if len(s) > self.min_claim_length]

    def _embed(self, texts: List[str]) -> np.ndarray:
        valid = [t for t in texts if t and t.strip()]
        if not valid:
            return np.array([])
        missing = [t for t in valid if t not in self.cache]
        if missing:
            for t, emb in zip(missing, self.embedder.encode(missing, convert_to_numpy=True)):
                self.cache[t] = emb
        return np.array([self.cache[t] for t in valid])

    def _deduplicate_claims(self, claims: List[str]) -> Tuple[List[str], float]:
        if len(claims) <= 1:
            return claims, 0.0
        embs = self._embed(claims)
        sims = cosine_similarity(embs)
        unique, dups = [], [False] * len(claims)
        for i in range(len(claims)):
            if dups[i]:
                continue
            unique.append(claims[i])
            for j in range(i + 1, len(claims)):
                if sims[i][j] >= self.dedup_threshold:
                    dups[j] = True
        return unique, (len(claims) - len(unique)) / len(claims)

    def _novelty_scores(self, unique_claims: List[str], context_claims: List[str]) -> List[float]:
        if not unique_claims or not context_claims:
            return [1.0] * len(unique_claims)
        claim_embs = self._embed(unique_claims)
        context_embs = self._embed(context_claims)
        if len(claim_embs) == 0 or len(context_embs) == 0:
            return [1.0] * len(unique_claims)
        max_sims = cosine_similarity(claim_embs, context_embs).max(axis=1)
        return np.clip(1.0 - max_sims, 0.0, 1.0).tolist()

    def score(self, answer: str, contexts: List[str]) -> Dict:
        if not answer or not answer.strip():
            return {"ids_score": 0.0, "total_tokens": 0, "avg_novelty": 0.0, "redundancy_ratio": 0.0,
                    "context_answer_ratio": 0.0}

        raw_claims = self._extract_claims(answer)
        context_claims = self._extract_claims(" ".join(contexts)) if contexts else []
        unique, redundancy = self._deduplicate_claims(raw_claims)
        novelty = self._novelty_scores(unique, context_claims)
        tokens = len(answer.split())
        context_tokens = len(" ".join(contexts).split()) if contexts else 0

        # check the ratio of context to answer
        # if the answer is too short relative to the context — this is omission
        context_answer_ratio = context_tokens / max(tokens, 1)

        # for queries like "everything about", "tell me about" we expect a fuller answer
        # check whether the context contains more information than the answer
        info_ratio = len(context_claims) / max(len(raw_claims), 1)

        # determine whether the query is "broad" (requires a full answer)
        # if the context has many claims (>3) but the answer is very short
        is_broad_query = len(context_claims) > 3 and tokens <= 10

        if tokens <= 10 and len(raw_claims) <= 2 and redundancy < 0.3:
            base_score = 0.3
            novelty_bonus = float(np.mean(novelty)) * 0.3 if novelty else 0
            ids = min(base_score + novelty_bonus, 1.0)

            # if the context has a lot of information but the answer is short — penalize
            if info_ratio > 2.0 and context_tokens > 30:
                ids = min(ids, 0.15)  # low IDS for overly brief answers to broad queries
            elif is_broad_query:
                ids = min(ids, 0.10)  # even lower for clearly broad queries
        else:
            ids = (sum(novelty) / tokens) * (1.0 - redundancy) if tokens > 0 else 0.0

            # even for long answers, if they're too short relative to the context
            if context_answer_ratio > 5.0 and len(context_claims) > len(raw_claims) * 2:
                ids = min(ids, 0.25)

        return {"ids_score": min(ids, 1.0),
                "total_tokens": tokens,
                "avg_novelty": float(np.mean(novelty)) if novelty else 0.0,
                "redundancy_ratio": redundancy,
                "context_answer_ratio": context_answer_ratio}