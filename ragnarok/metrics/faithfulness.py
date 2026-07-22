from .._libs import List, sent_tokenize, SentenceTransformer, cosine_similarity
import numpy as np


class NLIEntailment:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-large",
                 threshold: float = 0.7,
                 use_semantic_fallback: bool = True,
                 multilingual_threshold: float = 0.55):
        self.threshold = threshold
        self.use_semantic_fallback = use_semantic_fallback
        self.multilingual_threshold = multilingual_threshold
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

        # semantic similarity for cases where NLI is unsure
        if self.use_semantic_fallback:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def _extract_claims(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        return [s for s in sent_tokenize(text) if len(s) > 5]

    def _entailment_prob(self, premise: str, hypothesis: str) -> float:
        import torch
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt",
                                truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        # probs: [0]=contradiction, [1]=neutral, [2]=entailment
        return float(probs[0]), float(probs[1]), float(probs[2])

    def _semantic_similarity(self, claim: str, context: str) -> float:
        """ semantic similarity for when NLI is unsure"""
        if not self.use_semantic_fallback:
            return 0.0
        claim_emb = self.embedder.encode([claim])
        context_emb = self.embedder.encode([context])
        sim = cosine_similarity(claim_emb, context_emb)[0][0]
        return float(sim)

    def _detect_language(self, text: str) -> str:
        """Simple heuristic for language detection (latin vs cyrillic vs other)"""
        cyrillic = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
        latin = sum(1 for c in text if c.isascii() and c.isalpha())
        if cyrillic > latin * 0.3:
            return "cyrillic"
        elif latin > 0:
            return "latin"
        return "other"

    def _is_multilingual_pair(self, claim: str, context: str) -> bool:
        """Check whether the claim and context are in different languages"""
        claim_lang = self._detect_language(claim)
        context_lang = self._detect_language(context)
        return claim_lang != context_lang and claim_lang != "other" and context_lang != "other"

    def score(self, answer: str, contexts: List[str]) -> float:
        if not answer or not contexts:
            return 0.0
        answer_claims = self._extract_claims(answer)
        if not answer_claims:
            return 1.0

        context = " ".join(contexts)

        # truncate the context to 400 words
        context_words = context.split()
        if len(context_words) > 400:
            context = " ".join(context_words[:400])

        entailed_count = 0
        for claim in answer_claims:
            contra, neutral, entail = self._entailment_prob(context, claim)

            is_multilingual = self._is_multilingual_pair(claim, context)

            # SCORING LOGIC:
            # 1. If entailment >= threshold — definitely faithful
            # 2. If contradiction >= 0.5 — definitely unfaithful
            # 3. If neutral is high but semantic similarity is high — treat as faithful (soft approach)
            # 4. For multilingual pairs — use softer semantic similarity thresholds
            # 5. If contradiction is low and entailment is moderate — treat as faithful

            if entail >= self.threshold:
                entailed_count += 1
            elif contra >= 0.5:
                # explicit contradiction — not faithful
                pass
            elif self.use_semantic_fallback:
                # check semantic closeness
                sim = self._semantic_similarity(claim, context)

                if is_multilingual:
                    # for multilingual pairs — a softer threshold
                    # if similarity > 0.55 and contradiction isn't too high — treat as faithful
                    if sim > self.multilingual_threshold and contra < 0.4:
                        entailed_count += 1
                    # extra check: if entailment > 0.15 and contradiction is low
                    elif entail >= 0.15 and contra < 0.35 and sim > 0.45:
                        entailed_count += 1
                else:
                    # regular (monolingual) pairs — standard threshold
                    if sim > 0.7 and contra < 0.3:
                        entailed_count += 1
                    # soft threshold: if entailment > 0.25 and contradiction is low
                    elif entail >= 0.25 and contra < 0.3 and sim > 0.55:
                        entailed_count += 1
            elif entail >= 0.3 and contra < 0.3:
                # soft threshold: if entailment > 0.3 and contradiction is low
                entailed_count += 1

        return entailed_count / len(answer_claims)