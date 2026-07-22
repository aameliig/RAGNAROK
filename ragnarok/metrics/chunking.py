import re
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
import nltk

# Load NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')


class ChunkingQualityMetrics:
    """
    Metrics for evaluating chunking quality
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedding_model) if embedding_model else None

    @staticmethod
    def boundary_coherence(chunks: List[str]) -> Dict[str, Any]:
        """
        Fraction of chunks that start with a capital letter and end with
        proper punctuation (., !, ?, ", ), ] or }).
        """
        if not chunks:
            return {
                'coherent_fraction': 1.0,
                'proper_start_fraction': 1.0,
                'proper_end_fraction': 1.0,
                'coherent_chunks': [],
                'incoherent_chunks': []
            }

        proper_start = 0
        proper_end = 0
        coherent = 0
        coherent_examples = []
        incoherent_examples = []

        for chunk in chunks:
            chunk_stripped = chunk.strip()
            if not chunk_stripped:
                continue
            start_ok = False
            for char in chunk_stripped:
                if char.isalpha():
                    start_ok = char.isupper()
                    break
                elif char.isdigit():
                    # if it starts with a digit, consider it acceptable (e.g. "1. Introduction")
                    start_ok = True
                    break
            if start_ok:
                proper_start += 1

            # check the end: the last character is a proper delimiter
            end_chars = chunk_stripped[-1]
            end_ok = end_chars in {'.', '!', '?', '"', ')', ']', '}'}
            if end_ok:
                proper_end += 1

            if start_ok and end_ok:
                coherent += 1
                if len(coherent_examples) < 3:
                    coherent_examples.append(chunk[:80] + "..." if len(chunk) > 80 else chunk)
            else:
                if len(incoherent_examples) < 3:
                    incoherent_examples.append(chunk[:80] + "..." if len(chunk) > 80 else chunk)

        total = len(chunks)
        return {
            'coherent_fraction': round(coherent / total, 4) if total > 0 else 1.0,
            'proper_start_fraction': round(proper_start / total, 4) if total > 0 else 1.0,
            'proper_end_fraction': round(proper_end / total, 4) if total > 0 else 1.0,
            'coherent_chunks': coherent_examples,
            'incoherent_chunks': incoherent_examples
        }

    @staticmethod
    def chunk_size_variance(chunks: List[str], tokenize: bool = True) -> Dict[str, Any]:
        """
        Variability of chunk sizes in tokens (or characters)
        """
        if not chunks:
            return {
                'mean_size': 0.0,
                'std_size': 0.0,
                'cv': 0.0,
                'min_size': 0,
                'max_size': 0,
                'sizes': [],
                'diagnostic': "No chunks to analyze"
            }

        sizes = []
        for chunk in chunks:
            if tokenize:
                tokens = word_tokenize(chunk)
                sizes.append(len(tokens))
            else:
                sizes.append(len(chunk))

        mean = np.mean(sizes)
        std = np.std(sizes)
        cv = std / mean if mean > 0 else 0.0

        if cv < 0.3:
            diagnostic = "Good chunk size stability (low variability)"
        elif cv < 0.6:
            diagnostic = "Moderate chunk size variability"
        else:
            diagnostic = "High chunk size variability — possible issues with chunking"

        return {
            'mean_size': round(mean, 2),
            'std_size': round(std, 2),
            'cv': round(cv, 4),
            'min_size': int(np.min(sizes)),
            'max_size': int(np.max(sizes)),
            'sizes': sizes,
            'diagnostic': diagnostic
        }

    @staticmethod
    def informative_token_ratio(chunks: List[str]) -> Dict[str, Any]:
        """
        Fraction of informative tokens (nouns, verbs, adjectives,
        adverbs) in the chunks. High ITR = little fluff, low ITR = many stopwords.
        """
        if not chunks:
            return {'mean_itr': 0.0, 'per_chunk_itr': [], 'diagnostic': "No chunks to analyze"}

        # POS tags for informative parts of speech (Penn Treebank tagset)
        informative_tags = {'NN', 'NNS', 'NNP', 'NNPS',  # nouns
                            'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # verbs
                            'JJ', 'JJR', 'JJS',  # adjectives
                            'RB', 'RBR', 'RBS'}  # adverbs
        itrs = []
        for chunk in chunks:
            words = word_tokenize(chunk)
            if not words:
                itrs.append(0.0)
                continue
            tagged = pos_tag(words)
            informative_count = sum(1 for _, tag in tagged if tag in informative_tags)
            itr = informative_count / len(words)
            itrs.append(itr)

        mean_itr = np.mean(itrs) if itrs else 0.0

        if mean_itr > 0.45:
            diagnostic = "High informativeness (few stopwords / fluff)"
        elif mean_itr > 0.30:
            diagnostic = "Moderate informativeness, possible redundancy"
        else:
            diagnostic = "Low informativeness — excessive fluff, chunks may be diluted"
        
        return {
            'mean_itr': round(mean_itr, 4),
            'per_chunk_itr': [round(x, 4) for x in itrs],
            'diagnostic': diagnostic
        }

    def semantic_cohesion(self, chunks: List[str]) -> Dict[str, Any]:
        """
        Average cosine similarity between neighboring sentences within each chunk.
        High cohesion = chunks are logically unified.
        Low cohesion = the chunk mixes unrelated topics (information fragmentation).
        """
        if self.embedder is None:
            return {
                'mean_cohesion': 0.0,
                'per_chunk_cohesion': [],
                'diagnostic': "Embedding model not loaded",
                'requires_embedding': True
            }

        if not chunks:
            return {'mean_cohesion': 0.0, 'per_chunk_cohesion': [], 'diagnostic': "No chunks to analyze"}

        chunk_cohesions = []

        for chunk in chunks:
            sentences = sent_tokenize(chunk)
            if len(sentences) < 2:
                chunk_cohesions.append(1.0)  # a single sentence is maximally cohesive
                continue

            embeddings = self.embedder.encode(sentences, convert_to_numpy=True)
            if embeddings.shape[0] < 2:
                chunk_cohesions.append(1.0)
                continue

            similarities = []
            for i in range(len(embeddings) - 1):
                sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]) + 1e-8
                )
                similarities.append(sim)

            chunk_cohesions.append(np.mean(similarities))

        mean_cohesion = np.mean(chunk_cohesions) if chunk_cohesions else 0.0

        if mean_cohesion > 0.75:
            diagnostic = "High semantic cohesion within chunks (logically coherent)"
        elif mean_cohesion > 0.55:
            diagnostic = "Moderate cohesion — possible minor semantic shifts"
        else:
            diagnostic = "Low cohesion — chunks may contain heterogeneous information (fragmentation)"

        return {
            'mean_cohesion': round(mean_cohesion, 4),
            'per_chunk_cohesion': [round(x, 4) for x in chunk_cohesions],
            'diagnostic': diagnostic,
            'requires_embedding': False
        }

    def evaluate(self, chunks: List[str]) -> Dict[str, Any]:
        """
        Summary of all metrics
        """
        if not chunks:
            return {
                'overall_grade': "NO_DATA",
                'diagnostics': ["No chunks for evaluation"],
                'recommendations': ["Check chunking pipeline"]
            }

        boundary_res = self.boundary_coherence(chunks)
        variance_res = self.chunk_size_variance(chunks)
        itr_res = self.informative_token_ratio(chunks)
        cohesion_res = self.semantic_cohesion(chunks)

        diagnostics = []
        recommendations = []

        # Boundary coherence
        if boundary_res['coherent_fraction'] < 0.5:
            diagnostics.append(
                f"Low boundary coherence: only {boundary_res['coherent_fraction']*100:.1f}% "
                "of chunks start and end correctly."
            )
            recommendations.append(
                "Review the chunking strategy: avoid cutting sentences mid-way "
                "(use semantic chunking or overlap)."
            )
        elif boundary_res['coherent_fraction'] < 0.8:
            diagnostics.append(
                f"Moderate boundary coherence: {boundary_res['coherent_fraction']*100:.1f}% of chunks are correct."
            )
        else:
            diagnostics.append("Chunk boundaries look good (proper start and end).")

        # Size variance
        if variance_res['cv'] > 0.6:
            diagnostics.append(
                f"High chunk size variability (CV={variance_res['cv']:.2f}). "
                f"Sizes range from {variance_res['min_size']} to {variance_res['max_size']} tokens."
            )
            recommendations.append(
                "Use a fixed chunk size or an adaptive chunking strategy "
                "based on semantic boundaries."
            )
        elif variance_res['cv'] > 0.35:
            diagnostics.append(
                f"Moderate chunk size variability (CV={variance_res['cv']:.2f})."
            )
        else:
            diagnostics.append(f"Stable chunk sizes (CV={variance_res['cv']:.2f}).")

        # Informativeness
        if itr_res['mean_itr'] < 0.30:
            diagnostics.append(
                f"Low informative density (ITR={itr_res['mean_itr']:.2f}) — "
                "chunks contain a lot of stopwords and noise."
            )
            recommendations.append(
                "Clean up chunks by removing stopwords or applying more aggressive "
                "filtering (e.g. stopword removal or NER-based filtering)."
            )
        elif itr_res['mean_itr'] < 0.40:
            diagnostics.append(
                f"Moderate informative density (ITR={itr_res['mean_itr']:.2f})."
            )
        else:
            diagnostics.append(
                f"Good informative density (ITR={itr_res['mean_itr']:.2f})."
            )

        # Cohesion
        if cohesion_res.get('mean_cohesion', 1.0) < 0.50:
            diagnostics.append(
                f"Low semantic cohesion within chunks (cohesion={cohesion_res['mean_cohesion']:.2f}) "
                "— chunks mix unrelated topics."
            )
            recommendations.append(
                "Split chunks along semantic boundaries (paragraphs or sections) "
                "instead of by length alone."
            )
        elif cohesion_res.get('mean_cohesion', 1.0) < 0.70:
            diagnostics.append(
                f"Moderate semantic cohesion (cohesion={cohesion_res['mean_cohesion']:.2f})."
            )
        else:
            if not cohesion_res.get('requires_embedding', False):
                diagnostics.append(
                    f"High semantic cohesion (cohesion={cohesion_res['mean_cohesion']:.2f}) "
                    "— chunks are logically coherent."
                )
            else:
                diagnostics.append("Semantic cohesion not evaluated (no embedding model loaded).")

        # Overall score
        score = (
            boundary_res['coherent_fraction'] * 0.30
            + max(0, 1 - min(variance_res['cv'], 1.0)) * 0.20
            + min(itr_res['mean_itr'] * 2, 1.0) * 0.25
            + (cohesion_res.get('mean_cohesion', 1.0) if not cohesion_res.get('requires_embedding', False) else 0.6) * 0.25
        )

        if score >= 0.85:
            overall = "EXCELLENT"
        elif score >= 0.65:
            overall = "GOOD"
        elif score >= 0.45:
            overall = "FAIR"
        else:
            overall = "POOR"

        if overall in {"EXCELLENT", "GOOD"}:
            diagnostics.append("Chunking is performing well.")
            recommendations.append("The current strategy can be kept as is.")
        elif overall == "FAIR":
            recommendations.append("Consider revisiting the chunking strategy (overlap, semantic splitting).")
        else:
            recommendations.append("Urgently review the chunking pipeline — it is degrading overall RAG quality.")

        return {
            'boundary': boundary_res,
            'variance': variance_res,
            'informative_ratio': itr_res,
            'cohesion': cohesion_res,
            'overall_score': round(score, 4),
            'overall_grade': overall,
            'diagnostics': diagnostics,
            'recommendations': recommendations
        }
