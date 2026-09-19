"""Lexical reranker that scores query coverage and phrase overlap.

It is intentionally model-free so results are deterministic and auditable. The
``Reranker`` protocol allows a cross-encoder to be swapped in without changing agents.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol, runtime_checkable

from emrag.models import ScoredChunk
from emrag.text import content_tokens


@runtime_checkable
class Reranker(Protocol):
    """Reorders candidate chunks by relevance to one or more queries."""

    def rerank(
        self, queries: Sequence[str], candidates: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        """Return the best ``top_n`` candidates, dropping irrelevant ones."""


def _bigrams(tokens: Sequence[str]) -> set[tuple[str, str]]:
    return set(pairwise(tokens))


class LexicalReranker:
    """Score = weighted query-term coverage + bigram overlap + retriever prior."""

    def __init__(self, min_coverage: float = 0.2) -> None:
        self._min_coverage = min_coverage

    def rerank(
        self, queries: Sequence[str], candidates: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        if not candidates or top_n <= 0:
            return []
        query_sets = [(set(toks), _bigrams(toks)) for toks in map(content_tokens, queries) if toks]
        if not query_sets:
            return []
        max_prior = max(c.score for c in candidates) or 1.0

        rescored: list[ScoredChunk] = []
        for candidate in candidates:
            tokens = content_tokens(candidate.chunk.text)
            token_set, bigram_set = set(tokens), _bigrams(tokens)
            best = 0.0
            best_coverage = 0.0
            for query_tokens, query_bigrams in query_sets:
                coverage = len(query_tokens & token_set) / len(query_tokens)
                phrase = (
                    len(query_bigrams & bigram_set) / len(query_bigrams) if query_bigrams else 0.0
                )
                score = 0.65 * coverage + 0.2 * phrase
                if score > best:
                    best, best_coverage = score, coverage
            if best_coverage < self._min_coverage:
                continue
            final = best + 0.15 * (candidate.score / max_prior)
            rescored.append(ScoredChunk(chunk=candidate.chunk, score=final, retriever="reranked"))

        rescored.sort(key=lambda item: (-item.score, item.chunk.id))
        return rescored[:top_n]
