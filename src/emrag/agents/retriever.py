"""Retriever agent: multi-query hybrid retrieval followed by reranking."""

from __future__ import annotations

from emrag.agents.state import RAGState
from emrag.models import ScoredChunk
from emrag.retrieval.hybrid import HybridRetriever
from emrag.retrieval.reranker import Reranker


class RetrieverAgent:
    """Run every planned sub-query, merge the pools and rerank against the full question."""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Reranker,
        *,
        top_k: int,
        top_n: int,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._top_k = top_k
        self._top_n = top_n

    def run(self, state: RAGState) -> RAGState:
        """Populate ``state.evidence``. Each retry attempt widens the search."""
        plan = state.plan
        queries = plan.sub_queries if plan else [state.query]
        widen = 1 + state.attempt

        pool: dict[str, ScoredChunk] = {}
        for sub_query in queries:
            for hit in self._retriever.retrieve(sub_query, self._top_k * widen):
                existing = pool.get(hit.chunk.id)
                if existing is None or hit.score > existing.score:
                    pool[hit.chunk.id] = hit

        evidence = self._reranker.rerank(
            [state.query, *queries], list(pool.values()), self._top_n + 2 * state.attempt
        )
        state.evidence = evidence
        state.note = (
            f"{len(pool)} candidates -> {len(evidence)} evidence chunks (attempt {state.attempt})"
        )
        return state
