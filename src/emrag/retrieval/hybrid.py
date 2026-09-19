"""Hybrid dense + sparse retrieval fused with reciprocal rank fusion."""

from __future__ import annotations

from emrag.models import Chunk, ScoredChunk
from emrag.providers.embeddings import Embedder
from emrag.retrieval.bm25 import BM25Index
from emrag.stores.base import VectorStore


class HybridRetriever:
    """Combine vector search and BM25 using weighted reciprocal rank fusion (RRF)."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        sparse: BM25Index,
        *,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._sparse = sparse
        self._rrf_k = rrf_k
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight

    def refresh_sparse_index(self) -> None:
        """Rebuild BM25 from the vector store contents."""
        self._sparse.rebuild(self._store.all_chunks())

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        """Return up to ``k`` chunks ranked by fused score, best first."""
        pool = max(k * 3, k)
        dense = self._store.search(self._embedder.embed_query(query), pool)
        sparse = self._sparse.search(query, pool)

        fused: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for weight, results in ((self._dense_weight, dense), (self._sparse_weight, sparse)):
            for rank, result in enumerate(results, start=1):
                chunk_id = result.chunk.id
                chunks[chunk_id] = result.chunk
                fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (self._rrf_k + rank)

        ranked = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))[:k]
        return [
            ScoredChunk(chunk=chunks[cid], score=score, retriever="hybrid") for cid, score in ranked
        ]
