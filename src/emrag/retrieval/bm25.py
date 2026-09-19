"""Okapi BM25 sparse index."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

from emrag.models import Chunk, ScoredChunk
from emrag.text import content_tokens


class BM25Index:
    """In-memory BM25 index over chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._chunks: dict[str, Chunk] = {}
        self._tf: dict[str, Counter[str]] = {}
        self._length: dict[str, int] = {}
        self._df: Counter[str] = Counter()
        self._total_length = 0

    def __len__(self) -> int:
        return len(self._chunks)

    def rebuild(self, chunks: Iterable[Chunk]) -> None:
        """Replace the index contents with ``chunks``."""
        self._chunks.clear()
        self._tf.clear()
        self._length.clear()
        self._df.clear()
        self._total_length = 0
        for chunk in chunks:
            tokens = content_tokens(chunk.text)
            counts = Counter(tokens)
            self._chunks[chunk.id] = chunk
            self._tf[chunk.id] = counts
            self._length[chunk.id] = len(tokens)
            self._total_length += len(tokens)
            self._df.update(counts.keys())

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        """Return the ``k`` best-matching chunks, best first."""
        terms = set(content_tokens(query))
        n_docs = len(self._chunks)
        if not terms or n_docs == 0 or k <= 0:
            return []
        avg_len = self._total_length / n_docs or 1.0
        scores: dict[str, float] = {}
        for term in terms:
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for chunk_id, counts in self._tf.items():
                freq = counts.get(term, 0)
                if freq == 0:
                    continue
                norm = 1.0 - self._b + self._b * self._length[chunk_id] / avg_len
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * (
                    freq * (self._k1 + 1.0) / (freq + self._k1 * norm)
                )
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:k]
        return [
            ScoredChunk(chunk=self._chunks[cid], score=score, retriever="bm25")
            for cid, score in ranked
        ]
