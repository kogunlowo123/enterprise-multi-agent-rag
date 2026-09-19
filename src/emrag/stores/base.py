"""Vector store contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from emrag.models import Chunk, ScoredChunk


@runtime_checkable
class VectorStore(Protocol):
    """Persistent dense-vector index over chunks."""

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Insert or replace chunks by id."""

    def delete_document(self, doc_id: str) -> None:
        """Remove every chunk belonging to ``doc_id``."""

    def search(self, vector: Sequence[float], k: int) -> list[ScoredChunk]:
        """Return the ``k`` nearest chunks by cosine similarity, best first."""

    def all_chunks(self) -> list[Chunk]:
        """Return every stored chunk (used to rebuild the sparse index)."""

    def count(self) -> int:
        """Return the number of stored chunks."""

    def save(self) -> None:
        """Flush state to durable storage. A no-op for stores that write through."""
