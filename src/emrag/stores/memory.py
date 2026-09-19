"""In-process vector store with optional JSON persistence."""

from __future__ import annotations

import heapq
import json
import math
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ValidationError

from emrag.errors import StoreError
from emrag.models import Chunk, ScoredChunk


class _Item(BaseModel):
    chunk: Chunk
    vector: list[float]


class _Snapshot(BaseModel):
    dimension: int
    items: list[_Item]


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in vector))


class InMemoryVectorStore:
    """Exact cosine-similarity search held in memory.

    Suitable for corpora up to a few hundred thousand chunks. When ``path`` is given,
    :meth:`save` writes an atomic JSON snapshot that is reloaded on construction.
    """

    def __init__(self, dimension: int, path: Path | None = None) -> None:
        self._dimension = dimension
        self._path = path
        self._items: dict[str, _Item] = {}
        if path is not None and path.exists():
            self._load(path)

    def _load(self, path: Path) -> None:
        try:
            snapshot = _Snapshot.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, json.JSONDecodeError) as exc:
            raise StoreError(f"cannot read index at {path}: {exc}") from exc
        if snapshot.dimension != self._dimension:
            raise StoreError(
                f"index dimension {snapshot.dimension} does not match embedder dimension "
                f"{self._dimension}; re-ingest the corpus"
            )
        self._items = {item.chunk.id: item for item in snapshot.items}

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise StoreError("chunks and vectors must have equal length")
        for chunk, vector in zip(chunks, vectors, strict=True):
            if len(vector) != self._dimension:
                raise StoreError(f"vector has dimension {len(vector)}, expected {self._dimension}")
            self._items[chunk.id] = _Item(chunk=chunk, vector=list(vector))

    def delete_document(self, doc_id: str) -> None:
        stale = [cid for cid, item in self._items.items() if item.chunk.doc_id == doc_id]
        for chunk_id in stale:
            del self._items[chunk_id]

    def search(self, vector: Sequence[float], k: int) -> list[ScoredChunk]:
        if len(vector) != self._dimension:
            raise StoreError(f"query has dimension {len(vector)}, expected {self._dimension}")
        query_norm = _norm(vector)
        if query_norm == 0.0 or k <= 0:
            return []
        scored: list[tuple[float, str]] = []
        for chunk_id, item in self._items.items():
            item_norm = _norm(item.vector)
            if item_norm == 0.0:
                continue
            dot = sum(a * b for a, b in zip(vector, item.vector, strict=True))
            scored.append((dot / (query_norm * item_norm), chunk_id))
        top = heapq.nlargest(k, scored, key=lambda pair: (pair[0], pair[1]))
        return [
            ScoredChunk(chunk=self._items[cid].chunk, score=score, retriever="dense")
            for score, cid in top
            if score > 0.0
        ]

    def all_chunks(self) -> list[Chunk]:
        return [item.chunk for item in self._items.values()]

    def count(self) -> int:
        return len(self._items)

    def save(self) -> None:
        if self._path is None:
            return
        snapshot = _Snapshot(dimension=self._dimension, items=list(self._items.values()))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(snapshot.model_dump_json())
            os.replace(tmp_name, self._path)
        except OSError as exc:
            Path(tmp_name).unlink(missing_ok=True)
            raise StoreError(f"cannot write index to {self._path}: {exc}") from exc
