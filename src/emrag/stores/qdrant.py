"""Qdrant vector store over the REST API."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from pydantic import SecretStr, ValidationError

from emrag.errors import ProviderError, StoreError
from emrag.models import Chunk, ScoredChunk
from emrag.providers.http import JsonClient

_NAMESPACE = uuid.UUID("6f1d6c1e-3f3b-4a52-9d3a-0d6a1a0c7b11")


def _point_id(chunk_id: str) -> str:
    """Qdrant requires UUID or integer ids; derive a stable UUID from the chunk id."""
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class QdrantVectorStore:
    """Cosine-distance collection stored in a Qdrant server."""

    def __init__(
        self,
        client: JsonClient,
        *,
        url: str,
        collection: str,
        dimension: int,
        api_key: SecretStr | None = None,
    ) -> None:
        self._client = client
        self._base = f"{url.rstrip('/')}/collections/{collection}"
        self._dimension = dimension
        self._headers = {"api-key": api_key.get_secret_value()} if api_key else {}
        self._ensure_collection()

    def _call(self, method: str, suffix: str = "", body: Any = None) -> Any:
        try:
            return self._client.request(
                method, f"{self._base}{suffix}", json=body, headers=self._headers
            )
        except ProviderError as exc:
            raise StoreError(str(exc)) from exc

    def _ensure_collection(self) -> None:
        try:
            info = self._client.request("GET", self._base, headers=self._headers)
        except ProviderError as exc:
            if " 404" not in str(exc):
                raise StoreError(str(exc)) from exc
            self._call(
                "PUT",
                body={"vectors": {"size": self._dimension, "distance": "Cosine"}},
            )
            return
        size = (
            info.get("result", {})
            .get("config", {})
            .get("params", {})
            .get("vectors", {})
            .get("size")
        )
        if size is not None and size != self._dimension:
            raise StoreError(
                f"collection vector size {size} does not match embedder dimension {self._dimension}"
            )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise StoreError("chunks and vectors must have equal length")
        for start in range(0, len(chunks), 128):
            points = [
                {
                    "id": _point_id(chunk.id),
                    "vector": [float(x) for x in vector],
                    "payload": chunk.model_dump(),
                }
                for chunk, vector in zip(
                    chunks[start : start + 128], vectors[start : start + 128], strict=True
                )
            ]
            self._call("PUT", "/points?wait=true", {"points": points})

    def delete_document(self, doc_id: str) -> None:
        self._call(
            "POST",
            "/points/delete?wait=true",
            {"filter": {"must": [{"key": "doc_id", "match": {"value": doc_id}}]}},
        )

    def search(self, vector: Sequence[float], k: int) -> list[ScoredChunk]:
        body = self._call(
            "POST",
            "/points/search",
            {"vector": [float(x) for x in vector], "limit": k, "with_payload": True},
        )
        results: list[ScoredChunk] = []
        for hit in body.get("result", []):
            try:
                chunk = Chunk.model_validate(hit["payload"])
            except (KeyError, ValidationError) as exc:
                raise StoreError("malformed point payload in search result") from exc
            results.append(ScoredChunk(chunk=chunk, score=float(hit["score"]), retriever="dense"))
        return results

    def all_chunks(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        offset: Any = None
        while True:
            body = self._call(
                "POST",
                "/points/scroll",
                {"limit": 256, "offset": offset, "with_payload": True, "with_vector": False},
            )
            result = body.get("result", {})
            for point in result.get("points", []):
                try:
                    chunks.append(Chunk.model_validate(point["payload"]))
                except (KeyError, ValidationError) as exc:
                    raise StoreError("malformed point payload in scroll result") from exc
            offset = result.get("next_page_offset")
            if offset is None:
                return chunks

    def count(self) -> int:
        body = self._call("POST", "/points/count", {"exact": True})
        return int(body.get("result", {}).get("count", 0))

    def save(self) -> None:
        return None
