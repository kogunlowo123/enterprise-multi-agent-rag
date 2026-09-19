"""Embedding providers.

``HashingEmbedder`` is deterministic and dependency-free, which keeps the test suite
and local demos fully offline. ``OpenAIEmbedder`` is the production option.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol, runtime_checkable

from pydantic import SecretStr

from emrag.errors import ProviderError
from emrag.providers.http import JsonClient
from emrag.text import content_tokens


@runtime_checkable
class Embedder(Protocol):
    """Maps text to fixed-size dense vectors."""

    @property
    def dimension(self) -> int:
        """Length of every produced vector."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of passages."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""


class HashingEmbedder:
    """Signed feature-hashing embedder over unigrams and bigrams.

    It captures lexical overlap rather than semantics, so it works as an offline
    baseline and as a cheap fallback signal alongside BM25.
    """

    def __init__(self, dimension: int = 512, bigram_weight: float = 0.5) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self._dimension = dimension
        self._bigram_weight = bigram_weight

    @property
    def dimension(self) -> int:
        return self._dimension

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return value % self._dimension, sign

    def _embed(self, text: str) -> list[float]:
        tokens = content_tokens(text)
        vector = [0.0] * self._dimension
        for token in tokens:
            index, sign = self._bucket(token)
            vector[index] += sign
        for left, right in pairwise(tokens):
            index, sign = self._bucket(f"{left}_{right}")
            vector[index] += sign * self._bigram_weight
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OpenAIEmbedder:
    """Embeddings via the OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(
        self,
        client: JsonClient,
        *,
        api_key: SecretStr,
        model: str,
        dimension: int,
        base_url: str = "https://api.openai.com/v1",
        batch_size: int = 64,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._model = model
        self._dimension = dimension
        self._base_url = base_url.rstrip("/")
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            body = self._client.request(
                "POST",
                f"{self._base_url}/embeddings",
                json={"model": self._model, "input": batch},
                headers={"Authorization": f"Bearer {self._api_key.get_secret_value()}"},
            )
            try:
                items = sorted(body["data"], key=lambda item: item["index"])
                vectors.extend([float(x) for x in item["embedding"]] for item in items)
            except (KeyError, TypeError) as exc:
                raise ProviderError("unexpected embeddings response shape") from exc
        if len(vectors) != len(texts):
            raise ProviderError("embedding count does not match input count")
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
