"""Unit tests for vector stores, BM25, hybrid retrieval and reranking."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from emrag.errors import StoreError
from emrag.models import Chunk, ScoredChunk
from emrag.providers import HashingEmbedder
from emrag.retrieval import BM25Index, HybridRetriever, LexicalReranker
from emrag.stores import InMemoryVectorStore, QdrantVectorStore
from tests.conftest import FakeQdrant, json_client


def _chunk(cid: str, text: str, doc: str = "d") -> Chunk:
    return Chunk(id=cid, doc_id=doc, source=f"{doc}.md", text=text, position=0)


CHUNKS = [
    _chunk("a", "Passwords must be rotated every 90 days for privileged accounts.", "sec"),
    _chunk("b", "Hotel stays are capped at 220 dollars per night in major cities.", "travel"),
    _chunk("c", "Severity 1 incidents must be acknowledged within 15 minutes.", "ir"),
]


def _populate(store: InMemoryVectorStore | QdrantVectorStore, embedder: HashingEmbedder) -> None:
    store.upsert(CHUNKS, embedder.embed_documents([c.text for c in CHUNKS]))


class TestInMemoryStore:
    def test_search_returns_best_match_first(self) -> None:
        embedder = HashingEmbedder()
        store = InMemoryVectorStore(embedder.dimension)
        _populate(store, embedder)
        hits = store.search(embedder.embed_query("password rotation"), 2)
        assert hits[0].chunk.id == "a"
        assert hits[0].retriever == "dense"
        assert store.count() == 3

    def test_upsert_replaces_and_delete_document(self) -> None:
        embedder = HashingEmbedder()
        store = InMemoryVectorStore(embedder.dimension)
        _populate(store, embedder)
        store.upsert([_chunk("a", "replaced text", "sec")], embedder.embed_documents(["replaced"]))
        assert store.count() == 3
        store.delete_document("sec")
        assert sorted(c.id for c in store.all_chunks()) == ["b", "c"]

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        embedder = HashingEmbedder()
        path = tmp_path / "nested" / "index.json"
        store = InMemoryVectorStore(embedder.dimension, path)
        _populate(store, embedder)
        store.save()
        reloaded = InMemoryVectorStore(embedder.dimension, path)
        assert reloaded.count() == 3
        assert reloaded.search(embedder.embed_query("hotel night"), 1)[0].chunk.id == "b"

    def test_dimension_mismatch_on_load(self, tmp_path: Path) -> None:
        path = tmp_path / "index.json"
        store = InMemoryVectorStore(8, path)
        store.upsert([CHUNKS[0]], [[1.0] * 8])
        store.save()
        with pytest.raises(StoreError, match="dimension"):
            InMemoryVectorStore(16, path)

    def test_corrupt_index_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "index.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(StoreError, match="cannot read"):
            InMemoryVectorStore(8, path)

    def test_validates_inputs(self) -> None:
        store = InMemoryVectorStore(4)
        with pytest.raises(StoreError, match="equal length"):
            store.upsert([CHUNKS[0]], [])
        with pytest.raises(StoreError, match="dimension"):
            store.upsert([CHUNKS[0]], [[1.0]])
        with pytest.raises(StoreError, match="dimension"):
            store.search([1.0], 1)

    def test_zero_query_and_zero_k_return_nothing(self) -> None:
        store = InMemoryVectorStore(2)
        store.upsert([CHUNKS[0]], [[1.0, 0.0]])
        assert store.search([0.0, 0.0], 3) == []
        assert store.search([1.0, 0.0], 0) == []

    def test_save_without_path_is_noop(self) -> None:
        InMemoryVectorStore(2).save()


class TestQdrantStore:
    def _store(self, fake: FakeQdrant, dim: int) -> QdrantVectorStore:
        return QdrantVectorStore(
            json_client(fake.handler),
            url="http://qdrant:6333/",
            collection=fake.collection,
            dimension=dim,
            api_key=SecretStr("qk-secret"),
        )

    def test_creates_collection_and_round_trips(self) -> None:
        embedder = HashingEmbedder()
        fake = FakeQdrant(embedder.dimension)
        store = self._store(fake, embedder.dimension)
        assert fake.exists
        _populate(store, embedder)
        assert store.count() == 3
        assert store.search(embedder.embed_query("incident acknowledged"), 1)[0].chunk.id == "c"
        assert fake.api_keys == {"qk-secret"}

    def test_scroll_paginates(self) -> None:
        embedder = HashingEmbedder()
        fake = FakeQdrant(embedder.dimension, page_size=1)
        store = self._store(fake, embedder.dimension)
        _populate(store, embedder)
        assert sorted(c.id for c in store.all_chunks()) == ["a", "b", "c"]

    def test_delete_document_removes_points(self) -> None:
        embedder = HashingEmbedder()
        fake = FakeQdrant(embedder.dimension)
        store = self._store(fake, embedder.dimension)
        _populate(store, embedder)
        store.delete_document("travel")
        assert store.count() == 2

    def test_reuses_existing_collection(self) -> None:
        fake = FakeQdrant(8)
        fake.exists = True
        self._store(fake, 8)
        assert ("PUT", "/collections/test") not in fake.requests

    def test_dimension_mismatch_is_rejected(self) -> None:
        fake = FakeQdrant(8)
        fake.exists = True
        with pytest.raises(StoreError, match="does not match"):
            self._store(fake, 16)

    def test_length_mismatch_rejected(self) -> None:
        store = self._store(FakeQdrant(8), 8)
        with pytest.raises(StoreError, match="equal length"):
            store.upsert([CHUNKS[0]], [])

    def test_upstream_failure_becomes_store_error(self) -> None:
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200, json={"result": {}})
            return httpx.Response(400, text="bad request")

        store = QdrantVectorStore(
            json_client(handler), url="http://q", collection="test", dimension=8
        )
        with pytest.raises(StoreError):
            store.count()

    def test_save_is_noop(self) -> None:
        self._store(FakeQdrant(8), 8).save()


class TestBM25:
    def test_ranks_exact_terms_first(self) -> None:
        index = BM25Index()
        index.rebuild(CHUNKS)
        results = index.search("privileged password rotation", 3)
        assert results[0].chunk.id == "a"
        assert all(r.retriever == "bm25" for r in results)
        assert len(index) == 3

    def test_unknown_terms_and_empty_index(self) -> None:
        index = BM25Index()
        assert index.search("anything", 3) == []
        index.rebuild(CHUNKS)
        assert index.search("zzzz qqqq", 3) == []
        assert index.search("the of", 3) == []
        assert index.search("password", 0) == []

    def test_rebuild_replaces_contents(self) -> None:
        index = BM25Index()
        index.rebuild(CHUNKS)
        index.rebuild(CHUNKS[:1])
        assert len(index) == 1


class TestHybridAndRerank:
    def _retriever(self) -> HybridRetriever:
        embedder = HashingEmbedder()
        store = InMemoryVectorStore(embedder.dimension)
        _populate(store, embedder)
        retriever = HybridRetriever(embedder, store, BM25Index())
        retriever.refresh_sparse_index()
        return retriever

    def test_fuses_both_rankings(self) -> None:
        hits = self._retriever().retrieve("how often are passwords rotated", 2)
        assert hits[0].chunk.id == "a"
        assert hits[0].retriever == "hybrid"
        assert hits[0].score > hits[-1].score or len(hits) == 1

    def test_respects_k(self) -> None:
        assert len(self._retriever().retrieve("password hotel incident", 2)) == 2

    def test_reranker_filters_irrelevant_and_orders(self) -> None:
        candidates = [
            ScoredChunk(chunk=c, score=1.0 - i * 0.1, retriever="hybrid")
            for i, c in enumerate(CHUNKS)
        ]
        ranked = LexicalReranker(min_coverage=0.3).rerank(
            ["privileged password rotation"], candidates, 3
        )
        assert [r.chunk.id for r in ranked] == ["a"]
        assert ranked[0].retriever == "reranked"

    def test_reranker_edge_cases(self) -> None:
        reranker = LexicalReranker()
        assert reranker.rerank(["q"], [], 3) == []
        candidate = [ScoredChunk(chunk=CHUNKS[0], score=1.0, retriever="x")]
        assert reranker.rerank(["the of"], candidate, 3) == []
        assert reranker.rerank(["password"], candidate, 0) == []
