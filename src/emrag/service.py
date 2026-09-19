"""Application service: ingestion and question answering behind one facade."""

from __future__ import annotations

from pathlib import Path

from emrag.config import Settings
from emrag.ingestion.chunker import chunk_document
from emrag.ingestion.loader import load_documents
from emrag.logging_setup import get_logger
from emrag.metrics import MetricsRecorder, NullMetrics
from emrag.models import Answer, Chunk, IngestReport
from emrag.pipeline import RAGPipeline
from emrag.providers.embeddings import Embedder
from emrag.retrieval.hybrid import HybridRetriever
from emrag.security import redact_secrets, scan_for_injection, validate_query
from emrag.stores.base import VectorStore

_log = get_logger("service")


class RAGService:
    """Facade combining ingestion, retrieval and the agent pipeline.

    Collaborators are injected, so tests and alternative deployments can substitute
    any embedder, vector store or agent configuration.
    """

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: VectorStore,
        retriever: HybridRetriever,
        pipeline: RAGPipeline,
        metrics: MetricsRecorder | None = None,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._store = store
        self._retriever = retriever
        self._pipeline = pipeline
        self._metrics = metrics or NullMetrics()
        self._retriever.refresh_sparse_index()

    def ingest(self, path: Path) -> IngestReport:
        """Load, sanitize, chunk, embed and index every supported file under ``path``.

        Credentials found in documents are redacted before embedding. Chunks that read
        like instructions to a language model are quarantined and never indexed.
        """
        documents, skipped = load_documents(path, max_file_bytes=self._settings.max_file_bytes)
        report = IngestReport(skipped=skipped)

        for document in documents:
            clean_text, redactions = redact_secrets(document.text)
            report.redactions += redactions
            chunks = chunk_document(
                document.model_copy(update={"text": clean_text}),
                max_chars=self._settings.chunk_max_chars,
                overlap_chars=self._settings.chunk_overlap_chars,
            )
            accepted: list[Chunk] = []
            for chunk in chunks:
                findings = scan_for_injection(chunk.text)
                if findings:
                    report.quarantined += 1
                    _log.warning(
                        "chunk quarantined",
                        extra={"chunk_id": chunk.id, "source": chunk.source, "rules": findings},
                    )
                    continue
                accepted.append(chunk)

            self._store.delete_document(document.id)
            if accepted:
                vectors = self._embedder.embed_documents([c.text for c in accepted])
                self._store.upsert(accepted, vectors)
            report.documents += 1
            report.chunks += len(accepted)

        self._store.save()
        self._retriever.refresh_sparse_index()
        self._metrics.incr("emrag.ingested_chunks", report.chunks)
        _log.info(
            "ingestion completed",
            extra={
                "documents": report.documents,
                "chunks": report.chunks,
                "quarantined": report.quarantined,
                "redactions": report.redactions,
            },
        )
        return report

    def ask(self, query: str) -> Answer:
        """Validate ``query`` and return a grounded, cited :class:`Answer`.

        Raises:
            SecurityError: If the query fails input validation.
        """
        cleaned = validate_query(
            query,
            max_chars=self._settings.max_query_chars,
            reject_injection=self._settings.reject_injection_queries,
        )
        return self._pipeline.run(cleaned)

    def stats(self) -> dict[str, int | str]:
        """Return a small operational summary of the index."""
        return {
            "chunks": self._store.count(),
            "vector_store": self._settings.vector_store,
            "embedding_provider": self._settings.embedding_provider,
            "llm_provider": self._settings.llm_provider,
        }
