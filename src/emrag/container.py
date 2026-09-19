"""Composition root: builds a :class:`RAGService` from :class:`Settings`."""

from __future__ import annotations

import httpx

from emrag.agents import (
    AnswerAgent,
    ExtractiveGenerator,
    Generator,
    GroundingValidator,
    LLMGenerator,
    PlannerAgent,
    RetrieverAgent,
    ValidatorAgent,
)
from emrag.config import Settings
from emrag.errors import ConfigurationError
from emrag.metrics import MetricsRecorder, NullMetrics
from emrag.pipeline import RAGPipeline
from emrag.providers.embeddings import Embedder, HashingEmbedder, OpenAIEmbedder
from emrag.providers.http import JsonClient
from emrag.providers.llm import AnthropicChatClient, LLMClient, OpenAIChatClient
from emrag.retrieval import BM25Index, HybridRetriever, LexicalReranker
from emrag.service import RAGService
from emrag.stores import InMemoryVectorStore, QdrantVectorStore, VectorStore


def _require_key(value: object, name: str) -> None:
    if value is None:
        raise ConfigurationError(f"{name} must be set for the selected provider")


def build_service(
    settings: Settings,
    *,
    http_client: httpx.Client | None = None,
    metrics: MetricsRecorder | None = None,
) -> RAGService:
    """Assemble the full dependency graph.

    Args:
        settings: Validated runtime configuration.
        http_client: Optional client, mainly for tests that inject a mock transport.
        metrics: Optional metrics sink; defaults to a no-op recorder.

    Raises:
        ConfigurationError: If a selected provider is missing its credentials.
    """
    metrics = metrics or NullMetrics()
    http = JsonClient(
        http_client or httpx.Client(timeout=settings.http_timeout_seconds),
        attempts=settings.retry_attempts,
        min_wait=settings.retry_min_wait,
        max_wait=settings.retry_max_wait,
    )

    embedder: Embedder
    if settings.embedding_provider == "openai":
        _require_key(settings.openai_api_key, "EMRAG_OPENAI_API_KEY")
        assert settings.openai_api_key is not None
        embedder = OpenAIEmbedder(
            http,
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
            dimension=settings.openai_embedding_dim,
            base_url=settings.openai_base_url,
        )
    else:
        embedder = HashingEmbedder()

    store: VectorStore
    if settings.vector_store == "qdrant":
        store = QdrantVectorStore(
            http,
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
            dimension=embedder.dimension,
            api_key=settings.qdrant_api_key,
        )
    else:
        store = InMemoryVectorStore(embedder.dimension, settings.index_path)

    llm: LLMClient | None = None
    if settings.llm_provider == "openai":
        _require_key(settings.openai_api_key, "EMRAG_OPENAI_API_KEY")
        assert settings.openai_api_key is not None
        llm = OpenAIChatClient(
            http,
            api_key=settings.openai_api_key,
            model=settings.openai_chat_model,
            base_url=settings.openai_base_url,
        )
    elif settings.llm_provider == "anthropic":
        _require_key(settings.anthropic_api_key, "EMRAG_ANTHROPIC_API_KEY")
        assert settings.anthropic_api_key is not None
        llm = AnthropicChatClient(
            http,
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            max_tokens=settings.anthropic_max_tokens,
            base_url=settings.anthropic_base_url,
        )

    generator: Generator = LLMGenerator(llm) if llm else ExtractiveGenerator()
    retriever = HybridRetriever(embedder, store, BM25Index())
    pipeline = RAGPipeline(
        PlannerAgent(max_sub_queries=settings.max_sub_queries, llm=llm),
        RetrieverAgent(
            retriever,
            LexicalReranker(settings.min_query_coverage),
            top_k=settings.retrieval_top_k,
            top_n=settings.rerank_top_n,
        ),
        AnswerAgent(generator),
        ValidatorAgent(
            GroundingValidator(
                claim_threshold=settings.min_claim_support,
                min_grounding=settings.min_grounding_score,
            )
        ),
        max_attempts=settings.max_attempts,
        metrics=metrics,
    )
    return RAGService(settings, embedder, store, retriever, pipeline, metrics)
