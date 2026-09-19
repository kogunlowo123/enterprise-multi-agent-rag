"""Typed runtime configuration loaded from environment variables and ``.env`` files.

All settings use the ``EMRAG_`` prefix. Provider API keys additionally accept the
vendor-standard names (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``) so existing
shell environments work without changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    Secrets are held as :class:`pydantic.SecretStr` so they never appear in
    ``repr`` output, logs or serialized dumps.
    """

    model_config = SettingsConfigDict(
        env_prefix="EMRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider selection
    llm_provider: Literal["extractive", "openai", "anthropic"] = "extractive"
    embedding_provider: Literal["hashing", "openai"] = "hashing"
    vector_store: Literal["memory", "qdrant"] = "memory"

    # OpenAI
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("EMRAG_OPENAI_API_KEY", "OPENAI_API_KEY")
    )
    openai_base_url: str = "https://api.openai.com/v1"
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dim: int = Field(default=1536, gt=0)

    # Anthropic
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("EMRAG_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    )
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_model: str = "claude-sonnet-5"
    anthropic_max_tokens: int = Field(default=1024, gt=0)

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "emrag_chunks"

    # Storage
    data_dir: Path = Path(".emrag")

    # Ingestion
    chunk_max_chars: int = Field(default=900, ge=100)
    chunk_overlap_chars: int = Field(default=150, ge=0)
    max_file_bytes: int = Field(default=5_000_000, gt=0)

    # Retrieval
    retrieval_top_k: int = Field(default=12, ge=1)
    rerank_top_n: int = Field(default=5, ge=1)
    min_query_coverage: float = Field(default=0.2, ge=0.0, le=1.0)

    # Agents and validation
    max_sub_queries: int = Field(default=4, ge=1, le=8)
    min_claim_support: float = Field(default=0.6, ge=0.0, le=1.0)
    min_grounding_score: float = Field(default=0.8, ge=0.0, le=1.0)
    max_attempts: int = Field(default=2, ge=0, le=5)

    # Security
    max_query_chars: int = Field(default=2000, ge=1)
    reject_injection_queries: bool = True

    # Networking
    http_timeout_seconds: float = Field(default=30.0, gt=0)
    retry_attempts: int = Field(default=3, ge=1)
    retry_min_wait: float = Field(default=0.5, ge=0)
    retry_max_wait: float = Field(default=8.0, ge=0)

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_json: bool = True

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.chunk_overlap_chars >= self.chunk_max_chars:
            raise ValueError("chunk_overlap_chars must be smaller than chunk_max_chars")
        if self.retry_max_wait < self.retry_min_wait:
            raise ValueError("retry_max_wait must be >= retry_min_wait")
        return self

    @property
    def index_path(self) -> Path:
        """Location of the persisted in-memory index."""
        return self.data_dir / "index.json"
