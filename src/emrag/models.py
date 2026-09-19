"""Domain models shared across ingestion, retrieval, agents and the CLI."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    """A source document prior to chunking."""

    model_config = ConfigDict(frozen=True)

    id: str
    source: str
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable passage derived from a document."""

    id: str
    doc_id: str
    source: str
    text: str
    position: int
    section: str | None = None


class ScoredChunk(BaseModel):
    """A chunk together with the score assigned by a retriever or reranker."""

    chunk: Chunk
    score: float
    retriever: str


class QueryIntent(str, Enum):
    """Coarse query classes that influence planning."""

    FACTUAL = "factual"
    DEFINITION = "definition"
    PROCEDURE = "procedure"
    COMPARISON = "comparison"


class QueryPlan(BaseModel):
    """Output of the planner agent."""

    original: str
    intent: QueryIntent
    sub_queries: list[str]


class Claim(BaseModel):
    """A single sentence of a draft answer with its grounding verdict."""

    text: str
    citations: list[int] = Field(default_factory=list)
    supported: bool
    support_score: float
    reason: str | None = None


class ValidationReport(BaseModel):
    """Output of the validator agent."""

    claims: list[Claim]
    grounding_score: float
    invalid_citations: list[int] = Field(default_factory=list)
    passed: bool


class Citation(BaseModel):
    """A source passage cited by the final answer."""

    index: int
    chunk_id: str
    source: str
    section: str | None = None
    snippet: str


class TraceEvent(BaseModel):
    """One executed workflow node."""

    node: str
    detail: str = ""
    duration_ms: float = 0.0


class Answer(BaseModel):
    """Final response returned to the caller."""

    query: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    grounding_score: float
    abstained: bool = False
    removed_claims: int = 0
    plan: QueryPlan | None = None
    trace: list[TraceEvent] = Field(default_factory=list)


class IngestReport(BaseModel):
    """Summary of an ingestion run."""

    documents: int = 0
    chunks: int = 0
    quarantined: int = 0
    redactions: int = 0
    skipped: list[str] = Field(default_factory=list)
