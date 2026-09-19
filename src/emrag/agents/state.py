"""Shared mutable state passed between workflow nodes."""

from __future__ import annotations

from pydantic import BaseModel, Field

from emrag.models import Answer, QueryPlan, ScoredChunk, TraceEvent, ValidationReport


class Draft(BaseModel):
    """Candidate answer text produced by the generator."""

    text: str
    abstained: bool = False


class RAGState(BaseModel):
    """State threaded through the plan → retrieve → generate → validate loop."""

    query: str
    plan: QueryPlan | None = None
    evidence: list[ScoredChunk] = Field(default_factory=list)
    draft: Draft | None = None
    report: ValidationReport | None = None
    attempt: int = 0
    answer: Answer | None = None
    trace: list[TraceEvent] = Field(default_factory=list)
    note: str = ""
