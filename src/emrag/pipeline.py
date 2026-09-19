"""Workflow wiring for the multi-agent RAG loop.

::

    plan → retrieve → generate → validate ─┬─ passed ───────→ finish
                ↑                          ├─ retries left ─→ escalate ─┐
                └──────────────────────────┘                            │
                                           └─ exhausted ───→ finalize   │
    (retrieve/generate route to abstain when there is nothing to answer from)
"""

from __future__ import annotations

import time
from collections.abc import Callable

from emrag.agents.generator import ABSTAIN_TEXT, AnswerAgent
from emrag.agents.planner import PlannerAgent
from emrag.agents.retriever import RetrieverAgent
from emrag.agents.state import RAGState
from emrag.agents.validator import ValidatorAgent
from emrag.graph import END, WorkflowGraph
from emrag.logging_setup import get_logger
from emrag.metrics import MetricsRecorder, NullMetrics
from emrag.models import Answer, Citation, TraceEvent
from emrag.text import parse_citations

_log = get_logger("pipeline")


class RAGPipeline:
    """Runs one question through planner, retriever, answer and validator agents."""

    def __init__(
        self,
        planner: PlannerAgent,
        retriever: RetrieverAgent,
        answerer: AnswerAgent,
        validator: ValidatorAgent,
        *,
        max_attempts: int = 2,
        snippet_chars: int = 240,
        metrics: MetricsRecorder | None = None,
    ) -> None:
        self._planner = planner
        self._retriever = retriever
        self._answerer = answerer
        self._validator = validator
        self._max_attempts = max_attempts
        self._snippet_chars = snippet_chars
        self._metrics = metrics or NullMetrics()
        self._graph = self._build_graph()

    def run(self, query: str) -> Answer:
        """Answer ``query`` and return the final, validated :class:`Answer`."""
        self._metrics.incr("emrag.queries")
        state = self._graph.run(RAGState(query=query))
        assert state.answer is not None, "every terminal node must set an answer"
        # Terminal nodes build the answer before their own trace event is appended.
        state.answer.trace = list(state.trace)
        outcome = "abstained" if state.answer.abstained else "answered"
        self._metrics.incr("emrag.answers", outcome=outcome)
        self._metrics.observe("emrag.grounding_score", state.answer.grounding_score)
        _log.info(
            "query completed",
            extra={
                "outcome": outcome,
                "grounding": state.answer.grounding_score,
                "attempts": state.attempt + 1,
            },
        )
        return state.answer

    # -- graph construction -------------------------------------------------

    def _build_graph(self) -> WorkflowGraph[RAGState]:
        graph: WorkflowGraph[RAGState] = WorkflowGraph()
        nodes: dict[str, Callable[[RAGState], RAGState]] = {
            "plan": self._plan,
            "retrieve": self._retriever.run,
            "generate": self._answerer.run,
            "validate": self._validator.run,
            "escalate": self._escalate,
            "finalize": self._finalize,
            "finish": self._finish,
            "abstain": self._abstain,
        }
        for name, func in nodes.items():
            graph.add_node(name, self._traced(name, func))
        graph.set_entry("plan")
        graph.add_edge("plan", "retrieve")
        graph.add_conditional_edges("retrieve", lambda s: "generate" if s.evidence else "abstain")
        graph.add_conditional_edges(
            "generate", lambda s: "abstain" if s.draft and s.draft.abstained else "validate"
        )
        graph.add_conditional_edges("validate", self._route_validation)
        graph.add_edge("escalate", "retrieve")
        for terminal in ("finalize", "finish", "abstain"):
            graph.add_edge(terminal, END)
        return graph

    def _traced(
        self, name: str, func: Callable[[RAGState], RAGState]
    ) -> Callable[[RAGState], RAGState]:
        def wrapper(state: RAGState) -> RAGState:
            start = time.perf_counter()
            state.note = ""
            result = func(state)
            elapsed = (time.perf_counter() - start) * 1000.0
            result.trace.append(
                TraceEvent(node=name, detail=result.note, duration_ms=round(elapsed, 3))
            )
            self._metrics.observe("emrag.node_ms", elapsed, node=name)
            return result

        return wrapper

    def _route_validation(self, state: RAGState) -> str:
        assert state.report is not None
        if state.report.passed:
            return "finish"
        return "escalate" if state.attempt < self._max_attempts else "finalize"

    # -- nodes --------------------------------------------------------------

    def _plan(self, state: RAGState) -> RAGState:
        state.plan = self._planner.run(state.query)
        state.note = f"intent={state.plan.intent.value} sub_queries={len(state.plan.sub_queries)}"
        return state

    @staticmethod
    def _escalate(state: RAGState) -> RAGState:
        state.attempt += 1
        state.note = f"validation failed; widening retrieval (attempt {state.attempt})"
        return state

    def _finish(self, state: RAGState) -> RAGState:
        assert state.draft is not None and state.report is not None
        state.answer = self._build_answer(
            state, state.draft.text, state.report.grounding_score, removed=0
        )
        state.note = "answer accepted"
        return state

    def _finalize(self, state: RAGState) -> RAGState:
        """Keep only supported claims once retries are exhausted."""
        assert state.report is not None
        kept = [c for c in state.report.claims if c.supported]
        removed = len(state.report.claims) - len(kept)
        if not kept:
            return self._abstain(state, removed=removed)
        state.answer = self._build_answer(
            state, " ".join(c.text for c in kept), 1.0, removed=removed
        )
        state.note = f"dropped {removed} unsupported claim(s)"
        return state

    def _abstain(self, state: RAGState, removed: int = 0) -> RAGState:
        state.answer = Answer(
            query=state.query,
            text=ABSTAIN_TEXT,
            grounding_score=1.0,
            abstained=True,
            removed_claims=removed,
            plan=state.plan,
            trace=state.trace,
        )
        state.note = "no supported answer"
        return state

    def _build_answer(self, state: RAGState, text: str, grounding: float, removed: int) -> Answer:
        cited = sorted({i for i in parse_citations(text) if 1 <= i <= len(state.evidence)})
        citations = [
            Citation(
                index=i,
                chunk_id=state.evidence[i - 1].chunk.id,
                source=state.evidence[i - 1].chunk.source,
                section=state.evidence[i - 1].chunk.section,
                snippet=state.evidence[i - 1].chunk.text[: self._snippet_chars],
            )
            for i in cited
        ]
        return Answer(
            query=state.query,
            text=text,
            citations=citations,
            grounding_score=grounding,
            removed_claims=removed,
            plan=state.plan,
            trace=state.trace,
        )
