"""Planner agent: classifies a question and decomposes it into sub-queries."""

from __future__ import annotations

import json
import re

from emrag.errors import ProviderError
from emrag.logging_setup import get_logger
from emrag.models import QueryIntent, QueryPlan
from emrag.providers.llm import LLMClient
from emrag.text import content_tokens

_log = get_logger("agents.planner")

_INTERROGATIVE = r"(?:what|how|why|when|who|which|where|does|do|is|are|can|should)"
_COMPARE_PATTERNS = (
    re.compile(r"\bdifference(?:s)?\s+between\s+(?P<a>.+?)\s+and\s+(?P<b>.+?)\??$", re.I),
    re.compile(r"\bcompare\s+(?P<a>.+?)\s+(?:with|to|and|against)\s+(?P<b>.+?)\??$", re.I),
    re.compile(r"^(?P<a>.+?)\s+(?:vs\.?|versus)\s+(?P<b>.+?)\??$", re.I),
)

_PLANNER_SYSTEM = (
    "You are a query planner for a retrieval system. Reply with JSON only, in the form "
    '{"intent": "factual|definition|procedure|comparison", "sub_queries": ["..."]}. '
    "Sub-queries must be short, self-contained search queries. Do not answer the question."
)


def classify_intent(query: str) -> QueryIntent:
    """Classify ``query`` using surface patterns."""
    lowered = query.lower()
    if any(p.search(query) for p in _COMPARE_PATTERNS):
        return QueryIntent.COMPARISON
    if re.match(r"\s*(how\s+(do|can|should|to)|steps?\s+to|what\s+is\s+the\s+process)", lowered):
        return QueryIntent.PROCEDURE
    if re.match(r"\s*(what\s+(is|are)|define|meaning\s+of)\b", lowered):
        return QueryIntent.DEFINITION
    return QueryIntent.FACTUAL


def _dedupe(items: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        key = " ".join(content_tokens(item))
        if item and key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique[:limit]


def _rule_based_sub_queries(query: str, intent: QueryIntent) -> list[str]:
    parts = [p.strip() for p in re.split(r"\?\s+|;\s+", query) if p.strip()]
    expanded: list[str] = []
    for part in parts:
        pieces = re.split(rf"\s+and\s+(?={_INTERROGATIVE}\b)", part, flags=re.I)
        # Only split on "and" when every side is a substantive question of its own.
        if len(pieces) > 1 and all(len(content_tokens(p)) >= 2 for p in pieces):
            expanded.extend(p.strip(" ?") for p in pieces)
        elif content_tokens(part):
            expanded.append(part.strip(" ?"))
    sub_queries = expanded or [query.strip(" ?")]

    if intent is QueryIntent.COMPARISON:
        for pattern in _COMPARE_PATTERNS:
            match = pattern.search(query)
            if match:
                sub_queries.extend([match.group("a").strip(" ?"), match.group("b").strip(" ?")])
                break
    return sub_queries


class PlannerAgent:
    """Produce a :class:`QueryPlan`, optionally refined by an LLM.

    The deterministic rule-based plan is always computed first. If an LLM is configured,
    its JSON proposal is validated and merged; any malformed or failed response falls
    back to the rule-based plan, so planning never blocks answering.
    """

    def __init__(self, *, max_sub_queries: int = 4, llm: LLMClient | None = None) -> None:
        self._max = max_sub_queries
        self._llm = llm

    def run(self, query: str) -> QueryPlan:
        intent = classify_intent(query)
        candidates = [query.strip(" ?"), *_rule_based_sub_queries(query, intent)]

        if self._llm is not None:
            proposal = self._propose(query)
            if proposal is not None:
                intent = proposal[0]
                candidates = [query.strip(" ?"), *proposal[1], *candidates[1:]]

        return QueryPlan(
            original=query,
            intent=intent,
            sub_queries=_dedupe(candidates, self._max) or [query],
        )

    def _propose(self, query: str) -> tuple[QueryIntent, list[str]] | None:
        assert self._llm is not None
        try:
            raw = self._llm.complete(_PLANNER_SYSTEM, query)
            payload = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
            intent = QueryIntent(str(payload["intent"]).lower())
            subs = [str(s).strip() for s in payload["sub_queries"] if str(s).strip()]
        except (ProviderError, ValueError, KeyError, TypeError) as exc:
            _log.warning("planner LLM proposal discarded", extra={"reason": type(exc).__name__})
            return None
        return intent, subs
