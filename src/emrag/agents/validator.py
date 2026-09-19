"""Fact-validator agent: sentence-level grounding and hallucination detection.

Each sentence of a draft is a claim. A claim is supported only if it carries a valid
citation, most of its content words appear in the cited passages, and every number it
states appears in those passages. Anything else is flagged as unsupported.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from emrag.agents.state import RAGState
from emrag.models import Claim, ScoredChunk, ValidationReport
from emrag.text import (
    CITATION,
    content_tokens,
    normalize_citations,
    parse_citations,
    split_sentences,
)

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


class GroundingValidator:
    """Score how well a draft is supported by its cited evidence."""

    def __init__(self, *, claim_threshold: float = 0.6, min_grounding: float = 0.8) -> None:
        self._claim_threshold = claim_threshold
        self._min_grounding = min_grounding

    def validate(self, draft: str, evidence: Sequence[ScoredChunk]) -> ValidationReport:
        """Evaluate ``draft`` against ``evidence`` (citation ``n`` refers to ``evidence[n-1]``)."""
        claims: list[Claim] = []
        invalid: list[int] = []

        for sentence in split_sentences(normalize_citations(draft)):
            body = CITATION.sub("", sentence)
            claim_tokens = set(content_tokens(body))
            if not claim_tokens:
                continue
            cites = parse_citations(sentence)
            bad = [c for c in cites if not 1 <= c <= len(evidence)]
            invalid.extend(bad)

            if not cites:
                claims.append(self._verdict(sentence, cites, False, 0.0, "no citation"))
                continue
            if bad:
                claims.append(self._verdict(sentence, cites, False, 0.0, f"invalid citation {bad}"))
                continue

            cited_text = " ".join(evidence[c - 1].chunk.text for c in cites)
            support = len(claim_tokens & set(content_tokens(cited_text))) / len(claim_tokens)
            missing_numbers = _numbers(body) - _numbers(cited_text)
            if missing_numbers:
                reason = f"number not in evidence: {sorted(missing_numbers)[0]}"
                claims.append(self._verdict(sentence, cites, False, support, reason))
            elif support < self._claim_threshold:
                claims.append(self._verdict(sentence, cites, False, support, "low lexical support"))
            else:
                claims.append(self._verdict(sentence, cites, True, support, None))

        supported = sum(1 for c in claims if c.supported)
        score = supported / len(claims) if claims else 0.0
        return ValidationReport(
            claims=claims,
            grounding_score=round(score, 4),
            invalid_citations=sorted(set(invalid)),
            passed=bool(claims) and score >= self._min_grounding and not invalid,
        )

    @staticmethod
    def _verdict(
        text: str, cites: list[int], supported: bool, score: float, reason: str | None
    ) -> Claim:
        return Claim(
            text=text,
            citations=cites,
            supported=supported,
            support_score=round(score, 4),
            reason=reason,
        )


class ValidatorAgent:
    """Wraps :class:`GroundingValidator` for use as a workflow node."""

    def __init__(self, validator: GroundingValidator) -> None:
        self._validator = validator

    def run(self, state: RAGState) -> RAGState:
        assert state.draft is not None, "answer agent must run before the validator"
        state.report = self._validator.validate(state.draft.text, state.evidence)
        unsupported = sum(1 for c in state.report.claims if not c.supported)
        state.note = (
            f"grounding={state.report.grounding_score:.2f} "
            f"claims={len(state.report.claims)} unsupported={unsupported}"
        )
        return state
