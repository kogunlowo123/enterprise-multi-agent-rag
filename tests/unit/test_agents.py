"""Unit tests for planner, generator and validator agents."""

from __future__ import annotations

import json

from emrag.agents import (
    ABSTAIN_TEXT,
    AnswerAgent,
    ExtractiveGenerator,
    GroundingValidator,
    LLMGenerator,
    PlannerAgent,
    RAGState,
    ValidatorAgent,
)
from emrag.agents.planner import classify_intent
from emrag.models import Chunk, QueryIntent, QueryPlan, ScoredChunk
from tests.conftest import FakeLLM


def _evidence(*texts: str) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(id=f"c{i}", doc_id="d", source=f"doc{i}.md", text=t, position=i),
            score=1.0,
            retriever="reranked",
        )
        for i, t in enumerate(texts, 1)
    ]


class TestPlanner:
    def test_intent_classification(self) -> None:
        assert classify_intent("What is MFA?") is QueryIntent.DEFINITION
        assert classify_intent("How do I rotate a key?") is QueryIntent.PROCEDURE
        assert classify_intent("Compare AES and RSA") is QueryIntent.COMPARISON
        assert classify_intent("Who approves travel?") is QueryIntent.FACTUAL

    def test_splits_compound_questions(self) -> None:
        plan = PlannerAgent().run("How long must passwords be and how often are they rotated?")
        assert plan.sub_queries[0].startswith("How long must passwords be and")
        assert "How long must passwords be" in plan.sub_queries
        assert "how often are they rotated" in plan.sub_queries

    def test_does_not_split_noun_phrases(self) -> None:
        plan = PlannerAgent().run("What are the terms and conditions for travel?")
        assert len(plan.sub_queries) == 1

    def test_comparison_adds_each_side(self) -> None:
        plan = PlannerAgent().run("What is the difference between severity 1 and severity 2?")
        assert plan.intent is QueryIntent.COMPARISON
        assert "severity 1" in plan.sub_queries and "severity 2" in plan.sub_queries

    def test_versus_pattern(self) -> None:
        plan = PlannerAgent().run("remote work vs office work")
        assert plan.intent is QueryIntent.COMPARISON

    def test_caps_sub_queries(self) -> None:
        plan = PlannerAgent(max_sub_queries=2).run(
            "What is encryption? Who owns keys? Where are logs stored? When are backups taken?"
        )
        assert len(plan.sub_queries) == 2

    def test_llm_proposal_is_merged(self) -> None:
        reply = json.dumps(
            {"intent": "procedure", "sub_queries": ["rotate credentials", "audit log"]}
        )
        llm = FakeLLM(planner_reply=f"Sure! {reply}")
        plan = PlannerAgent(llm=llm).run("Explain credential handling")
        assert plan.intent is QueryIntent.PROCEDURE
        assert plan.sub_queries[:3] == [
            "Explain credential handling",
            "rotate credentials",
            "audit log",
        ]

    def test_bad_llm_output_falls_back_to_rules(self) -> None:
        plan = PlannerAgent(llm=FakeLLM(planner_reply="I cannot do that")).run("What is MFA?")
        assert plan.intent is QueryIntent.DEFINITION
        assert plan.sub_queries == ["What is MFA"]

    def test_llm_provider_error_falls_back(self) -> None:
        from emrag.errors import ProviderError

        class Broken:
            def complete(self, system: str, user: str) -> str:
                raise ProviderError("down")

        assert PlannerAgent(llm=Broken()).run("Who approves travel?").sub_queries


class TestGenerators:
    plan = QueryPlan(original="q", intent=QueryIntent.FACTUAL, sub_queries=["q"])

    def test_extractive_quotes_and_cites(self) -> None:
        evidence = _evidence(
            "Hotels are capped at 220 dollars per night. Rental cars need director approval.",
            "Passwords must be rotated every 90 days.",
        )
        text = ExtractiveGenerator().generate(
            "What is the hotel cap per night?",
            QueryPlan(original="q", intent=QueryIntent.FACTUAL, sub_queries=["hotel cap night"]),
            evidence,
        )
        assert text == "Hotels are capped at 220 dollars per night [1]."

    def test_extractive_abstains_without_overlap(self) -> None:
        text = ExtractiveGenerator().generate(
            "swallow velocity", self.plan, _evidence("Unrelated text.")
        )
        assert text == ABSTAIN_TEXT

    def test_extractive_abstains_on_stopword_only_query(self) -> None:
        plan = QueryPlan(original="the of", intent=QueryIntent.FACTUAL, sub_queries=["the of"])
        assert ExtractiveGenerator().generate("the of", plan, _evidence("x y z")) == ABSTAIN_TEXT

    def test_llm_generator_builds_delimited_prompt(self) -> None:
        llm = FakeLLM("Answer [1].")
        text = LLMGenerator(llm).generate(
            "Q?", self.plan, _evidence("Evidence one.", "Evidence two.")
        )
        system, user = llm.calls[0]
        assert text == "Answer [1]."
        assert "<evidence>" in user and "[2] (source: doc2.md)" in user and "Question: Q?" in user
        assert "untrusted" in system

    def test_answer_agent_marks_abstention(self) -> None:
        class Fixed:
            def generate(self, query: str, plan: QueryPlan, evidence: object) -> str:
                return ABSTAIN_TEXT

        state = RAGState(query="q", plan=self.plan)
        assert AnswerAgent(Fixed()).run(state).draft.abstained is True  # type: ignore[union-attr]


class TestValidator:
    evidence = _evidence(
        "Passwords must be at least 14 characters long. Rotation happens every 90 days.",
        "Hotel stays are capped at 220 dollars per night.",
    )

    def test_supported_answer_passes(self) -> None:
        report = GroundingValidator().validate(
            "Passwords must be at least 14 characters long [1].", self.evidence
        )
        assert report.passed and report.grounding_score == 1.0
        assert report.claims[0].citations == [1]

    def test_trailing_citation_style_is_supported(self) -> None:
        report = GroundingValidator().validate(
            "Hotels are capped at 220 dollars per night. [2]", self.evidence
        )
        assert report.passed

    def test_wrong_number_is_flagged(self) -> None:
        report = GroundingValidator().validate(
            "Hotel stays are capped at 500 dollars per night [2].", self.evidence
        )
        assert not report.passed
        assert report.claims[0].reason == "number not in evidence: 500"

    def test_uncited_claim_is_flagged(self) -> None:
        report = GroundingValidator().validate(
            "Passwords must be at least 14 characters long.", self.evidence
        )
        assert not report.passed and report.claims[0].reason == "no citation"

    def test_invalid_citation_is_flagged(self) -> None:
        report = GroundingValidator().validate("Passwords need 14 characters [7].", self.evidence)
        assert report.invalid_citations == [7] and not report.passed

    def test_unsupported_content_is_flagged(self) -> None:
        report = GroundingValidator().validate(
            "Employees receive unlimited vacation and free lunches [1].", self.evidence
        )
        assert not report.claims[0].supported
        assert report.claims[0].reason == "low lexical support"

    def test_mixed_answer_scores_fraction(self) -> None:
        report = GroundingValidator(min_grounding=0.9).validate(
            "Passwords must be at least 14 characters long [1]. Bananas are yellow [1].",
            self.evidence,
        )
        assert report.grounding_score == 0.5 and not report.passed

    def test_empty_draft_fails(self) -> None:
        report = GroundingValidator().validate("", self.evidence)
        assert not report.passed and report.claims == []

    def test_validator_agent_writes_report_and_note(self) -> None:
        state = RAGState(query="q", evidence=self.evidence)
        from emrag.agents import Draft

        state.draft = Draft(text="Hotels are capped at 220 dollars per night [2].")
        result = ValidatorAgent(GroundingValidator()).run(state)
        assert result.report is not None and result.report.passed
        assert "grounding=1.00" in result.note
