# ADR 0003: Lexical grounding validator with claim filtering

- Status: Accepted
- Date: 2026-09-19

## Context

Answers must not contain facts unsupported by the retrieved passages. Options are an LLM judge, an NLI
model, or a deterministic check.

## Decision

Validate each sentence deterministically: valid citations, content-word overlap with the cited
passages, and numeric consistency. When validation fails after the retry budget, return only the
supported claims and report how many were removed; abstain if none survive.

## Consequences

- Catches the most damaging real failures (invented figures, uncited assertions, phantom citations)
  at zero cost and with reproducible results.
- Cannot detect a claim that reuses supported words in an unsupported relationship. This is documented
  in the README and SECURITY.md so it is not mistaken for full entailment checking.
- The `ValidatorAgent` depends on `GroundingValidator` through composition, so an NLI or LLM-judge
  implementation can be added alongside it and combined.
