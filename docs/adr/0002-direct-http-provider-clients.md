# ADR 0002: Direct HTTP provider clients

- Status: Accepted
- Date: 2026-09-19

## Context

The system needs chat completions, embeddings and vector search from OpenAI-compatible APIs, Anthropic
and Qdrant. Vendor SDKs each bring their own retry, timeout and error models.

## Decision

Call the three REST APIs through one `JsonClient` built on `httpx`. It applies a single retry policy
(exponential backoff on HTTP 408, 409, 425, 429, 500, 502, 503, 504 and transport errors), a single error taxonomy
(`TransientProviderError`, `ProviderError`) and redacts response bodies before they reach an exception.

## Consequences

- One dependency, uniform behaviour, and every provider is testable with `httpx.MockTransport`.
- The OpenAI client also works with any OpenAI-compatible endpoint through `EMRAG_OPENAI_BASE_URL`.
- New provider features (streaming, tool use, batch APIs) must be implemented by hand. Adopting an SDK
  behind the existing `LLMClient` protocol remains possible without touching agents.
