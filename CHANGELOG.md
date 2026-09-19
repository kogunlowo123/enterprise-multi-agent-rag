# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-09-19

### Added

- Four-agent pipeline: planner, retriever, answer and validator, orchestrated by a bounded
  state-machine engine with retry-and-widen and claim-filtering fallbacks.
- Hybrid retrieval combining dense vectors and BM25 with reciprocal rank fusion, followed by a lexical
  reranker.
- Sentence-level grounding validation with citation checks and numeric consistency checks.
- Extractive, OpenAI and Anthropic answer generation; hashing and OpenAI embeddings; in-memory and
  Qdrant vector stores.
- Ingestion with sentence-aware chunking, Markdown section tracking, secret redaction and quarantine of
  chunks that resemble prompt injection.
- Input validation, redacting structured logging, metrics hooks and bounded retries for upstream calls.
- `emrag` CLI (`ingest`, `ask`, `stats`), Dockerfile, Compose stack, Makefile and GitHub Actions
  workflows for lint, format, types, tests, dependency audit, CodeQL and build validation.
