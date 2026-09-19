# Security Policy

## Supported versions

Security fixes are released for the latest minor version on the `main` branch.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |

## Reporting a vulnerability

Do not open a public issue for security reports. Use GitHub's private vulnerability reporting (the **Report a vulnerability** button on this
repository's **Security** tab) and include:

- a description of the issue and its impact,
- the affected version or commit,
- a minimal reproduction (input corpus, query, configuration),
- any suggested mitigation.

You can expect an acknowledgement within 3 business days and a triage decision within 10 business
days. Confirmed issues are fixed on a private branch, released with an advisory, and credited to the
reporter unless they prefer otherwise.

## Scope

In scope: the `emrag` package, its Docker image, and the CI configuration in this repository.
Out of scope: vulnerabilities in third-party model providers, Qdrant, or Python itself; report those
upstream.

## Security controls

| Threat | Control | Location |
| ------ | ------- | -------- |
| Prompt injection in user queries | Pattern screening; rejected before any agent runs (`SecurityError`) | `security.validate_query` |
| Prompt injection in corpus text | Matching chunks are quarantined at ingestion and never indexed | `service.RAGService.ingest` |
| Injection that survives screening | Evidence is delimited and declared untrusted in the model prompt; the validator rejects claims not supported by cited passages | `agents/generator.py`, `agents/validator.py` |
| Credentials in documents | Token formats and `key=value` secrets are redacted before embedding or storage | `security.redact_secrets` |
| Credentials in logs and errors | All log records and CLI errors pass through redaction; API keys are `SecretStr` | `logging_setup.py`, `config.py` |
| Oversized or malformed input | Query length cap, control-character stripping, file size limit, binary detection | `security.py`, `ingestion/loader.py` |
| Path traversal via symlinks | Symbolic links are never followed during ingestion | `ingestion/loader.py` |
| Runaway agent loops | Bounded retry attempts and a hard workflow step budget | `pipeline.py`, `graph.py` |
| Unreliable upstreams | Bounded exponential backoff on 429/5xx and transport errors only | `retry.py`, `providers/http.py` |
| Vulnerable dependencies | `pip-audit` in CI, Dependabot, CodeQL | `.github/` |
| Container hardening | Multi-stage build, non-root user, no build tooling in the runtime image | `Dockerfile` |

## Known limitations

- Injection screening is heuristic. It reduces risk; it does not eliminate it. Treat model output as
  untrusted when it feeds downstream automation.
- Secret redaction is pattern based and can miss unusual credential formats.
- Grounding validation is lexical (content-word overlap plus number matching). It catches invented
  figures and unsupported claims, but it is not a semantic entailment check.
- `EMRAG_OPENAI_BASE_URL`, `EMRAG_ANTHROPIC_BASE_URL` and `EMRAG_QDRANT_URL` are trusted
  configuration. Do not let untrusted parties set them.
- The on-disk index (`index.json`) is not encrypted. Use disk-level encryption or a Qdrant deployment
  with access control for sensitive corpora.
