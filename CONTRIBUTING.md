# Contributing

Thanks for helping improve this project. This guide covers the workflow and the quality bar.

## Development setup

```bash
git clone <repository-url>
cd enterprise-multi-agent-rag
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

## Checks

Every change must pass the same gates CI enforces:

```bash
make lint        # ruff check + ruff format --check
make typecheck   # mypy --strict
make cov         # pytest with a coverage gate of 80%
```

`make format` applies safe autofixes and formatting.

## Workflow

1. Open an issue for anything larger than a small fix so the design can be discussed first.
2. Branch from `main`: `feature/<short-name>` or `fix/<short-name>`.
3. Keep commits focused. Use imperative, present-tense subjects ("Add cross-encoder reranker").
4. Add or update tests with the change. Bug fixes need a regression test that fails without the fix.
5. Update `CHANGELOG.md` under **Unreleased** and any affected documentation.
6. Open a pull request describing the problem, the approach and how you verified it.

## Code standards

- Python 3.10+, fully type-annotated, `mypy --strict` clean.
- Public modules, classes and functions have docstrings that explain behaviour, not restate names.
- New collaborators are injected through constructors and defined as `Protocol`s where more than one
  implementation is plausible. Wiring lives in `container.py` only.
- Errors raised deliberately derive from `EmragError`. Do not swallow exceptions silently.
- Never log or persist secrets. Route any user-visible error text through `security.redact`.
- Tests must run offline. Use `httpx.MockTransport` (see `tests/conftest.py`) instead of real network
  calls.

## Adding a component

| To add | Implement | Register in |
| ------ | --------- | ----------- |
| Embedding provider | `providers.embeddings.Embedder` | `container.build_service`, `config.Settings` |
| LLM provider | `providers.llm.LLMClient` | `container.build_service`, `config.Settings` |
| Vector store | `stores.base.VectorStore` | `container.build_service`, `config.Settings` |
| Reranker | `retrieval.reranker.Reranker` | `container.build_service` |
| Answer strategy | `agents.generator.Generator` | `container.build_service` |

## Reporting security issues

See [SECURITY.md](SECURITY.md). Do not file public issues for vulnerabilities.

## License

By contributing you agree that your contributions are licensed under the MIT License.
