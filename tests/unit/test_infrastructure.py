"""Unit tests for the graph engine, configuration, metrics and logging."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from emrag.config import Settings
from emrag.container import build_service
from emrag.errors import ConfigurationError, GraphError
from emrag.graph import END, WorkflowGraph
from emrag.logging_setup import JsonFormatter, configure_logging, get_logger
from emrag.metrics import InMemoryMetrics, MetricsRecorder, NullMetrics, timed
from emrag.retry import call_with_retry
from tests.conftest import make_settings


class TestGraph:
    def _graph(self) -> WorkflowGraph[list[str]]:
        graph: WorkflowGraph[list[str]] = WorkflowGraph()
        graph.add_node("a", lambda s: [*s, "a"])
        graph.add_node("b", lambda s: [*s, "b"])
        graph.add_node("c", lambda s: [*s, "c"])
        graph.set_entry("a")
        return graph

    def test_static_and_conditional_edges(self) -> None:
        graph = self._graph()
        graph.add_conditional_edges("a", lambda s: "b" if len(s) < 5 else "c")
        graph.add_edge("b", END)
        graph.add_edge("c", END)
        assert graph.run([]) == ["a", "b"]

    def test_loop_until_condition(self) -> None:
        graph = self._graph()
        graph.add_conditional_edges("a", lambda s: END if len(s) >= 3 else "a")
        assert graph.run([]) == ["a", "a", "a"]

    def test_step_budget_prevents_infinite_loop(self) -> None:
        graph = self._graph()
        graph.add_edge("a", "a")
        with pytest.raises(GraphError, match="exceeded"):
            graph.run([], max_steps=5)

    def test_validation_errors(self) -> None:
        graph: WorkflowGraph[list[str]] = WorkflowGraph()
        with pytest.raises(GraphError, match="entry"):
            graph.validate()
        graph = self._graph()
        graph.add_edge("a", "missing")
        with pytest.raises(GraphError, match="unknown node"):
            graph.validate()
        graph = self._graph()
        graph.add_edge("ghost", "a")
        with pytest.raises(GraphError, match="unknown node"):
            graph.validate()
        graph = self._graph()
        graph.add_conditional_edges("ghost", lambda s: END)
        with pytest.raises(GraphError, match="unknown node"):
            graph.validate()

    def test_duplicate_and_reserved_names(self) -> None:
        graph = self._graph()
        with pytest.raises(GraphError):
            graph.add_node("a", lambda s: s)
        with pytest.raises(GraphError):
            graph.add_node(END, lambda s: s)

    def test_missing_outgoing_edge(self) -> None:
        with pytest.raises(GraphError, match="no outgoing"):
            self._graph().run([])

    def test_router_to_unknown_node(self) -> None:
        graph = self._graph()
        graph.add_conditional_edges("a", lambda s: "nowhere")
        with pytest.raises(GraphError, match="unknown node"):
            graph.run([])


class TestSettings:
    def test_defaults_are_offline(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path)
        assert (settings.llm_provider, settings.embedding_provider, settings.vector_store) == (
            "extractive",
            "hashing",
            "memory",
        )
        assert settings.index_path.name == "index.json"

    def test_secrets_are_masked(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path, openai_api_key="sk-supersecretvalue0000000000")
        assert "supersecret" not in repr(settings)
        assert "supersecret" not in settings.model_dump_json()

    def test_env_prefix_and_vendor_alias(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMRAG_MAX_ATTEMPTS", "4")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "abc123")
        settings = Settings(_env_file=None)
        assert settings.max_attempts == 4
        assert settings.anthropic_api_key is not None
        assert settings.anthropic_api_key.get_secret_value() == "abc123"

    def test_rejects_inconsistent_values(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="chunk_overlap"):
            make_settings(tmp_path, chunk_max_chars=200, chunk_overlap_chars=200)
        with pytest.raises(ValidationError, match="retry_max_wait"):
            make_settings(tmp_path, retry_min_wait=5.0, retry_max_wait=1.0)


class TestContainer:
    @pytest.mark.parametrize(
        ("overrides", "name"),
        [
            ({"llm_provider": "openai"}, "EMRAG_OPENAI_API_KEY"),
            ({"embedding_provider": "openai"}, "EMRAG_OPENAI_API_KEY"),
            ({"llm_provider": "anthropic"}, "EMRAG_ANTHROPIC_API_KEY"),
        ],
    )
    def test_missing_credentials_fail_fast(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, str], name: str
    ) -> None:
        for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(ConfigurationError, match=name):
            build_service(make_settings(tmp_path, **overrides))


class TestMetrics:
    def test_in_memory_metrics_aggregate(self) -> None:
        metrics = InMemoryMetrics()
        metrics.incr("hits")
        metrics.incr("hits", 2, outcome="ok")
        metrics.observe("latency", 5.0)
        metrics.observe("latency", 7.0)
        snapshot = metrics.snapshot()
        assert snapshot["counters"] == {"hits": 1.0, "hits{outcome=ok}": 2.0}
        assert snapshot["observations"]["latency"] == {"count": 2, "sum": 12.0, "max": 7.0}  # type: ignore[index]

    def test_timed_records_duration_even_on_error(self) -> None:
        metrics = InMemoryMetrics()
        with pytest.raises(RuntimeError), timed(metrics, "op", stage="x"):
            raise RuntimeError("fail")
        assert len(metrics.observations["op{stage=x}"]) == 1

    def test_protocol_conformance(self) -> None:
        assert isinstance(NullMetrics(), MetricsRecorder)
        NullMetrics().incr("x")
        NullMetrics().observe("x", 1.0)


class TestLogging:
    def test_json_formatter_redacts_and_includes_extras(self) -> None:
        record = logging.LogRecord(
            "emrag.test",
            logging.INFO,
            __file__,
            1,
            "token sk-abcdefghijklmnopqrstuvwxyz123456",
            (),
            None,
        )
        record.tenant = "acme"
        record.note = "Bearer abcdefghijklmnop12345678"
        payload = json.loads(JsonFormatter().format(record))
        assert payload["message"] == "token [REDACTED]"
        assert payload["tenant"] == "acme"
        assert payload["note"] == "[REDACTED]"
        assert payload["level"] == "INFO"

    def test_formatter_includes_redacted_exception(self) -> None:
        try:
            raise ValueError("key sk-abcdefghijklmnopqrstuvwxyz123456")
        except ValueError:
            import sys

            record = logging.LogRecord(
                "emrag", logging.ERROR, __file__, 1, "boom", (), sys.exc_info()
            )
        payload = json.loads(JsonFormatter().format(record))
        assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in payload["exception"]

    def test_configure_is_idempotent_and_text_mode_redacts(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging("INFO", json_output=False)
        configure_logging("INFO", json_output=False)
        logger = get_logger("unit")
        assert len(logging.getLogger("emrag").handlers) == 1
        logger.info("password = hunter2hunter2")
        captured = capsys.readouterr().err
        assert "hunter2" not in captured and "[REDACTED]" in captured
        configure_logging("CRITICAL")

    def test_get_logger_namespacing(self) -> None:
        assert get_logger("x").name == "emrag.x"
        assert get_logger("emrag.y").name == "emrag.y"


class TestRetry:
    def test_returns_value(self) -> None:
        assert call_with_retry(lambda: 7, attempts=2, min_wait=0, max_wait=0) == 7
