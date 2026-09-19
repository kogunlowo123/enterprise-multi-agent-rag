"""Metrics hooks.

The pipeline emits counters and timing observations through the
:class:`MetricsRecorder` protocol. The default recorder discards them; deployments
can plug in Prometheus, StatsD or OpenTelemetry adapters without touching agent code.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol, runtime_checkable


@runtime_checkable
class MetricsRecorder(Protocol):
    """Sink for counters and value observations."""

    def incr(self, name: str, value: float = 1.0, **tags: str) -> None:
        """Increment a counter."""

    def observe(self, name: str, value: float, **tags: str) -> None:
        """Record a single observation (for example a latency in milliseconds)."""


class NullMetrics:
    """Recorder that drops every event."""

    def incr(self, name: str, value: float = 1.0, **tags: str) -> None:
        return None

    def observe(self, name: str, value: float, **tags: str) -> None:
        return None


class InMemoryMetrics:
    """Recorder that keeps events in process, useful for tests and the CLI."""

    def __init__(self) -> None:
        self.counters: dict[str, float] = defaultdict(float)
        self.observations: dict[str, list[float]] = defaultdict(list)

    @staticmethod
    def _key(name: str, tags: dict[str, str]) -> str:
        if not tags:
            return name
        suffix = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}{{{suffix}}}"

    def incr(self, name: str, value: float = 1.0, **tags: str) -> None:
        self.counters[self._key(name, tags)] += value

    def observe(self, name: str, value: float, **tags: str) -> None:
        self.observations[self._key(name, tags)].append(value)

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-serializable summary of everything recorded so far."""
        return {
            "counters": dict(self.counters),
            "observations": {
                key: {"count": len(vals), "sum": sum(vals), "max": max(vals)}
                for key, vals in self.observations.items()
                if vals
            },
        }


@contextmanager
def timed(metrics: MetricsRecorder, name: str, **tags: str) -> Iterator[None]:
    """Record the wall-clock duration of a block in milliseconds."""
    start = time.perf_counter()
    try:
        yield
    finally:
        metrics.observe(name, (time.perf_counter() - start) * 1000.0, **tags)
