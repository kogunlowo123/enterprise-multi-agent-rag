"""Shared fixtures and fakes."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from emrag.config import Settings
from emrag.container import build_service
from emrag.providers.http import JsonClient
from emrag.service import RAGService

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CORPUS = REPO_ROOT / "data" / "sample_corpus"


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    """Build settings that ignore the developer's environment and .env file."""
    base: dict[str, object] = {
        "data_dir": tmp_path / "index",
        "retry_min_wait": 0.0,
        "retry_max_wait": 0.0,
        "log_level": "CRITICAL",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def json_client(
    handler: Callable[[httpx.Request], httpx.Response], attempts: int = 3
) -> JsonClient:
    """A JsonClient backed by an in-process mock transport."""
    return JsonClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        attempts=attempts,
        min_wait=0.0,
        max_wait=0.0,
    )


class FakeLLM:
    """Scripted LLM: returns queued replies, records every call."""

    def __init__(self, *replies: str, planner_reply: str | None = None) -> None:
        self.replies = list(replies)
        self.planner_reply = planner_reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if "query planner" in system:
            if self.planner_reply is None:
                return "not json"
            return self.planner_reply
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


class FakeQdrant:
    """Minimal in-memory imitation of the Qdrant REST API used by the store."""

    def __init__(self, dim: int, collection: str = "test", page_size: int = 2) -> None:
        self.dim = dim
        self.collection = collection
        self.page_size = page_size
        self.exists = False
        self.points: dict[str, dict[str, object]] = {}
        self.requests: list[tuple[str, str]] = []
        self.api_keys: set[str | None] = set()

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.method, request.url.path))
        self.api_keys.add(request.headers.get("api-key"))
        base = f"/collections/{self.collection}"
        path = request.url.path
        body = json.loads(request.content) if request.content else {}

        if path == base and request.method == "GET":
            if not self.exists:
                return httpx.Response(404, json={"status": {"error": "not found"}})
            return httpx.Response(
                200, json={"result": {"config": {"params": {"vectors": {"size": self.dim}}}}}
            )
        if path == base and request.method == "PUT":
            self.exists = True
            return httpx.Response(200, json={"result": True})
        if path == f"{base}/points" and request.method == "PUT":
            for point in body["points"]:
                self.points[point["id"]] = point
            return httpx.Response(200, json={"result": {"status": "completed"}})
        if path == f"{base}/points/delete":
            doc_id = body["filter"]["must"][0]["match"]["value"]
            self.points = {
                pid: p
                for pid, p in self.points.items()
                if p["payload"]["doc_id"] != doc_id  # type: ignore[index]
            }
            return httpx.Response(200, json={"result": {"status": "completed"}})
        if path == f"{base}/points/search":
            query = body["vector"]
            hits = []
            for pid, point in self.points.items():
                vec = point["vector"]
                score = sum(a * b for a, b in zip(query, vec, strict=True))  # type: ignore[arg-type]
                hits.append({"id": pid, "score": score, "payload": point["payload"]})
            hits.sort(key=lambda h: -h["score"])
            return httpx.Response(200, json={"result": hits[: body["limit"]]})
        if path == f"{base}/points/scroll":
            ordered = sorted(self.points.values(), key=lambda p: str(p["id"]))
            start = int(body["offset"] or 0)
            page = ordered[start : start + self.page_size]
            nxt = start + self.page_size if start + self.page_size < len(ordered) else None
            return httpx.Response(
                200,
                json={
                    "result": {
                        "points": [{"payload": p["payload"]} for p in page],
                        "next_page_offset": nxt,
                    }
                },
            )
        if path == f"{base}/points/count":
            return httpx.Response(200, json={"result": {"count": len(self.points)}})
        return httpx.Response(500, json={"error": f"unhandled {request.method} {path}"})


@pytest.fixture
def sample_corpus() -> Path:
    return SAMPLE_CORPUS


@pytest.fixture
def service(tmp_path: Path) -> RAGService:
    """Ingested service using the offline extractive configuration."""
    svc = build_service(make_settings(tmp_path))
    svc.ingest(SAMPLE_CORPUS)
    return svc
