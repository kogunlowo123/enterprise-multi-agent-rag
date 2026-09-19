"""Unit tests for HTTP retry handling, embedders and chat clients."""

from __future__ import annotations

import json
import math

import httpx
import pytest
from pydantic import SecretStr

from emrag.errors import ProviderError, TransientProviderError
from emrag.providers import (
    AnthropicChatClient,
    HashingEmbedder,
    OpenAIChatClient,
    OpenAIEmbedder,
)
from tests.conftest import json_client

KEY = SecretStr("sk-test-key-000000000000000000")


class TestJsonClient:
    def test_retries_transient_then_succeeds(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(503)
            return httpx.Response(200, json={"ok": True})

        assert json_client(handler).request("GET", "http://x/y") == {"ok": True}
        assert calls["n"] == 3

    def test_raises_transient_after_exhausting_attempts(self) -> None:
        with pytest.raises(TransientProviderError):
            json_client(lambda r: httpx.Response(429), attempts=2).request("GET", "http://x")

    def test_does_not_retry_client_errors(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(400, text="bad key sk-abcdefghijklmnopqrstuvwxyz123456")

        with pytest.raises(ProviderError) as info:
            json_client(handler).request("POST", "http://x", json={})
        assert calls["n"] == 1
        assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in str(info.value)

    def test_transport_errors_are_retried(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom")

        with pytest.raises(TransientProviderError, match="transport error"):
            json_client(handler, attempts=2).request("GET", "http://x")

    def test_non_json_body_is_provider_error(self) -> None:
        with pytest.raises(ProviderError, match="non-JSON"):
            json_client(lambda r: httpx.Response(200, text="<html>")).request("GET", "http://x")


class TestHashingEmbedder:
    def test_vectors_are_unit_length_and_deterministic(self) -> None:
        embedder = HashingEmbedder(dimension=64)
        first = embedder.embed_query("Passwords rotate every 90 days")
        assert first == embedder.embed_query("Passwords rotate every 90 days")
        assert len(first) == embedder.dimension == 64
        assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0, rel_tol=1e-9)

    def test_similar_text_scores_higher(self) -> None:
        embedder = HashingEmbedder()
        query = embedder.embed_query("password rotation policy")
        close = embedder.embed_documents(["Password rotation policy is 90 days"])[0]
        far = embedder.embed_documents(["Hotel stays are capped per night"])[0]
        dot = lambda a, b: sum(x * y for x, y in zip(a, b, strict=True))  # noqa: E731
        assert dot(query, close) > dot(query, far)

    def test_empty_text_gives_zero_vector(self) -> None:
        assert not any(HashingEmbedder(dimension=16).embed_query("the of and"))

    def test_rejects_tiny_dimension(self) -> None:
        with pytest.raises(ValueError):
            HashingEmbedder(dimension=2)


class TestOpenAIEmbedder:
    def test_batches_and_orders_results(self) -> None:
        seen: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            assert request.headers["authorization"] == f"Bearer {KEY.get_secret_value()}"
            seen.append(len(payload["input"]))
            data = [
                {"index": i, "embedding": [float(i), 0.0]}
                for i in reversed(range(len(payload["input"])))
            ]
            return httpx.Response(200, json={"data": data})

        embedder = OpenAIEmbedder(
            json_client(handler), api_key=KEY, model="m", dimension=2, batch_size=2
        )
        vectors = embedder.embed_documents(["a", "b", "c"])
        assert seen == [2, 1]
        assert vectors == [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]

    def test_bad_shape_raises(self) -> None:
        embedder = OpenAIEmbedder(
            json_client(lambda r: httpx.Response(200, json={"nope": 1})),
            api_key=KEY,
            model="m",
            dimension=2,
        )
        with pytest.raises(ProviderError, match="shape"):
            embedder.embed_query("x")

    def test_count_mismatch_raises(self) -> None:
        embedder = OpenAIEmbedder(
            json_client(lambda r: httpx.Response(200, json={"data": []})),
            api_key=KEY,
            model="m",
            dimension=2,
        )
        with pytest.raises(ProviderError, match="count"):
            embedder.embed_documents(["a"])


class TestChatClients:
    def test_openai_chat(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["messages"][0] == {"role": "system", "content": "sys"}
            assert body["temperature"] == 0
            return httpx.Response(200, json={"choices": [{"message": {"content": " hi "}}]})

        client = OpenAIChatClient(json_client(handler), api_key=KEY, model="gpt")
        assert client.complete("sys", "user") == "hi"

    def test_openai_chat_bad_shape(self) -> None:
        client = OpenAIChatClient(
            json_client(lambda r: httpx.Response(200, json={"choices": []})), api_key=KEY, model="m"
        )
        with pytest.raises(ProviderError):
            client.complete("s", "u")

    def test_anthropic_chat(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["x-api-key"] == KEY.get_secret_value()
            assert request.headers["anthropic-version"] == "2023-06-01"
            assert request.url.path == "/v1/messages"
            body = json.loads(request.content)
            assert body["system"] == "sys" and body["messages"][0]["content"] == "user"
            return httpx.Response(
                200,
                json={
                    "content": [
                        {"type": "text", "text": "Hello "},
                        {"type": "text", "text": "there"},
                    ]
                },
            )

        client = AnthropicChatClient(json_client(handler), api_key=KEY, model="claude")
        assert client.complete("sys", "user") == "Hello there"

    def test_anthropic_no_text_blocks(self) -> None:
        client = AnthropicChatClient(
            json_client(lambda r: httpx.Response(200, json={"content": [{"type": "tool_use"}]})),
            api_key=KEY,
            model="m",
        )
        with pytest.raises(ProviderError, match="no text"):
            client.complete("s", "u")

    def test_anthropic_bad_shape(self) -> None:
        client = AnthropicChatClient(
            json_client(lambda r: httpx.Response(200, json={})), api_key=KEY, model="m"
        )
        with pytest.raises(ProviderError, match="shape"):
            client.complete("s", "u")
