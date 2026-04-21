"""Phase 8 Track A2 — real `Bgem3Embedder` backed by Ollama.

Network is mocked with respx. Pins:

* Loopback-only: non-loopback base_url rejected before any I/O.
* `/api/embeddings` payload shape (`model` + `prompt`).
* Dim auto-detected from first response, pinned thereafter.
* Returned vectors are L2-normalized.
* Dim drift on subsequent responses is rejected.
* Malformed response bodies raise structured errors.
"""
from __future__ import annotations

import json
import math

import httpx
import pytest
import respx

from memory.embeddings import Bgem3Embedder, EmbedderHostError

pytestmark = pytest.mark.unit


def test_refuses_remote_host() -> None:
    with pytest.raises(EmbedderHostError):
        Bgem3Embedder(base_url="http://evil.example.com:11434")


def test_refuses_non_http_scheme() -> None:
    with pytest.raises(EmbedderHostError):
        Bgem3Embedder(base_url="file:///tmp/ollama")


def test_accepts_loopback_variants() -> None:
    for url in (
        "http://127.0.0.1:11434",
        "http://localhost:11434",
        "http://[::1]:11434",
    ):
        Bgem3Embedder(base_url=url)


def test_dim_raises_before_first_embed() -> None:
    e = Bgem3Embedder(base_url="http://127.0.0.1:11434")
    with pytest.raises(RuntimeError):
        _ = e.dim


def test_embed_detects_dim_and_normalizes() -> None:
    e = Bgem3Embedder(base_url="http://127.0.0.1:11434")
    raw = [3.0, 4.0]  # pre-normalization norm = 5.0
    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": raw})
        )
        vec = e.embed("hello")

    assert e.dim == 2
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-6
    assert vec == pytest.approx([0.6, 0.8])


def test_embed_sends_model_and_prompt() -> None:
    e = Bgem3Embedder(base_url="http://127.0.0.1:11434", model="bge-m3")
    captured: dict[str, object] = {}

    def _handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(200, json={"embedding": [1.0, 0.0, 0.0]})

    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/embeddings").mock(side_effect=_handler)
        e.embed("the quick brown fox")

    assert captured == {"model": "bge-m3", "prompt": "the quick brown fox"}


def test_embed_pins_dim_and_rejects_drift() -> None:
    e = Bgem3Embedder(base_url="http://127.0.0.1:11434")
    with respx.mock() as mock:
        route = mock.post("http://127.0.0.1:11434/api/embeddings").mock(
            side_effect=[
                httpx.Response(200, json={"embedding": [1.0, 0.0]}),
                httpx.Response(200, json={"embedding": [1.0, 0.0, 0.0]}),
            ]
        )
        e.embed("first")
        assert e.dim == 2
        with pytest.raises(ValueError, match="dim drift"):
            e.embed("second")
    assert route.call_count == 2


def test_expected_dim_validated_on_first_call() -> None:
    e = Bgem3Embedder(base_url="http://127.0.0.1:11434", expected_dim=4)
    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": [1.0, 0.0]})
        )
        with pytest.raises(ValueError, match="dim drift"):
            e.embed("hi")


def test_missing_embedding_field_raises() -> None:
    e = Bgem3Embedder(base_url="http://127.0.0.1:11434")
    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/embeddings").mock(
            return_value=httpx.Response(200, json={"nope": []})
        )
        with pytest.raises(ValueError, match="missing non-empty"):
            e.embed("x")


def test_non_object_body_raises() -> None:
    e = Bgem3Embedder(base_url="http://127.0.0.1:11434")
    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/embeddings").mock(
            return_value=httpx.Response(200, json=[1.0, 2.0])
        )
        with pytest.raises(ValueError, match="non-object body"):
            e.embed("x")


def test_http_error_propagates() -> None:
    e = Bgem3Embedder(base_url="http://127.0.0.1:11434")
    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/embeddings").mock(
            return_value=httpx.Response(500, json={"error": "boom"})
        )
        with pytest.raises(httpx.HTTPStatusError):
            e.embed("x")
