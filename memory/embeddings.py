"""Embedder protocol + reference implementations.

Phase 7 step 2 introduced the contract; Phase 8 Track A2 wired the
real bge-m3 path to Ollama's `/api/embeddings` endpoint.

* `Embedder` — `Protocol` describing the surface (`dim`, `embed(text)`).
* `FakeEmbedder` — deterministic hash-based vectors. Used by tests
  and the bootstrap path so the system is queryable before bge-m3
  weights are pulled. Vectors are L2-normalized so cosine similarity
  collapses to a dot product.
* `Bgem3Embedder` — calls Ollama `/api/embeddings`. Loopback-only
  (same SSRF guard as `OllamaClient`). Dim is detected on first
  response and pinned; subsequent responses of a different length
  raise. Vectors are always L2-normalized before returning.

The Tier 2 store stores vectors as raw float32 BLOBs (atamai
pattern). All embedders MUST return `list[float]` of length `dim`;
the store enforces that at the boundary.
"""
from __future__ import annotations

import hashlib
import logging
import math
import struct
from typing import Final, Protocol, runtime_checkable
from urllib.parse import urlparse

import httpx

_logger = logging.getLogger(__name__)

DEFAULT_MODEL = "bge-m3"
DEFAULT_DIM = 64
"""Fake-embedder dimensionality. Real bge-m3 returns 1024."""

_ALLOWED_HOSTS: Final[frozenset[str]] = frozenset({"127.0.0.1", "localhost", "::1"})
_EMBED_TIMEOUT: Final[httpx.Timeout] = httpx.Timeout(
    connect=2.0, read=30.0, write=10.0, pool=5.0
)


class EmbedderHostError(ValueError):
    """Raised when the embedder base URL points at a non-loopback host."""


def _validate_local(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https"):
        raise EmbedderHostError(
            f"embedder base_url must be http/https, got {parsed.scheme!r}"
        )
    host = (parsed.hostname or "").lower()
    if host not in _ALLOWED_HOSTS:
        raise EmbedderHostError(
            f"embedder base_url host {host!r} is not loopback; refusing to connect"
        )
    return base_url.rstrip("/")


def vec_to_blob(vec: list[float]) -> bytes:
    """Pack a float32 vector for BLOB storage."""
    return struct.pack(f"{len(vec)}f", *vec)


def blob_to_vec(blob: bytes, dim: int) -> list[float]:
    """Unpack a BLOB into a python list. Raises if length mismatches."""
    expected = dim * 4
    if len(blob) != expected:
        raise ValueError(
            f"embedding blob length {len(blob)} != expected {expected} for dim={dim}"
        )
    return list(struct.unpack(f"{dim}f", blob))


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedder surface. Implementations must return L2-normalized vectors."""

    @property
    def dim(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


class FakeEmbedder:
    """Deterministic hash → vector. No model weights, no I/O.

    Same `text` always produces the same vector; different texts
    diverge. Useful for tests and for bootstrapping the index before
    bge-m3 is available.
    """

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        # Stretch SHA-256 with counter rounds until we have `dim` floats.
        raw = bytearray()
        counter = 0
        while len(raw) < self._dim * 4:
            h = hashlib.sha256(f"{counter}|{text}".encode()).digest()
            raw.extend(h)
            counter += 1
        floats: list[float] = []
        for i in range(self._dim):
            chunk = bytes(raw[i * 4 : i * 4 + 4])
            # Map uint32 → [-1, 1]. Stable across platforms.
            n = int.from_bytes(chunk, "big", signed=False)
            floats.append((n / 0xFFFFFFFF) * 2.0 - 1.0)
        return _l2_normalize(floats)


class Bgem3Embedder:
    """bge-m3 via Ollama `/api/embeddings`.

    Dim is detected on the first successful call and pinned. The
    `dim` property raises before that first call — the tier-2 store
    only needs it at index time, so calling `embed()` before reading
    `dim` is the expected sequence. Callers that need a pre-bound
    `dim` (e.g. a persisted index with a fixed schema) should pass
    `expected_dim` to the constructor and the first response is
    validated against it.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = DEFAULT_MODEL,
        *,
        expected_dim: int | None = None,
    ) -> None:
        self._base_url = _validate_local(base_url)
        self.model = model
        self._dim: int | None = expected_dim

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError(
                "Bgem3Embedder.dim is unknown until the first embed() call; "
                "call embed() first or pass expected_dim to the constructor"
            )
        return self._dim

    def embed(self, text: str) -> list[float]:
        with httpx.Client(
            base_url=self._base_url, timeout=_EMBED_TIMEOUT, follow_redirects=False
        ) as client:
            resp = client.post(
                "/api/embeddings", json={"model": self.model, "prompt": text}
            )
            resp.raise_for_status()
            body = resp.json()
        if not isinstance(body, dict):
            raise ValueError(
                f"ollama embeddings returned non-object body: {type(body).__name__}"
            )
        raw = body.get("embedding")
        if not isinstance(raw, list) or not raw:
            raise ValueError("ollama embeddings response missing non-empty 'embedding' array")
        try:
            vec = [float(x) for x in raw]
        except (TypeError, ValueError) as exc:
            raise ValueError("ollama embeddings response contained non-numeric values") from exc

        if self._dim is None:
            self._dim = len(vec)
        elif len(vec) != self._dim:
            raise ValueError(
                f"ollama embeddings dim drift: got {len(vec)}, expected {self._dim}"
            )
        return _l2_normalize(vec)


def build_embedder(*, expected_dim: int = 1024, try_bgem3: bool = True) -> Embedder:
    """Return Bgem3Embedder when available, FakeEmbedder otherwise.

    Pass ``try_bgem3=False`` when Ollama is already known to be unreachable
    to skip the instantiation attempt and go straight to FakeEmbedder.
    All callers that write to the same store must use this helper so the
    embedder dim is consistent across the chat pipeline, vault indexer,
    and scheduled reindex.
    """
    if try_bgem3:
        try:
            return Bgem3Embedder(expected_dim=expected_dim)
        except Exception:
            _logger.exception("build_embedder.bgem3_unavailable; using FakeEmbedder")
    return FakeEmbedder()
