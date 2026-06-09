"""POST /v1/embeddings — exercised end-to-end with a fake embedder.

The fake is injected via `app.dependency_overrides[get_embedder]`, so these
tests never load BGE-M3 / torch / FlagEmbedding and never touch the network.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from pharmagpt_vn.api.dependencies import get_embedder
from pharmagpt_vn.api.main import app


@dataclass(frozen=True)
class _Pair:
    """Stand-in for `EmbeddingPair` — the route only reads `.dense`."""

    dense: list[float]
    sparse: dict[int, float]


class _FakeEmbedder:
    """Deterministic embedder returning fixed-length dense vectors.

    Records the texts it was asked to encode so tests can assert the route
    passed inputs through unchanged and in order. The vector for input ``i`` is
    ``[float(i)] * dim`` — distinct per input so order is verifiable.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> list[_Pair]:
        self.calls.append(list(texts))
        return [_Pair(dense=[float(i)] * self.dim, sparse={}) for i, _ in enumerate(texts)]


def _client_with(embedder: _FakeEmbedder) -> TestClient:
    app.dependency_overrides[get_embedder] = lambda: embedder
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_embeddings_returns_dense_vectors_with_correct_dim() -> None:
    fake = _FakeEmbedder(dim=1024)
    client = _client_with(fake)

    resp = client.post(
        "/v1/embeddings",
        json={"inputs": ["paracetamol 500mg", "metformin"]},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["embeddings"]) == 2
    assert body["dim"] == 1024
    assert all(len(v) == 1024 for v in body["embeddings"])
    assert body["model"] == "bge-m3"


def test_embeddings_preserve_input_order() -> None:
    fake = _FakeEmbedder(dim=4)
    client = _client_with(fake)

    resp = client.post("/v1/embeddings", json={"inputs": ["a", "b", "c"]})

    assert resp.status_code == 200
    body = resp.json()
    # Fake encodes input i as [i, i, i, i]; order preserved means strictly 0,1,2.
    assert [v[0] for v in body["embeddings"]] == [0.0, 1.0, 2.0]
    # And the embedder saw the inputs in the same order it was given.
    assert fake.calls == [["a", "b", "c"]]


def test_embeddings_empty_input_is_noop() -> None:
    fake = _FakeEmbedder(dim=1024)
    client = _client_with(fake)

    resp = client.post("/v1/embeddings", json={"inputs": []})

    assert resp.status_code == 200
    body = resp.json()
    assert body["embeddings"] == []
    assert body["dim"] == 0
    # Empty input must not invoke the embedder at all.
    assert fake.calls == []


def test_embeddings_custom_model_name_echoed() -> None:
    fake = _FakeEmbedder(dim=8)
    client = _client_with(fake)

    resp = client.post(
        "/v1/embeddings",
        json={"inputs": ["x"], "model": "bge-m3-custom"},
    )

    assert resp.status_code == 200
    assert resp.json()["model"] == "bge-m3-custom"
