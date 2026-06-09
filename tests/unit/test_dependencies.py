"""api/dependencies.py — DI factories for embedder, retriever, classifier.

These assert the wiring the SPEC requires (R3, R6) without instantiating real
`Settings` (which needs env vars): the factories only read attributes, so a
lightweight namespace stands in. No model load, no Qdrant, no network.
"""

from __future__ import annotations

from types import SimpleNamespace

import pharmagpt_vn.rag.embeddings as embeddings_mod
from pharmagpt_vn.api.dependencies import (
    _default_classifier,
    get_chat_service,
    get_disambiguation_service,
    get_embedder,
    get_llm_client,
    get_reranker,
    get_retriever,
)
from pharmagpt_vn.core.refusal import HeuristicRefusalClassifier
from pharmagpt_vn.models.openai_client import OpenAIClient
from pharmagpt_vn.rag.embeddings import BGEM3Embedder
from pharmagpt_vn.rag.openrouter_reranker import OpenRouterReranker
from pharmagpt_vn.rag.retriever import HybridRetriever, QdrantBackend
from pharmagpt_vn.services.chat_service import ChatService
from pharmagpt_vn.services.disambiguation_service import DisambiguationService


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        embedding_model="BAAI/bge-m3",
        embedder_device="cpu",
        qdrant_url="http://localhost:6333",
        qdrant_api_key=None,
        qdrant_collection="vn_pharma_corpus",
        qu_rrf_k=60,
        # LLM + reranker + query-understanding knobs for the service factories.
        openai_base_url="http://llm/v1",
        openai_api_key="k",
        llm_model_main="m",
        openrouter_api_key="ork",
        reranker_model="cohere/rerank-v3.5",
        openrouter_base_url="https://openrouter.ai/api/v1",
        qu_rewrite_n=3,
        qu_decompose_max=4,
        qu_hyde_enabled=True,
        qu_hyde_max_tokens=256,
        qu_per_branch_top_k=30,
        enforce_citations_for_clinical=True,
    )


def test_get_embedder_returns_bgem3_without_loading(monkeypatch) -> None:
    # Guard: constructing the embedder must NOT call _load() (no model download).
    def _boom(self):
        raise AssertionError("model load triggered at construction")

    monkeypatch.setattr(embeddings_mod.BGEM3Embedder, "_load", _boom)

    emb = get_embedder(_settings())
    assert isinstance(emb, BGEM3Embedder)
    assert emb.model_name == "BAAI/bge-m3"
    assert emb.device == "cpu"
    # Model not loaded yet.
    assert emb._model is None


def test_get_retriever_wires_embedder_and_backend() -> None:
    retriever = get_retriever(_settings())
    assert isinstance(retriever, HybridRetriever)
    # R6 acceptance: both must be wired (no silent placeholder returning []).
    assert retriever._embedder is not None
    assert retriever._backend is not None
    assert isinstance(retriever._embedder, BGEM3Embedder)
    assert isinstance(retriever._backend, QdrantBackend)


def test_get_retriever_construction_does_no_io(monkeypatch) -> None:
    # Neither the embedder's model nor the Qdrant client may be created eagerly.
    monkeypatch.setattr(
        embeddings_mod.BGEM3Embedder,
        "_load",
        lambda self: (_ for _ in ()).throw(AssertionError("loaded")),
    )
    monkeypatch.setattr(
        QdrantBackend,
        "_connect",
        lambda self: (_ for _ in ()).throw(AssertionError("connected")),
    )
    retriever = get_retriever(_settings())
    # Reaching here means no eager load/connect happened.
    assert retriever._backend is not None


def test_default_classifier_is_heuristic() -> None:
    clf = _default_classifier()
    assert isinstance(clf, HeuristicRefusalClassifier)
    # Sanity: it actually discriminates (not the old always-clinical_safe stub).
    assert clf.classify("thời tiết hôm nay?").label == "out_of_scope"
    assert clf.classify("liều metformin?").label == "clinical_safe"


# --- Remaining DI factories construct their objects without I/O -------------
# (Each provider just wires config into a lazily-connecting object.)


def test_get_llm_client_builds_openai_client() -> None:
    llm = get_llm_client(_settings())
    assert isinstance(llm, OpenAIClient)
    assert llm.base_url == "http://llm/v1"
    assert llm.model == "m"


def test_get_reranker_builds_openrouter_reranker() -> None:
    reranker = get_reranker(_settings())
    assert isinstance(reranker, OpenRouterReranker)


def test_get_chat_service_wires_full_pipeline() -> None:
    svc = get_chat_service(
        settings=_settings(),
        retriever=get_retriever(_settings()),
        reranker=get_reranker(_settings()),
        llm=get_llm_client(_settings()),
    )
    assert isinstance(svc, ChatService)
    # Heuristic classifier is the one wired into the service.
    assert isinstance(svc._classifier, HeuristicRefusalClassifier)
    # HyDE enabled in settings -> branch C generator present.
    assert svc._hyde is not None


def test_get_chat_service_omits_hyde_when_disabled() -> None:
    settings = _settings()
    settings.qu_hyde_enabled = False
    svc = get_chat_service(
        settings=settings,
        retriever=get_retriever(settings),
        reranker=get_reranker(settings),
        llm=get_llm_client(settings),
    )
    assert svc._hyde is None


def test_get_disambiguation_service_wraps_retriever() -> None:
    svc = get_disambiguation_service(retriever=get_retriever(_settings()))
    assert isinstance(svc, DisambiguationService)
