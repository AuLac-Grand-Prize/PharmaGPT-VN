"""FastAPI DI factories — kept thin so routes are easy to test."""

from functools import lru_cache

from fastapi import Depends

from pharmagpt_vn.core.config import Settings, get_settings
from pharmagpt_vn.core.refusal import Classification, RefusalClassifier
from pharmagpt_vn.models.llm_client import LLMClient
from pharmagpt_vn.models.vllm_client import VLLMClient
from pharmagpt_vn.rag.reranker import CrossEncoderReranker
from pharmagpt_vn.rag.retriever import HybridRetriever
from pharmagpt_vn.services.chat_service import ChatService
from pharmagpt_vn.services.disambiguation_service import DisambiguationService


class _DefaultRefusalClassifier(RefusalClassifier):
    """Placeholder until distilled classifier ships (Plan §3.5.2).

    Always returns clinical_safe with low confidence — callers still benefit from
    PII redaction + RAG enforcement downstream.
    """

    def classify(self, query: str) -> Classification:
        return Classification(label="clinical_safe", confidence=0.5)


@lru_cache
def _default_classifier() -> RefusalClassifier:
    return _DefaultRefusalClassifier()


def get_llm_client(settings: Settings = Depends(get_settings)) -> LLMClient:
    return VLLMClient(base_url=f"http://localhost:{settings.app_port}", model=settings.base_model)


def get_retriever(settings: Settings = Depends(get_settings)) -> HybridRetriever:
    return HybridRetriever(
        qdrant_url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        embedding_model=settings.embedding_model,
    )


def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


def get_chat_service(
    settings: Settings = Depends(get_settings),
    retriever: HybridRetriever = Depends(get_retriever),
    reranker: CrossEncoderReranker = Depends(get_reranker),
    llm: LLMClient = Depends(get_llm_client),
) -> ChatService:
    return ChatService(
        retriever=retriever,
        reranker=reranker,
        refusal_classifier=_default_classifier(),
        llm=llm,
        enforce_citations=settings.enforce_citations_for_clinical,
    )


def get_disambiguation_service(
    retriever: HybridRetriever = Depends(get_retriever),
) -> DisambiguationService:
    return DisambiguationService(retriever)
