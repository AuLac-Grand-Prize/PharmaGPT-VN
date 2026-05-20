"""FastAPI DI factories — kept thin so routes are easy to test."""

from functools import lru_cache

from fastapi import Depends

from pharmagpt_vn.core.config import Settings, get_settings
from pharmagpt_vn.core.refusal import Classification, RefusalClassifier
from pharmagpt_vn.models.llm_client import LLMClient
from pharmagpt_vn.models.openai_client import OpenAIClient
from pharmagpt_vn.rag.hyde import HyDEGenerator
from pharmagpt_vn.rag.openrouter_reranker import OpenRouterReranker
from pharmagpt_vn.rag.query_decompose import HeuristicVNDecomposer
from pharmagpt_vn.rag.query_rewrite import HeuristicVNRewriter, MultiQueryRetriever
from pharmagpt_vn.rag.reranker import Reranker
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
    return OpenAIClient(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        model=settings.llm_model_main,
    )


def get_retriever(settings: Settings = Depends(get_settings)) -> HybridRetriever:
    return HybridRetriever(
        qdrant_url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        embedding_model=settings.embedding_model,
    )


def get_reranker(settings: Settings = Depends(get_settings)) -> Reranker:
    return OpenRouterReranker(
        api_key=settings.openrouter_api_key,
        model=settings.reranker_model,
        base_url=settings.openrouter_base_url,
    )


def get_chat_service(
    settings: Settings = Depends(get_settings),
    retriever: HybridRetriever = Depends(get_retriever),
    reranker: Reranker = Depends(get_reranker),
    llm: LLMClient = Depends(get_llm_client),
) -> ChatService:
    rewriter = HeuristicVNRewriter()
    multi_query = MultiQueryRetriever(
        inner=retriever, rewriter=rewriter, n_rewrites=settings.qu_rewrite_n
    )
    decomposer = HeuristicVNDecomposer(max_sub=settings.qu_decompose_max)
    hyde = (
        HyDEGenerator(llm=llm, max_tokens=settings.qu_hyde_max_tokens)
        if settings.qu_hyde_enabled
        else None
    )
    return ChatService(
        retriever=retriever,
        reranker=reranker,
        refusal_classifier=_default_classifier(),
        llm=llm,
        enforce_citations=settings.enforce_citations_for_clinical,
        query_rewriter=rewriter,
        multi_query_retriever=multi_query,
        decomposer=decomposer,
        hyde=hyde,
        per_branch_top_k=settings.qu_per_branch_top_k,
        rrf_k=settings.qu_rrf_k,
    )


def get_disambiguation_service(
    retriever: HybridRetriever = Depends(get_retriever),
) -> DisambiguationService:
    return DisambiguationService(retriever)
