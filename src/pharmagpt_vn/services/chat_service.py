"""Chat orchestrator — refusal → PII redact → retrieve → rerank → generate → validate."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from pharmagpt_vn.core.guardrails import (
    MEDICAL_DISCLAIMER,
    is_clinical_query,
    redact_pii,
)
from pharmagpt_vn.core.refusal import (
    REFUSAL_TEMPLATES,
    Classification,
    RefusalClassifier,
    should_refuse,
)
from pharmagpt_vn.core.tracing import NoopTracer, Tracer
from pharmagpt_vn.core.validators import (
    DosageRange,
    ValidationResult,
    validate_citations,
    validate_dosage_sanity,
    validate_drug_names,
    validate_tone,
)
from pharmagpt_vn.models.llm_client import GenerationRequest, LLMClient
from pharmagpt_vn.rag.crag import CRAGResult, RelevanceGrader
from pharmagpt_vn.rag.query_rewrite import QueryRewriter
from pharmagpt_vn.rag.reranker import CrossEncoderReranker, RerankedChunk
from pharmagpt_vn.rag.retriever import HybridRetriever, RetrievedChunk
from pharmagpt_vn.services.prompt import build_chat_prompt


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class ChatTrace:
    classification: Classification
    rag_used: bool
    chunks_retrieved: int
    chunks_after_rerank: int
    validators: list[ValidationResult] = field(default_factory=list)
    crag: CRAGResult | None = None
    retries_used: int = 0
    rewritten_query: str | None = None


@dataclass(frozen=True)
class ChatResult:
    content: str
    citations: list[str] = field(default_factory=list)
    trace: ChatTrace | None = None
    refused: bool = False
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatService:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        refusal_classifier: RefusalClassifier,
        llm: LLMClient,
        known_drugs: Iterable[str] = (),
        dosage_ranges: Iterable[DosageRange] = (),
        enforce_citations: bool = True,
        always_on_rag: bool = True,
        crag_grader: RelevanceGrader | None = None,
        query_rewriter: QueryRewriter | None = None,
        crag_max_retries: int = 1,
        tracer: Tracer | None = None,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._classifier = refusal_classifier
        self._llm = llm
        self._known_drugs = list(known_drugs)
        self._dosage_ranges = list(dosage_ranges)
        self._enforce_citations = enforce_citations
        self._always_on_rag = always_on_rag
        self._crag_grader = crag_grader
        self._query_rewriter = query_rewriter
        self._crag_max_retries = max(0, crag_max_retries)
        self._tracer = tracer or NoopTracer()

    async def complete(
        self,
        messages: list[ChatMessage],
        rag_top_k: int = 5,
        rerank_keep: int = 5,
        retrieve_pool: int = 50,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> ChatResult:
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        if last_user is None:
            return ChatResult(content="", refused=True)

        # Refusal classifier (Plan §3.5.2) — runs before PII strip so it sees raw input.
        with self._tracer.start_span("classify") as sp:
            cls = self._classifier.classify(last_user.content)
            sp.set_attribute("label", cls.label)
            sp.set_attribute("confidence", cls.confidence)
        if should_refuse(cls):
            template = REFUSAL_TEMPLATES.get(cls.label, REFUSAL_TEMPLATES["ambiguous"])
            return ChatResult(
                content=template,
                refused=True,
                trace=ChatTrace(
                    classification=cls, rag_used=False, chunks_retrieved=0, chunks_after_rerank=0
                ),
            )

        query = redact_pii(last_user.content)

        # Always-on RAG for clinical queries (Plan §3.5.1) — refuse if no chunks.
        is_clinical = is_clinical_query(query) or cls.label == "clinical_high_risk"
        retrieved: list[RetrievedChunk] = []
        reranked: list[RerankedChunk] = []
        crag_result: CRAGResult | None = None
        active_query = query
        retries_used = 0
        rewritten_query: str | None = None

        if self._always_on_rag or is_clinical:
            retrieved, reranked = await self._retrieve_and_rerank(
                active_query, retrieve_pool, rerank_keep
            )
            # CRAG retry-with-rewrite (Yan et al. 2024 §3.2): when the grader
            # is unsure, give retrieval one more chance with a rewritten query
            # before refusing. Bounded — clinical safety > recall.
            if (
                self._crag_grader is not None
                and is_clinical
                and reranked
                and self._query_rewriter is not None
            ):
                crag_result = self._crag_grader.grade(active_query, reranked)
                while (
                    not crag_result.is_sufficient
                    and retries_used < self._crag_max_retries
                ):
                    rewrites = self._query_rewriter.rewrite(query, n=1)
                    if not rewrites:
                        break
                    rewritten_query = rewrites[0]
                    new_retrieved, new_reranked = await self._retrieve_and_rerank(
                        rewritten_query, retrieve_pool, rerank_keep
                    )
                    if not new_reranked:
                        retries_used += 1
                        continue
                    new_grade = self._crag_grader.grade(rewritten_query, new_reranked)
                    retries_used += 1
                    # Adopt rewrite results only if they're at least as good.
                    if _label_rank(new_grade.label) > _label_rank(crag_result.label):
                        retrieved, reranked, crag_result = new_retrieved, new_reranked, new_grade
                        active_query = rewritten_query
                    if crag_result.is_sufficient:
                        break
            elif self._crag_grader is not None and is_clinical and reranked:
                crag_result = self._crag_grader.grade(active_query, reranked)

        if is_clinical and self._enforce_citations and not reranked:
            return ChatResult(
                content=(
                    "Chưa đủ căn cứ trong dược thư để trả lời an toàn. "
                    "Vui lòng đối chiếu thủ công hoặc liên hệ Hội đồng KH."
                ),
                refused=True,
                trace=ChatTrace(
                    classification=cls,
                    rag_used=True,
                    chunks_retrieved=len(retrieved),
                    chunks_after_rerank=0,
                    retries_used=retries_used,
                    rewritten_query=rewritten_query,
                ),
            )

        if (
            crag_result is not None
            and crag_result.should_refuse
            and is_clinical
            and self._enforce_citations
        ):
            return ChatResult(
                content=(
                    "Trích đoạn tìm được chưa đủ bao phủ câu hỏi lâm sàng "
                    "(CRAG: insufficient). Vui lòng diễn đạt cụ thể hơn "
                    f"(thuốc, đối tượng, tình huống) hoặc tham vấn dược sĩ. {MEDICAL_DISCLAIMER}"
                ),
                refused=True,
                finish_reason="crag_insufficient",
                trace=ChatTrace(
                    classification=cls,
                    rag_used=True,
                    chunks_retrieved=len(retrieved),
                    chunks_after_rerank=len(reranked),
                    crag=crag_result,
                    retries_used=retries_used,
                    rewritten_query=rewritten_query,
                ),
            )

        prompted = build_chat_prompt(active_query, reranked)
        with self._tracer.start_span("generate", max_tokens=max_tokens, temperature=temperature) as sp:
            gen = await self._llm.generate(
                GenerationRequest(
                    prompt=prompted.prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
            sp.set_attribute("prompt_tokens", gen.prompt_tokens)
            sp.set_attribute("completion_tokens", gen.completion_tokens)
            sp.set_attribute("finish_reason", gen.finish_reason)

        with self._tracer.start_span("validate") as sp:
            validators = self._run_validators(gen.text, set(prompted.citation_ids))
            sp.set_attribute("failed", [v.name for v in validators if not v.passed])
        if any(not v.passed for v in validators) and self._enforce_citations:
            return ChatResult(
                content=(
                    f"Câu trả lời nháp không đạt kiểm tra an toàn ({_summarise(validators)}). "
                    "Vui lòng đặt lại câu hỏi cụ thể hơn hoặc tham vấn dược sĩ. "
                    f"{MEDICAL_DISCLAIMER}"
                ),
                refused=True,
                finish_reason="safety_block",
                prompt_tokens=gen.prompt_tokens,
                completion_tokens=gen.completion_tokens,
                trace=ChatTrace(
                    classification=cls,
                    rag_used=bool(reranked),
                    chunks_retrieved=len(retrieved),
                    chunks_after_rerank=len(reranked),
                    validators=validators,
                    crag=crag_result,
                    retries_used=retries_used,
                    rewritten_query=rewritten_query,
                ),
            )

        citations = [f"[REF:{i}] {r.chunk.source}" for i, r in enumerate(reranked, start=1)]
        return ChatResult(
            content=f"{gen.text.strip()}\n\n{MEDICAL_DISCLAIMER}",
            citations=citations,
            trace=ChatTrace(
                classification=cls,
                rag_used=bool(reranked),
                chunks_retrieved=len(retrieved),
                chunks_after_rerank=len(reranked),
                validators=validators,
                crag=crag_result,
                retries_used=retries_used,
                rewritten_query=rewritten_query,
            ),
            finish_reason=gen.finish_reason,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
        )

    async def _retrieve_and_rerank(
        self, query: str, retrieve_pool: int, rerank_keep: int
    ) -> tuple[list[RetrievedChunk], list[RerankedChunk]]:
        with self._tracer.start_span("retrieve", top_k=retrieve_pool) as sp:
            retrieved = await self._retriever.retrieve(query, top_k=retrieve_pool)
            sp.set_attribute("hits", len(retrieved))
        with self._tracer.start_span("rerank", keep=rerank_keep) as sp:
            reranked = await self._reranker.arerank(query, retrieved, top_k=rerank_keep)
            sp.set_attribute("kept", len(reranked))
            if reranked:
                sp.set_attribute("top_score", reranked[0].rerank_score)
        return retrieved, reranked

    def _run_validators(self, text: str, available_refs: set[int]) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        cit, _ = validate_citations(text, available_refs)
        results.append(cit)
        if self._known_drugs:
            results.append(validate_drug_names(text, self._known_drugs))
        if self._dosage_ranges:
            results.append(validate_dosage_sanity(text, self._dosage_ranges))
        results.append(validate_tone(text))
        return results


def _summarise(validators: list[ValidationResult]) -> str:
    failed = [v.name for v in validators if not v.passed]
    return ", ".join(failed) or "n/a"


def _label_rank(label: str) -> int:
    """Higher = better. Used to decide whether to adopt a rewrite's CRAG outcome."""
    return {"insufficient": 0, "ambiguous": 1, "sufficient": 2}.get(label, 0)
