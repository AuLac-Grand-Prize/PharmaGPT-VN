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
from pharmagpt_vn.core.validators import (
    DosageRange,
    ValidationResult,
    validate_citations,
    validate_dosage_sanity,
    validate_drug_names,
    validate_tone,
)
from pharmagpt_vn.models.llm_client import GenerationRequest, LLMClient
from pharmagpt_vn.rag.reranker import CrossEncoderReranker
from pharmagpt_vn.rag.retriever import HybridRetriever
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
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._classifier = refusal_classifier
        self._llm = llm
        self._known_drugs = list(known_drugs)
        self._dosage_ranges = list(dosage_ranges)
        self._enforce_citations = enforce_citations
        self._always_on_rag = always_on_rag

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
        cls = self._classifier.classify(last_user.content)
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
        retrieved, reranked = [], []
        if self._always_on_rag or is_clinical:
            retrieved = await self._retriever.retrieve(query, top_k=retrieve_pool)
            reranked = self._reranker.rerank(query, retrieved, top_k=rerank_keep)

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
                ),
            )

        prompted = build_chat_prompt(query, reranked)
        gen = await self._llm.generate(
            GenerationRequest(
                prompt=prompted.prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )

        validators = self._run_validators(gen.text, set(prompted.citation_ids))
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
            ),
            finish_reason=gen.finish_reason,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
        )

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
