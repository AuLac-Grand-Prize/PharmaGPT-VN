"""Stage 4 callee — drug-disambiguation grounded on VietDrug KG retrieval.

Flow per request:
  1. Retrieve drug monographs for each candidate name (one query per candidate).
  2. Render candidates + retrieved monographs + clinical context into a prompt.
  3. Ask LLM for ranked candidates with reasoning + [REF:n] citations.
  4. Parse JSON response; fall back to vision top-1 on parse failure.

The fallback is deliberate: a malformed LLM output should never break the
PrescriptionVision pipeline — caller catches `PharmaGPTUnavailable` upstream,
and here we mimic that contract by returning a confidence-preserving response
when reasoning is unavailable.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterable

from pharmagpt_vn.contracts.disambiguation import (
    Candidate,
    DisambiguationRequest,
    DisambiguationResponse,
    PrescriptionContext,
    RankedCandidate,
)
from pharmagpt_vn.models.llm_client import GenerationRequest, LLMClient
from pharmagpt_vn.rag.reranker import Reranker, RerankedChunk
from pharmagpt_vn.rag.retriever import HybridRetriever, RetrievedChunk

log = logging.getLogger(__name__)

DISAMBIGUATION_SYSTEM_PROMPT = (
    "Bạn là PharmaGPT-VN. Nhiệm vụ: chọn đúng thuốc trong danh sách candidate dựa "
    "trên chẩn đoán + tiền sử bệnh nhân + thuốc đang dùng. Trả lời bằng JSON đúng "
    "schema: {\"top_candidates\":[{\"name\":..., \"strength\":..., \"confidence\":0..1, "
    "\"reasoning\":\"...\", \"citations\":[\"REF:n\", ...]}], \"latency_ms\":<int>}. "
    "Mỗi reasoning PHẢI có [REF:n] trỏ tới chunk được cung cấp. KHÔNG được bịa."
)

JSON_BLOCK = re.compile(r"\{[\s\S]+\}")


class DisambiguationService:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Reranker | None = None,
        llm: LLMClient | None = None,
        chunks_per_candidate: int = 3,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._llm = llm
        self._chunks_per_candidate = chunks_per_candidate

    async def rank(self, req: DisambiguationRequest) -> DisambiguationResponse:
        started = time.perf_counter()
        retrieved = await self._collect_monographs(req.candidates, req.context)
        ranked: list[RankedCandidate]
        if self._llm is None:
            ranked = _fallback_rank(req.candidates, retrieved)
        else:
            ranked = await self._llm_rank(req, retrieved)
        ranked = ranked[: req.return_top_k] or _fallback_rank(req.candidates, retrieved)
        return DisambiguationResponse(
            top_candidates=ranked,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    async def _collect_monographs(
        self, candidates: list[Candidate], context: PrescriptionContext
    ) -> list[RerankedChunk]:
        diagnosis_hint = (context.diagnosis or "").strip()
        seen_keys: set[str] = set()
        merged: list[RerankedChunk] = []
        for cand in candidates:
            query = f"{cand.name} {diagnosis_hint}".strip()
            chunks = await self._retriever.retrieve(query, top_k=self._chunks_per_candidate)
            scored = (
                self._reranker.rerank(query, chunks, top_k=self._chunks_per_candidate)
                if self._reranker is not None
                else [RerankedChunk(chunk=c, rerank_score=c.score) for c in chunks]
            )
            for rc in scored:
                key = (rc.chunk.source, rc.chunk.text[:80])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                merged.append(rc)
        merged.sort(key=lambda r: r.rerank_score, reverse=True)
        return merged

    async def _llm_rank(
        self, req: DisambiguationRequest, retrieved: list[RerankedChunk]
    ) -> list[RankedCandidate]:
        prompt = _build_prompt(req, retrieved)
        try:
            gen = await self._llm.generate(  # type: ignore[union-attr]
                GenerationRequest(prompt=prompt, temperature=0.1, max_tokens=512)
            )
        except Exception as exc:  # pragma: no cover - network / runtime
            log.warning("LLM call failed in disambiguation: %s", exc)
            return _fallback_rank(req.candidates, retrieved)
        return _parse_ranked(gen.text, fallback=req.candidates, retrieved=retrieved)


def _build_prompt(req: DisambiguationRequest, retrieved: Iterable[RerankedChunk]) -> str:
    refs = []
    for idx, rc in enumerate(retrieved, start=1):
        refs.append(f"[REF:{idx}] {rc.chunk.source}\n{rc.chunk.text.strip()}")
    cand_lines = [
        f"- {c.name} {c.strength or ''} (vision_conf={c.confidence:.2f})".rstrip()
        for c in req.candidates
    ]
    patient = req.context.patient
    return (
        f"<|system|>\n{DISAMBIGUATION_SYSTEM_PROMPT}\n"
        f"<|context|>\n"
        f"Chẩn đoán: {req.context.diagnosis or 'N/A'}\n"
        f"Tuổi: {patient.age or 'N/A'} | Giới: {patient.sex or 'N/A'} | "
        f"Comorbidities: {', '.join(patient.comorbidities) or 'none'}\n"
        f"Thuốc khác trong đơn: {', '.join(req.context.other_drugs_in_prescription) or 'none'}\n"
        f"Candidates:\n" + "\n".join(cand_lines) + "\n"
        f"<|monographs|>\n" + ("\n\n".join(refs) if refs else "(không có)") + "\n"
        f"<|user|>\nReturn JSON only.\n<|assistant|>\n"
    )


def _parse_ranked(
    text: str,
    *,
    fallback: list[Candidate],
    retrieved: list[RerankedChunk],
) -> list[RankedCandidate]:
    match = JSON_BLOCK.search(text)
    if match is None:
        return _fallback_rank(fallback, retrieved)
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return _fallback_rank(fallback, retrieved)
    raw = payload.get("top_candidates")
    if not isinstance(raw, list) or not raw:
        return _fallback_rank(fallback, retrieved)
    out: list[RankedCandidate] = []
    for item in raw:
        try:
            out.append(RankedCandidate.model_validate(item))
        except Exception:  # noqa: BLE001 - tolerate per-item parse errors
            continue
    return out or _fallback_rank(fallback, retrieved)


def _fallback_rank(
    candidates: list[Candidate], retrieved: list[RerankedChunk]
) -> list[RankedCandidate]:
    sources = sorted({rc.chunk.source for rc in retrieved})
    citations = [f"REF:{i+1}" for i, _ in enumerate(retrieved[:3])] if sources else []
    return [
        RankedCandidate(
            name=c.name,
            strength=c.strength,
            confidence=c.confidence,
            reasoning=(
                "fallback: vision confidence pass-through (LLM unavailable or "
                "produced unparseable output)"
            ),
            citations=citations,
        )
        for c in sorted(candidates, key=lambda x: x.confidence, reverse=True)
    ]


# Re-export for convenience.
__all__ = ["DisambiguationService", "RetrievedChunk"]
