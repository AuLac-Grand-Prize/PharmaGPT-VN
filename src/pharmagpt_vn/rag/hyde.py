"""Hypothetical Document Embeddings (HyDE) — Gao et al. 2022.

Generate an answer-shaped hypothetical document with the LLM, then use *that*
document (instead of the raw question) as the retrieval anchor. Intuition:
answer-shaped text sits closer to the right monograph in embedding space than
a short question does — boosts recall ~10-20% on clinical Q&A.

Cost: 1 LLM call per query. Gated upstream by `is_clinical_query` so non-
clinical chats don't pay.

Fallback: empty string when LLM fails — orchestrator skips Branch C and
falls back to Branches A + B.
"""

from __future__ import annotations

import logging

from pharmagpt_vn.models.llm_client import GenerationRequest, LLMClient

logger = logging.getLogger(__name__)


HYDE_PROMPT_TEMPLATE = (
    "Bạn là dược sĩ lâm sàng. Viết một đoạn 100-150 từ trả lời câu hỏi sau "
    "như thể trích từ Dược thư Quốc gia Việt Nam. Dùng thuật ngữ chuyên ngành, "
    "nêu rõ liều/đối tượng/chống chỉ định khi liên quan. "
    "KHÔNG ghi 'tôi không biết' hoặc 'cần tham vấn dược sĩ'.\n\n"
    "Câu hỏi: {query}\n\n"
    "Đoạn trích:"
)


class HyDEGenerator:
    def __init__(
        self,
        llm: LLMClient,
        *,
        prompt_template: str = HYDE_PROMPT_TEMPLATE,
        max_tokens: int = 256,
        temperature: float = 0.3,
    ) -> None:
        self._llm = llm
        self._prompt_template = prompt_template
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def generate(self, query: str) -> str:
        prompt = self._prompt_template.format(query=query)
        try:
            result = await self._llm.generate(
                GenerationRequest(
                    prompt=prompt,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
            )
        except Exception:
            logger.warning("HyDE generation failed; returning empty string", exc_info=True)
            return ""
        return (result.text or "").strip()
