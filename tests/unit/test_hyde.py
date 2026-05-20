"""HyDEGenerator — happy path, prompt injection, LLM failure fallback."""

from __future__ import annotations

import pytest

from pharmagpt_vn.models.llm_client import GenerationRequest, GenerationResult
from pharmagpt_vn.rag.hyde import HyDEGenerator


class _StubLLM:
    def __init__(self, text: str = "fake answer") -> None:
        self.last_request: GenerationRequest | None = None
        self._text = text

    async def generate(self, req: GenerationRequest) -> GenerationResult:
        self.last_request = req
        return GenerationResult(text=self._text)


class _FailingLLM:
    async def generate(self, req: GenerationRequest) -> GenerationResult:
        raise RuntimeError("LLM offline")


@pytest.mark.asyncio
async def test_generate_returns_llm_text_stripped() -> None:
    llm = _StubLLM(text="  Metformin chống chỉ định khi eGFR < 30.  \n")
    hyde = HyDEGenerator(llm=llm)
    out = await hyde.generate("metformin suy thận?")
    assert out == "Metformin chống chỉ định khi eGFR < 30."


@pytest.mark.asyncio
async def test_prompt_contains_query() -> None:
    llm = _StubLLM(text="x")
    hyde = HyDEGenerator(llm=llm)
    await hyde.generate("liều paracetamol cho trẻ em?")
    assert llm.last_request is not None
    assert "liều paracetamol cho trẻ em?" in llm.last_request.prompt
    # Sanity: prompt template asks for an answer-shaped paragraph.
    assert "Dược thư" in llm.last_request.prompt


@pytest.mark.asyncio
async def test_failing_llm_returns_empty_string() -> None:
    hyde = HyDEGenerator(llm=_FailingLLM())
    out = await hyde.generate("any query")
    assert out == ""


@pytest.mark.asyncio
async def test_custom_max_tokens_and_temperature_propagate() -> None:
    llm = _StubLLM(text="x")
    hyde = HyDEGenerator(llm=llm, max_tokens=128, temperature=0.5)
    await hyde.generate("q")
    assert llm.last_request is not None
    assert llm.last_request.max_tokens == 128
    assert llm.last_request.temperature == 0.5
