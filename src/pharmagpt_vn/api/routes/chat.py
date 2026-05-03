"""POST /v1/chat/completions — OpenAI-compatible interface backed by ChatService."""

from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from pharmagpt_vn.api.dependencies import get_chat_service
from pharmagpt_vn.services.chat_service import ChatMessage as DomainMessage
from pharmagpt_vn.services.chat_service import ChatService

router = APIRouter()


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class RAGOptions(BaseModel):
    enabled: bool = True
    top_k: int = 5


class ChatRequest(BaseModel):
    model: str = "pharmagpt-vn-8b-instruct"
    messages: list[ChatMessage]
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    rag: RAGOptions = RAGOptions()


class Citation(BaseModel):
    id: int
    source: str


class ChatResponseChoice(BaseModel):
    message: ChatMessage
    citations: list[Citation] = []
    finish_reason: str


class ChatResponse(BaseModel):
    id: str
    model: str
    choices: list[ChatResponseChoice]
    usage: dict
    guardrails: dict


@router.post("/chat/completions", response_model=ChatResponse)
async def chat_completions(
    req: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    domain_messages = [DomainMessage(role=m.role, content=m.content) for m in req.messages]
    result = await service.complete(
        domain_messages,
        rag_top_k=req.rag.top_k if req.rag.enabled else 0,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )
    citations = [
        Citation(id=i + 1, source=src.split(" ", 1)[1] if " " in src else src)
        for i, src in enumerate(result.citations)
    ]
    return ChatResponse(
        id="chatcmpl-stub",
        model=req.model,
        choices=[
            ChatResponseChoice(
                message=ChatMessage(role="assistant", content=result.content),
                citations=citations,
                finish_reason=result.finish_reason,
            )
        ],
        usage={
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        },
        guardrails={
            "refused": result.refused,
            "validators": [
                {"name": v.name, "passed": v.passed, "detail": v.detail}
                for v in (result.trace.validators if result.trace else [])
            ],
            "rag_used": bool(result.trace and result.trace.rag_used),
        },
    )
