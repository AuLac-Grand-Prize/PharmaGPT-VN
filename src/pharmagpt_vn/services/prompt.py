"""Prompt builders enforcing Plan §3.4.5 citation policy.

Output discipline:
  - System turn forces "use [REF:n] inline; one citation per clinical claim".
  - Each retrieved chunk is rendered with a stable `id` that we expose to the
    validator (citations resolved by id, not by free-text source).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pharmagpt_vn.rag.reranker import RerankedChunk

SYSTEM_PROMPT_VI = (
    "Bạn là PharmaGPT-VN — trợ lý dược lâm sàng tiếng Việt.\n"
    "Quy tắc bắt buộc:\n"
    "1. Mọi khẳng định lâm sàng (liều, chỉ định, chống chỉ định, tương tác) PHẢI có "
    "trích dẫn dạng [REF:n] ngay sau câu, n là id của chunk được cung cấp.\n"
    "2. Không tự bịa số liệu hoặc tên thuốc ngoài danh sách chunk. Nếu chunk không đủ "
    "thông tin, hãy trả lời: \"Chưa đủ căn cứ; vui lòng đối chiếu với dược thư.\"\n"
    "3. Không khuyên bệnh nhân tự dùng thuốc kê đơn; luôn hướng dẫn xác nhận với "
    "dược sĩ/bác sĩ.\n"
)


@dataclass(frozen=True)
class PromptedQuery:
    prompt: str
    citation_ids: tuple[int, ...]


def render_chunks(chunks: Sequence[RerankedChunk]) -> tuple[str, tuple[int, ...]]:
    rendered = []
    ids: list[int] = []
    for idx, rc in enumerate(chunks, start=1):
        ids.append(idx)
        path = " / ".join(rc.chunk.metadata.get("parent_path", ()))
        header = f"[REF:{idx}] {rc.chunk.source}{(' — ' + path) if path else ''}"
        rendered.append(f"{header}\n{rc.chunk.text.strip()}")
    return "\n\n".join(rendered), tuple(ids)


def build_chat_prompt(
    user_query: str,
    chunks: Sequence[RerankedChunk],
    system_prompt: str = SYSTEM_PROMPT_VI,
) -> PromptedQuery:
    context, ids = render_chunks(chunks)
    body = (
        f"<|system|>\n{system_prompt}\n"
        f"<|context|>\n{context if context else '(không có chunk)'}\n"
        f"<|user|>\n{user_query}\n"
        f"<|assistant|>\n"
    )
    return PromptedQuery(prompt=body, citation_ids=ids)
