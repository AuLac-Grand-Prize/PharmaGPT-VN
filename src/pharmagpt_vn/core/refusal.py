"""Refusal classifier interface (Plan §3.5.2).

Production: distill ~100M classifier from Qwen2.5-1.5B over labelled queries.
This module defines the contract; concrete classifier loads in models/ later.
"""

from dataclasses import dataclass
from typing import Literal, Protocol

QueryClass = Literal[
    "clinical_safe",      # standard drug info / dosage / interaction
    "clinical_high_risk", # pediatric, pregnancy, narrow therapeutic index
    "out_of_scope",       # non-pharma question
    "unsafe",             # self-harm, dosage to overdose, illegal use
    "ambiguous",          # cannot classify — ask clarifying question
]


@dataclass(frozen=True)
class Classification:
    label: QueryClass
    confidence: float
    rationale: str = ""


class RefusalClassifier(Protocol):
    def classify(self, query: str) -> Classification: ...


REFUSAL_TEMPLATES: dict[QueryClass, str] = {
    "out_of_scope": (
        "Em chỉ trả lời câu hỏi liên quan đến dược lâm sàng. "
        "Vui lòng đặt lại câu hỏi về thuốc, liều dùng, hoặc tương tác."
    ),
    "unsafe": (
        "Em không thể hỗ trợ yêu cầu này vì có nguy cơ an toàn. "
        "Vui lòng liên hệ trực tiếp dược sĩ hoặc bác sĩ."
    ),
    "ambiguous": (
        "Em chưa rõ câu hỏi. Anh/chị có thể nêu rõ tên thuốc, "
        "tình trạng bệnh nhân và mục đích cần tư vấn không?"
    ),
}


def should_refuse(c: Classification) -> bool:
    return c.label in {"out_of_scope", "unsafe"} or (
        c.label == "ambiguous" and c.confidence > 0.7
    )
