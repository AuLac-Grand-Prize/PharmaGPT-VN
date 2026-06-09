"""Refusal classifier interface + a deterministic heuristic (Plan §3.5.2).

Production target: distill a ~100M classifier from Qwen2.5-1.5B over labelled
queries (future ML work). Until that ships, `HeuristicRefusalClassifier` below
gives the engine real, offline, dependency-free triage so obvious out-of-scope
and unsafe queries are actually refused at the classifier stage — replacing the
previous always-`clinical_safe` placeholder that failed open silently.
"""

from __future__ import annotations

import re
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


# --- Heuristic classifier (offline, deterministic) -------------------------
#
# Rule order matters: unsafe is checked first (safety dominates), then clinical
# signals promote a query to clinical_safe/high_risk, then non-pharma topic
# markers flag out_of_scope. A clinical drug query never trips the out_of_scope
# rules because clinical detection runs before them.


def _compile_words(words: list[str]) -> re.Pattern[str]:
    r"""Build a case-insensitive, accent-aware alternation.

    Vietnamese terms carry diacritics, so we don't use ``\b`` word boundaries
    (they behave oddly around combining marks); a plain substring alternation is
    what we want for keyword triage.
    """
    return re.compile("|".join(re.escape(w) for w in words), re.IGNORECASE)


# Self-harm / overdose-intent / illegal-use markers (VN + EN). Refused as unsafe.
_UNSAFE_TERMS = [
    "tự tử",
    "tự sát",
    "tự làm hại",
    "tự gây thương tích",
    "kết liễu",
    "để chết",
    "đến chết",
    "cho chết",
    "liều gây chết",
    "liều gây tử vong",
    "liều chết người",
    "quá liều để",
    "để quá liều",
    "đầu độc",
    "gây mê người khác",
    "điều chế ma túy",
    "chế ma túy",
    "nấu ma túy",
    "tổng hợp ma túy",
    "ma túy đá",
    "chất cấm",
    "doping",
    "suicide",
    "kill myself",
    "kill him",
    "kill her",
    "overdose to die",
    "lethal dose",
    "how to make meth",
    "make drugs",
    "poison someone",
]

# Clinical / pharma signal terms (VN + EN). Promote to clinical_safe.
_CLINICAL_TERMS = [
    "thuốc",
    "liều",
    "liều dùng",
    "tương tác",
    "chống chỉ định",
    "chỉ định",
    "tác dụng phụ",
    "phản ứng phụ",
    "kháng sinh",
    "kê đơn",
    "đơn thuốc",
    "dược",
    "biệt dược",
    "hoạt chất",
    "hàm lượng",
    "viên nén",
    "tiêm",
    "uống",
    "metformin",
    "insulin",
    "paracetamol",
    "amoxicillin",
    "ibuprofen",
    "aspirin",
    "warfarin",
    "dose",
    "dosage",
    "drug",
    "medication",
    "interaction",
    "contraindication",
    "side effect",
    "antibiotic",
    "prescription",
]

# High-risk clinical sub-population / window markers. Promote to clinical_high_risk.
_HIGH_RISK_TERMS = [
    "trẻ sơ sinh",
    "trẻ em",
    "trẻ nhỏ",
    "nhi khoa",
    "phụ nữ có thai",
    "mang thai",
    "có thai",
    "thai kỳ",
    "cho con bú",
    "suy gan",
    "suy thận",
    "khoảng điều trị hẹp",
    "pregnant",
    "pregnancy",
    "breastfeeding",
    "neonate",
    "infant",
    "pediatric",
    "renal failure",
    "hepatic failure",
    "narrow therapeutic",
]

# Obvious non-pharma topics. Flag out_of_scope when no clinical signal present.
_OUT_OF_SCOPE_TERMS = [
    "thời tiết",
    "bóng đá",
    "thể thao",
    "tỷ số",
    "chứng khoán",
    "giá vàng",
    "bitcoin",
    "tiền điện tử",
    "chính trị",
    "bầu cử",
    "nấu ăn",
    "công thức nấu",
    "du lịch",
    "khách sạn",
    "vé máy bay",
    "phim",
    "bài hát",
    "ca sĩ",
    "tình yêu",
    "người yêu",
    "lập trình",
    "viết code",
    "toán học",
    "weather",
    "football",
    "soccer",
    "stock price",
    "politics",
    "election",
    "recipe",
    "cooking",
    "travel",
    "movie",
    "song",
    "programming",
    "write code",
]

_UNSAFE_RE = _compile_words(_UNSAFE_TERMS)
_CLINICAL_RE = _compile_words(_CLINICAL_TERMS)
_HIGH_RISK_RE = _compile_words(_HIGH_RISK_TERMS)
_OUT_OF_SCOPE_RE = _compile_words(_OUT_OF_SCOPE_TERMS)


class HeuristicRefusalClassifier:
    """Deterministic, dependency-free triage over `QueryClass`.

    Not a substitute for the future distilled model — it's a transparent
    keyword/regex gate that (a) refuses obvious self-harm/illegal queries,
    (b) keeps clinical drug questions flowing as ``clinical_safe`` (or
    ``clinical_high_risk`` for sensitive populations), and (c) flags plainly
    non-pharma questions as ``out_of_scope`` so the engine stops failing open.

    Empty / whitespace-only input is treated as ``ambiguous`` (ask to clarify)
    rather than silently passed through.
    """

    def classify(self, query: str) -> Classification:
        text = (query or "").strip()
        if not text:
            return Classification(
                label="ambiguous",
                confidence=0.9,
                rationale="empty query",
            )

        if _UNSAFE_RE.search(text):
            return Classification(
                label="unsafe",
                confidence=0.95,
                rationale="matched self-harm / illegal-use keyword",
            )

        if _CLINICAL_RE.search(text):
            if _HIGH_RISK_RE.search(text):
                return Classification(
                    label="clinical_high_risk",
                    confidence=0.8,
                    rationale="clinical query touching a high-risk population/window",
                )
            return Classification(
                label="clinical_safe",
                confidence=0.7,
                rationale="matched clinical/pharma keyword",
            )

        if _OUT_OF_SCOPE_RE.search(text):
            return Classification(
                label="out_of_scope",
                confidence=0.9,
                rationale="matched non-pharma topic with no clinical signal",
            )

        # No clinical signal and no recognised topic: this is a pharma-only
        # assistant, so treat an unrecognised question as out of scope rather
        # than guessing it is clinical. confidence kept moderate.
        return Classification(
            label="out_of_scope",
            confidence=0.6,
            rationale="no clinical signal detected",
        )
