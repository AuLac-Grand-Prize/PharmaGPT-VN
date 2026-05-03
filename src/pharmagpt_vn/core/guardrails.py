"""Medical / clinical guardrails — PII redaction, scope check, citation enforcement."""

import re

CCCD_PATTERN = re.compile(r"\b\d{12}\b")
PHONE_PATTERN = re.compile(r"\b(0|\+84)\d{9,10}\b")
BHYT_PATTERN = re.compile(r"\b[A-Z]{2}\d{13}\b")

CLINICAL_KEYWORDS = (
    "thuốc",
    "liều",
    "tương tác",
    "chống chỉ định",
    "tác dụng phụ",
    "metformin",
    "insulin",
)


def redact_pii(text: str) -> str:
    text = CCCD_PATTERN.sub("[CCCD]", text)
    text = PHONE_PATTERN.sub("[PHONE]", text)
    text = BHYT_PATTERN.sub("[BHYT]", text)
    return text


def is_clinical_query(text: str) -> bool:
    lower = text.lower()
    return any(k in lower for k in CLINICAL_KEYWORDS)


MEDICAL_DISCLAIMER = (
    "Câu trả lời này do AI tổng hợp để tham khảo. "
    "Vui lòng xác nhận với dược sĩ hoặc bác sĩ trước khi quyết định sử dụng thuốc."
)
