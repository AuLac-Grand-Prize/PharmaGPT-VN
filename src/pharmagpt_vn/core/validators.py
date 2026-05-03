"""Output validators for clinical responses (Plan §3.5.3).

Run after generation, before returning to user. Each validator returns a
ValidationResult; orchestrator decides regenerate / disclaimer / reject.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

CITATION_PATTERN = re.compile(r"\[REF:(\d+)\]")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class ValidationResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class CitationReport:
    coverage: float
    invalid_refs: tuple[int, ...] = field(default_factory=tuple)
    sentence_count: int = 0
    cited_sentences: int = 0


def extract_citation_ids(text: str) -> set[int]:
    return {int(m.group(1)) for m in CITATION_PATTERN.finditer(text)}


def validate_citations(
    text: str,
    available_refs: Iterable[int],
    *,
    min_coverage: float = 0.85,
) -> tuple[ValidationResult, CitationReport]:
    """Verify every [REF:n] points to a retrieved chunk and coverage ≥ threshold."""
    available = set(available_refs)
    used = extract_citation_ids(text)
    invalid = tuple(sorted(used - available))

    sentences = [s for s in SENTENCE_SPLIT.split(text.strip()) if s]
    cited = sum(1 for s in sentences if CITATION_PATTERN.search(s))
    coverage = cited / len(sentences) if sentences else 0.0

    report = CitationReport(
        coverage=coverage,
        invalid_refs=invalid,
        sentence_count=len(sentences),
        cited_sentences=cited,
    )
    if invalid:
        return (
            ValidationResult("citation", False, f"invalid refs: {invalid}"),
            report,
        )
    if coverage < min_coverage:
        return (
            ValidationResult(
                "citation", False, f"coverage {coverage:.2f} < min {min_coverage:.2f}"
            ),
            report,
        )
    return ValidationResult("citation", True), report


def validate_drug_names(text: str, known_drugs: Iterable[str]) -> ValidationResult:
    """Reject any drug-like token not in VietDrug KG (Plan §3.5.3)."""
    known = {d.lower() for d in known_drugs}
    stripped = CITATION_PATTERN.sub("", text)
    candidates = {
        w.strip(".,;:()[]").lower()
        for w in re.findall(r"\b[A-Z][a-zA-Z0-9-]{2,}\b", stripped)
    }
    unknown = candidates - known
    if unknown:
        return ValidationResult(
            "drug_name",
            False,
            f"potential hallucinated drug(s): {sorted(unknown)}",
        )
    return ValidationResult("drug_name", True)


@dataclass(frozen=True)
class DosageRange:
    drug: str
    min_mg_per_day: float
    max_mg_per_day: float


_DOSE_PATTERN = re.compile(r"(\d+(?:[.,]\d+)?)\s*mg", re.IGNORECASE)


def validate_dosage_sanity(text: str, ranges: Iterable[DosageRange]) -> ValidationResult:
    lower = text.lower()
    for r in ranges:
        if r.drug.lower() not in lower:
            continue
        for match in _DOSE_PATTERN.finditer(text):
            value = float(match.group(1).replace(",", "."))
            if value < r.min_mg_per_day / 10 or value > r.max_mg_per_day * 2:
                return ValidationResult(
                    "dosage_sanity",
                    False,
                    f"{r.drug} dose {value}mg outside [{r.min_mg_per_day}, {r.max_mg_per_day}]",
                )
    return ValidationResult("dosage_sanity", True)


_TONE_PATTERNS = (
    re.compile(r"\bbạn nên tự (mua|uống|dùng)\b", re.IGNORECASE),
    re.compile(r"\bkhông cần (gặp|hỏi) (bác sĩ|dược sĩ)\b", re.IGNORECASE),
)


def validate_tone(text: str) -> ValidationResult:
    for pat in _TONE_PATTERNS:
        if pat.search(text):
            return ValidationResult("tone", False, f"unsafe self-medication phrasing: {pat.pattern}")
    return ValidationResult("tone", True)
