"""Heuristic VN decomposer — break complex pharmacy queries into sub-queries.

Strategies (combined, deduped, capped):
  1. Drug-drug interaction: when ≥2 drugs detected → "{A} tương tác {B}" per pair.
  2. Drug × clinical context: pair each drug with detected condition
     (suy thận, suy gan, thai kỳ, trẻ em…) → "{drug} {context}".
  3. Conjunction split: last-resort split on "và/với/cùng/kèm/hoặc" when
     no drug/context detected.

Output contract: ≤max_sub sub-queries. Falls back to `[query]` when nothing
matches — caller always gets at least one query to retrieve on.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Protocol


class Decomposer(Protocol):
    def decompose(self, query: str) -> list[str]: ...


_CONJ_RE = re.compile(r"\s+(?:và|với|cùng|kèm|hoặc)\s+", flags=re.IGNORECASE)
# Capitalized token of ≥4 chars, supporting VN diacritics — covers most generic
# drug names like Metformin / Amlodipine / Paracetamol. Used as a fallback when
# no whitelist is provided.
_CAP_DRUG_RE = re.compile(
    r"\b[A-ZÀ-Ỹ][a-zA-Zà-ỹÀ-Ỹ]{3,}\b"
)

_CLINICAL_CONTEXTS: tuple[str, ...] = (
    "suy thận",
    "suy gan",
    "thai kỳ",
    "phụ nữ có thai",
    "phụ nữ mang thai",
    "trẻ em",
    "nhi khoa",
    "người cao tuổi",
    "người già",
    "đái tháo đường",
    "tiểu đường",
    "tăng huyết áp",
    "egfr thấp",
)


class HeuristicVNDecomposer:
    def __init__(
        self,
        max_sub: int = 4,
        known_drugs: Iterable[str] = (),
        contexts: Iterable[str] = _CLINICAL_CONTEXTS,
    ) -> None:
        self._max_sub = max_sub
        self._known = tuple(known_drugs)
        self._contexts = tuple(contexts)

    def decompose(self, query: str) -> list[str]:
        drugs = self._detect_drugs(query)
        contexts = self._detect_contexts(query)

        out: list[str] = []

        # 1. Drug-drug interactions
        if len(drugs) >= 2:
            for i in range(len(drugs)):
                for j in range(i + 1, len(drugs)):
                    out.append(f"{drugs[i]} tương tác {drugs[j]}")

        # 2. Drug × clinical context
        for drug in drugs:
            for ctx in contexts:
                out.append(f"{drug} {ctx}")

        # 3. Conjunction split (only when patterns 1&2 produced nothing)
        if not out:
            parts = [p.strip() for p in _CONJ_RE.split(query) if len(p.strip().split()) >= 2]
            if len(parts) >= 2:
                out = parts

        # Dedupe normalized
        seen: set[str] = set()
        unique: list[str] = []
        for s in out:
            key = " ".join(s.lower().split())
            if key in seen:
                continue
            seen.add(key)
            unique.append(s)

        return unique[: self._max_sub] if unique else [query]

    def _detect_drugs(self, query: str) -> list[str]:
        if self._known:
            found: list[str] = []
            ql = query.lower()
            for drug in self._known:
                key = drug.lower()
                idx = ql.find(key)
                if idx >= 0:
                    found.append(query[idx : idx + len(key)])
            # Preserve original order, drop dupes.
            seen: set[str] = set()
            ordered: list[str] = []
            for d in found:
                if d.lower() not in seen:
                    seen.add(d.lower())
                    ordered.append(d)
            return ordered
        # Fallback: capitalized tokens.
        seen_cap: set[str] = set()
        out: list[str] = []
        for m in _CAP_DRUG_RE.findall(query):
            if m.lower() in seen_cap:
                continue
            seen_cap.add(m.lower())
            out.append(m)
        return out

    def _detect_contexts(self, query: str) -> list[str]:
        ql = query.lower()
        return [c for c in self._contexts if c in ql]
