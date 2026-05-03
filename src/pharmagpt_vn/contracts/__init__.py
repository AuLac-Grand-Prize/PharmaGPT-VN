"""Cross-engine contracts (locked from M2 — see Plan §4.1)."""

from pharmagpt_vn.contracts.disambiguation import (
    CONTRACT_VERSION,
    Candidate,
    DisambiguationRequest,
    DisambiguationResponse,
    PatientContext,
    PrescriptionContext,
    RankedCandidate,
)

__all__ = [
    "CONTRACT_VERSION",
    "Candidate",
    "DisambiguationRequest",
    "DisambiguationResponse",
    "PatientContext",
    "PrescriptionContext",
    "RankedCandidate",
]
