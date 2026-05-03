"""Drug-disambiguation contract between PrescriptionVision Stage 4 → PharmaGPT-VN.

Reference: PharmLink AI Implementation Plan v1.0 §4.1.
This file is the canonical schema; the mirrored copy in `prescriptionvision.contracts`
MUST stay byte-identical at the field level. A round-trip test in both repos asserts
that a shared JSON fixture parses to the same shape on both sides.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_VERSION = "1.0.0"


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Drug name (brand or INN)")
    strength: str | None = Field(default=None, description="e.g. '500mg'")
    confidence: float = Field(..., ge=0.0, le=1.0)


class PatientContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int | None = Field(default=None, ge=0, le=130)
    sex: Literal["M", "F", "O"] | None = None
    comorbidities: list[str] = Field(default_factory=list)


class PrescriptionContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    diagnosis: str | None = None
    patient: PatientContext = Field(default_factory=PatientContext)
    other_drugs_in_prescription: list[str] = Field(default_factory=list)


class DisambiguationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Literal["drug_disambiguation"] = "drug_disambiguation"
    candidates: list[Candidate] = Field(..., min_length=1, max_length=10)
    context: PrescriptionContext = Field(default_factory=PrescriptionContext)
    return_top_k: int = Field(default=1, ge=1, le=5)
    contract_version: str = CONTRACT_VERSION


class RankedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    strength: str | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    citations: list[str] = Field(default_factory=list)


class DisambiguationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_candidates: list[RankedCandidate] = Field(..., min_length=1)
    latency_ms: int = Field(..., ge=0)
    contract_version: str = CONTRACT_VERSION
