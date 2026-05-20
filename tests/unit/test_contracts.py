"""Contract round-trip tests — fixture must match the mirror in PrescriptionVision."""

import json
from pathlib import Path

import pytest

from pharmagpt_vn.contracts import (
    CONTRACT_VERSION,
    DisambiguationRequest,
    DisambiguationResponse,
    RankedCandidate,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "disambiguation_example.json"


def test_fixture_parses() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    req = DisambiguationRequest.model_validate(payload)
    assert req.task == "drug_disambiguation"
    assert req.contract_version == CONTRACT_VERSION
    assert len(req.candidates) == 3
    assert req.context.patient.age == 58
    assert req.return_top_k == 1


def test_request_rejects_extra_fields() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["surprise_field"] = "should fail"
    with pytest.raises(ValueError):
        DisambiguationRequest.model_validate(payload)


def test_request_clamps_confidence() -> None:
    bad = {
        "candidates": [{"name": "X", "confidence": 1.5}],
    }
    with pytest.raises(ValueError):
        DisambiguationRequest.model_validate(bad)


def test_response_round_trip() -> None:
    resp = DisambiguationResponse(
        top_candidates=[
            RankedCandidate(
                name="Metformin",
                strength="500mg",
                confidence=0.96,
                reasoning="match T2DM",
                citations=["REF:dược_điển_VN_metformin_2017"],
            )
        ],
        latency_ms=1240,
    )
    dumped = resp.model_dump()
    restored = DisambiguationResponse.model_validate(dumped)
    assert restored == resp
