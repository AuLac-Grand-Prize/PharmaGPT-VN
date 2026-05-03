"""POST /v1/disambiguate — Stage 4 cross-engine endpoint (Plan §4.1)."""

from fastapi import APIRouter, Depends

from pharmagpt_vn.api.dependencies import get_disambiguation_service
from pharmagpt_vn.contracts.disambiguation import DisambiguationRequest, DisambiguationResponse
from pharmagpt_vn.services.disambiguation_service import DisambiguationService

router = APIRouter()


@router.post("/disambiguate", response_model=DisambiguationResponse)
async def disambiguate(
    req: DisambiguationRequest,
    service: DisambiguationService = Depends(get_disambiguation_service),
) -> DisambiguationResponse:
    return await service.rank(req)
