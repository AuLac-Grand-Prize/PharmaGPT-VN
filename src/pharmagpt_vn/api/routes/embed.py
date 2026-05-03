from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class EmbedRequest(BaseModel):
    inputs: list[str]
    model: str = "bge-m3"


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dim: int


@router.post("/embeddings", response_model=EmbedResponse)
async def embeddings(req: EmbedRequest) -> EmbedResponse:
    # TODO: load bge-m3 once, encode batch
    return EmbedResponse(embeddings=[[] for _ in req.inputs], model=req.model, dim=0)
