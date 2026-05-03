"""Cross-encoder re-ranker (Plan §3.4.4 Stage C).

Wraps BGE-reranker-v2-m3 — runs only on a small candidate set so the cost is
acceptable. Production loads the model once at startup; this module exposes the
interface and a deterministic stub for tests.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from pharmagpt_vn.rag.retriever import RetrievedChunk


@dataclass(frozen=True)
class RerankedChunk:
    chunk: RetrievedChunk
    rerank_score: float


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.device = device
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is not None:
            return self._model
        from FlagEmbedding import FlagReranker  # type: ignore[import-not-found]

        self._model = FlagReranker(self.model_name, use_fp16=self.use_fp16, devices=[self.device])
        return self._model

    def rerank(
        self, query: str, candidates: Iterable[RetrievedChunk], top_k: int = 10
    ) -> list[RerankedChunk]:
        cand_list = list(candidates)
        if not cand_list:
            return []
        try:
            model = self._load()
            scores = model.compute_score(  # type: ignore[attr-defined]
                [(query, c.text) for c in cand_list], normalize=True
            )
        except Exception:
            # Fallback to retrieval order — keeps the pipeline runnable when the
            # reranker model is missing (e.g. local dev without GPU).
            scores = [c.score for c in cand_list]
        ranked = [RerankedChunk(chunk=c, rerank_score=float(s)) for c, s in zip(cand_list, scores)]
        ranked.sort(key=lambda x: x.rerank_score, reverse=True)
        return ranked[:top_k]
