"""BGE-M3 embedder — produces dense + sparse vectors in one pass (Plan §3.4.2).

The class wraps `FlagEmbedding.BGEM3FlagModel`. Heavy deps (torch, FlagEmbedding)
are imported inside `_load()` so unit-testing higher layers doesn't pay the cost.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingPair:
    dense: list[float]
    sparse: dict[int, float]  # token-id → weight


class BGEM3Embedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.device = device
        self._model = None

    def _load(self):  # type: ignore[no-untyped-def]
        if self._model is not None:
            return self._model
        from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-not-found]

        self._model = BGEM3FlagModel(
            self.model_name, use_fp16=self.use_fp16, devices=[self.device]
        )
        return self._model

    def encode(self, texts: Sequence[str]) -> list[EmbeddingPair]:
        model = self._load()
        out = model.encode(
            list(texts),
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = out["dense_vecs"]
        sparse = out["lexical_weights"]
        return [
            EmbeddingPair(
                dense=list(map(float, dense[i])),
                sparse={int(k): float(v) for k, v in sparse[i].items()},
            )
            for i in range(len(texts))
        ]

    def encode_query(self, text: str) -> EmbeddingPair:
        return self.encode([text])[0]
