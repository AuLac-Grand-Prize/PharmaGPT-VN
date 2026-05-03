"""Run VN-PharmBench against a deployed PharmaGPT-VN endpoint.

Example:
    python scripts/eval_bench.py \
        --bench data/vn_pharmbench/v1.jsonl \
        --judge substring \
        --out reports/vn_pharmbench_v1.json
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from pharmagpt_vn.api.dependencies import _default_classifier  # type: ignore[attr-defined]
from pharmagpt_vn.core.config import get_settings
from pharmagpt_vn.evaluation import BenchmarkRunner, load_jsonl
from pharmagpt_vn.evaluation.benchmark import report_to_json
from pharmagpt_vn.evaluation.judges import SubstringJudge
from pharmagpt_vn.models.vllm_client import VLLMClient
from pharmagpt_vn.rag.reranker import CrossEncoderReranker
from pharmagpt_vn.rag.retriever import HybridRetriever
from pharmagpt_vn.services.chat_service import ChatService


def _build_chat() -> ChatService:
    s = get_settings()
    return ChatService(
        retriever=HybridRetriever(
            qdrant_url=s.qdrant_url,
            collection=s.qdrant_collection,
            embedding_model=s.embedding_model,
        ),
        reranker=CrossEncoderReranker(),
        refusal_classifier=_default_classifier(),
        llm=VLLMClient(base_url=f"http://localhost:{s.app_port}", model=s.base_model),
    )


async def _run(args: argparse.Namespace) -> int:
    examples = load_jsonl(args.bench)
    runner = BenchmarkRunner(_build_chat(), SubstringJudge(), concurrency=args.concurrency)
    report = await runner.run(examples)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_to_json(report), encoding="utf-8")
    print(report_to_json(report))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", required=True, help="Path to JSONL benchmark file")
    parser.add_argument("--out", default="reports/vn_pharmbench.json")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--judge", default="substring", choices=["substring"])
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
