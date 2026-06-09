# Phase: PharmaGPT-VN Embeddings Endpoint + Remaining Stubs — Specification

**Created:** 2026-06-09
**Ambiguity score:** 0.08 (gate: ≤ 0.20)
**Requirements:** 6 locked

## Goal
Implement the real `POST /v1/embeddings` endpoint backed by the existing `BGEM3Embedder` so consumers (PrescriptionVision, the Gateway, ingest) receive correctly-dimensioned BGE-M3 dense vectors, and close the remaining hard-blocking stubs (missing `pharmagpt_vn.models` LLM-client package, placeholder refusal classifier, unwired hybrid retriever) — all verifiable by offline tests with fakes and no model download, Qdrant, or network.

## Background
This repo (PharmaGPT-VN, FastAPI, port 8003) is a mature Vietnamese pharma RAG engine: hybrid BGE-M3 retrieval, 3-branch query understanding, Corrective RAG, citation enforcement, PII redaction, ~21 test files. The mature paths are out of scope. The genuinely unfinished parts, verified against the code on `main`, are:

- **Embeddings stub.** `src/pharmagpt_vn/api/routes/embed.py` declares `EmbedRequest{inputs, model}` / `EmbedResponse{embeddings, model, dim}` but the handler is a TODO: `return EmbedResponse(embeddings=[[] for _ in req.inputs], model=req.model, dim=0)`. It never calls the embedder.
- **A real embedder already exists.** `src/pharmagpt_vn/rag/embeddings.py` defines `BGEM3Embedder` with `encode(texts) -> list[EmbeddingPair]` and `encode_query(text) -> EmbeddingPair`, where `EmbeddingPair{dense: list[float], sparse: dict[int, float]}`. Heavy deps (`FlagEmbedding`, torch) are imported only inside `_load()`, so callers that inject a fake embedder pay no model cost. BGE-M3 dense dimension is **1024** (documented in `src/pharmagpt_vn/rag/qdrant_store.py` line 8: "dense : named vector \"dense\" (cosine, BGE-M3 = 1024 dim)").
- **The `pharmagpt_vn.models` package does not exist.** `src/pharmagpt_vn/api/dependencies.py` imports `from pharmagpt_vn.models.llm_client import LLMClient` and `from pharmagpt_vn.models.openai_client import OpenAIClient`; `services/chat_service.py`, `services/disambiguation_service.py`, and `rag/hyde.py` import `GenerationRequest`/`GenerationResult`/`LLMClient` from `pharmagpt_vn.models.llm_client`; four test files (`test_chat_service.py`, `test_disambiguation_service.py`, `test_hyde.py`, `test_tracing.py`) import the same. No `src/pharmagpt_vn/models/` directory is present on disk or tracked in git. Consequently `from pharmagpt_vn.api.main import app` and the whole test suite fail to import (`ModuleNotFoundError: No module named 'pharmagpt_vn.models'`). The required surface is fully constrained by existing call sites: `GenerationRequest(prompt, temperature, max_tokens)`, `GenerationResult(text, prompt_tokens, completion_tokens, finish_reason)`, `LLMClient` Protocol with `async generate(GenerationRequest) -> GenerationResult`, and `OpenAIClient(base_url, api_key, model)` implementing it.
- **Refusal classifier is a placeholder.** `core/refusal.py` defines only the `RefusalClassifier` Protocol, `Classification`, `QueryClass`, `REFUSAL_TEMPLATES`, and `should_refuse()`. The wired implementation in `dependencies.py` (`_DefaultRefusalClassifier`) always returns `Classification(label="clinical_safe", confidence=0.5)` regardless of input — so `out_of_scope`/`unsafe` queries are never refused at the classifier stage.
- **Hybrid retriever is unwired in production.** `rag/retriever.py` `HybridRetriever.retrieve()` returns `[]` when `embedder`/`backend` are `None` ("placeholder until production wires both"), and `get_retriever()` in `dependencies.py` constructs `HybridRetriever(qdrant_url=..., collection=..., embedding_model=...)` without passing `embedder` or `backend`. The deployed retriever therefore always returns zero chunks.

Settings already expose `embedding_model` and `embedder_device` (`core/config.py`, `.env.example`: `EMBEDDING_MODEL=BAAI/bge-m3`, `EMBEDDER_DEVICE=cpu`). Test config: `pytest` with `asyncio_mode=auto`, coverage gate `--cov-fail-under=80`. No `conftest.py` exists yet; existing tests inject fakes via constructors.

## Requirements

1. **Embeddings endpoint returns real dense vectors**: `POST /v1/embeddings` encodes every input string through an injected embedder and returns one dense vector per input with a correct, non-zero `dim`.
   - Current: handler returns `embeddings=[[] for _ in req.inputs]` and `dim=0`; embedder never invoked (`api/routes/embed.py:18-21`).
   - Target: handler obtains an embedder via a FastAPI dependency, calls `embedder.encode(req.inputs)`, and returns `EmbedResponse(embeddings=[pair.dense for pair in pairs], model=req.model, dim=len(embeddings[0]) if embeddings else 0)`. Output order matches input order; `len(response.embeddings) == len(req.inputs)`; every vector has identical length equal to `dim`.
   - Acceptance: with a fake embedder injected via `app.dependency_overrides` producing 1024-length dense vectors, `POST /v1/embeddings {"inputs": ["paracetamol 500mg", "metformin"]}` returns HTTP 200, `len(embeddings) == 2`, `dim == 1024`, `all(len(v) == 1024 for v in embeddings)`, and `model == "bge-m3"`. FAIL if any returned vector is empty, `dim == 0`, lengths differ, or order is not preserved.

2. **Embeddings endpoint handles empty input deterministically**: an empty `inputs` list yields an empty result without raising.
   - Current: `[[] for _ in []]` returns `embeddings=[]` but `dim=0` with no embedder call (incidental, not designed).
   - Target: `req.inputs == []` returns HTTP 200 with `embeddings == []` and `dim == 0`, and the embedder is not required to be called.
   - Acceptance: `POST /v1/embeddings {"inputs": []}` returns 200, `embeddings == []`, `dim == 0`. FAIL on any 4xx/5xx or exception.

3. **Embedder dependency factory exists and is offline-overridable**: a DI provider supplies a `BGEM3Embedder` in production but is trivially replaced by a fake in tests.
   - Current: no embedder factory in `api/dependencies.py`; `BGEM3Embedder` is never constructed for the API layer.
   - Target: add `get_embedder(settings) -> BGEM3Embedder` (or an equivalent Protocol-typed provider) to `api/dependencies.py` that constructs `BGEM3Embedder(model_name=settings.embedding_model, device=settings.embedder_device)`. The embeddings route depends on it via `Depends`. The provider must not trigger model loading at construction time (loading stays inside `BGEM3Embedder._load()`), so importing the app downloads nothing.
   - Acceptance: a test overrides the provider through `app.dependency_overrides[get_embedder]` with a fake and observes the fake's vectors in the response; importing `pharmagpt_vn.api.main` performs no network or model download (no `FlagEmbedding` import at module import time). FAIL if the route hard-codes a concrete embedder that cannot be overridden, or if importing the app imports `FlagEmbedding`/torch.

4. **`pharmagpt_vn.models` LLM-client package is implemented to its existing contract**: the package imported across the codebase exists and exposes the exact symbols its call sites use.
   - Current: `src/pharmagpt_vn/models/` is absent; 8 modules import from `pharmagpt_vn.models.llm_client` / `pharmagpt_vn.models.openai_client`; `import pharmagpt_vn.models.llm_client` raises `ModuleNotFoundError`.
   - Target: create `src/pharmagpt_vn/models/__init__.py`, `llm_client.py`, and `openai_client.py`. `llm_client.py` exports `GenerationRequest(prompt: str, temperature: float = …, max_tokens: int = …)`, `GenerationResult(text: str, prompt_tokens: int = 0, completion_tokens: int = 0, finish_reason: str = "stop")`, and `LLMClient` (a `typing.Protocol` with `async def generate(self, req: GenerationRequest) -> GenerationResult`). `openai_client.py` exports `OpenAIClient(base_url: str, api_key: str, model: str)` implementing `LLMClient`; any HTTP client (`httpx`) is imported lazily/inside methods so import is dependency-light and no network call occurs at construction.
   - Acceptance: `from pharmagpt_vn.models.llm_client import GenerationRequest, GenerationResult, LLMClient` and `from pharmagpt_vn.models.openai_client import OpenAIClient` both succeed; constructing `OpenAIClient(base_url="http://x/v1", api_key="k", model="m")` makes no network call; the existing tests in `tests/unit/test_chat_service.py`, `test_disambiguation_service.py`, `test_hyde.py`, and `test_tracing.py` collect and run. FAIL if any current import site still raises `ModuleNotFoundError`, if `GenerationResult` lacks `finish_reason`, or if constructing `OpenAIClient` performs I/O.

5. **App and full test suite import cleanly**: every module imports and pytest collects with zero import/collection errors.
   - Current: `from pharmagpt_vn.api.main import app` and `pytest` fail at import due to the missing `models` package (R4).
   - Target: after R4, `pharmagpt_vn.api.main:app` imports successfully and `pytest --collect-only` reports 0 errors. No production import path pulls in `FlagEmbedding`, torch, `qdrant_client`, or `redis` at import time (these remain lazily loaded).
   - Acceptance: `pytest --collect-only -q` exits 0 with no collection errors; `python -c "from pharmagpt_vn.api.main import app"` exits 0 offline. FAIL on any `ModuleNotFoundError`/`ImportError` during collection or app import.

6. **Refusal classifier and hybrid-retriever wiring are no longer silent placeholders**: each is either implemented with real (offline-testable) behavior or documented as deliberately deferred so it cannot fail closed unnoticed.
   - Current: `_DefaultRefusalClassifier` always returns `clinical_safe`/0.5 (`dependencies.py:21-29`); `get_retriever()` builds `HybridRetriever` with no `embedder`/`backend`, so `retrieve()` always returns `[]` (`retriever.py:69-70`).
   - Target: provide a deterministic, dependency-free heuristic `RefusalClassifier` (e.g. keyword/regex rules over `QueryClass` that flag obvious `out_of_scope` and `unsafe` Vietnamese/English queries while defaulting clinical drug queries to `clinical_safe`) wired in `dependencies.py`; AND wire `get_retriever()` to construct `HybridRetriever` with a real `embedder` (the `BGEM3Embedder` from R3) and a `QdrantBackend(url, collection, api_key)` so the deployed retriever is functional. If, after investigation, full retriever wiring is judged out of this phase's scope, it MUST be recorded as an explicit deferral note in code and in this SPEC rather than left as an unannotated `return []`.
   - Acceptance: a unit test drives the new classifier with an obvious out-of-scope query ("thời tiết hôm nay?") → `label == "out_of_scope"` and `should_refuse(...) is True`, and a clinical query ("liều metformin cho người suy thận?") → `label == "clinical_safe"`; and `get_retriever(settings)` returns a `HybridRetriever` whose `_embedder` and `_backend` are both not `None` (or the deferral is explicitly documented). FAIL if the wired classifier still returns `clinical_safe` for every input with no rules, or if the retriever remains constructed without `embedder`/`backend` and without a documented deferral.

## Boundaries

**In scope:**
- Real `POST /v1/embeddings` returning BGE-M3 **dense** vectors per input with correct `dim`.
- An offline-overridable embedder DI factory.
- Creating the missing `pharmagpt_vn.models` package (`llm_client`, `openai_client`) to the exact contract its existing call sites require.
- Restoring clean app import and pytest collection.
- Replacing the always-`clinical_safe` refusal placeholder with a deterministic heuristic, and wiring the hybrid retriever (or documenting an explicit deferral).
- Offline tests using fakes (fake embedder, fake/in-memory backends, stub LLM).

**Out of scope:**
- Returning **sparse** vectors or ColBERT vectors from `/v1/embeddings` — the response schema (`EmbedResponse.embeddings: list[list[float]]`) models dense only; sparse output would be a schema change deferred to a follow-up.
- Downloading or running the real BGE-M3 weights in CI — tests must stay offline; a real-model smoke test is a separate, manually-run task.
- The 3-branch query understanding, CRAG, reranker, chunker, ingest, cache, tracing, and chat/disambiguation orchestration logic — already implemented and tested; not modified beyond what R4/R5 require to import.
- Distilling the production ~100M refusal classifier from Qwen2.5-1.5B — that ML training effort (Plan §3.5.2) is explicitly future work; this phase only removes the silent always-safe placeholder.
- Authentication/authorization on `/v1/embeddings` — handled at the Gateway; not introduced here.
- Real Qdrant/Redis connectivity, end-to-end live retrieval quality, and latency KPIs — require infrastructure and a populated corpus; out of an offline phase.

## Constraints
- **Offline tests only**: no model download, no Qdrant, no Redis, no network. Heavy deps (`FlagEmbedding`, torch, `qdrant_client`, `redis`, `httpx` calls) stay lazily imported inside methods, never at module import.
- **Coverage gate**: existing `--cov-fail-under=80` (pyproject `[tool.pytest.ini_options]`) must continue to pass for `src/pharmagpt_vn`.
- **Determinism**: fakes produce fixed vectors; no randomness in assertions.
- **Backward compatibility**: existing `EmbedRequest`/`EmbedResponse` field names and the `ChatService`/`DisambiguationService`/`HyDEGenerator` LLM-client interface are preserved exactly (no breaking signature changes to the 8 call sites).
- **Lint/type gates**: `ruff` (E, F, I, N, W, B, UP, RUF) and `mypy src` as configured must pass.
- **Python ≥ 3.11**, FastAPI app at `pharmagpt_vn.api.main:app`.

## Acceptance Criteria
- [ ] `POST /v1/embeddings` returns one dense vector per input, in input order, with `dim == 1024` under a 1024-d fake embedder, HTTP 200.
- [ ] Empty `inputs` returns 200 with `embeddings == []`, `dim == 0`.
- [ ] `get_embedder` (or equivalent) is overridable via `app.dependency_overrides`; importing the app downloads no model and imports no `FlagEmbedding`/torch.
- [ ] `src/pharmagpt_vn/models/{__init__,llm_client,openai_client}.py` exist; `GenerationRequest`, `GenerationResult` (incl. `finish_reason`), `LLMClient`, `OpenAIClient` import successfully.
- [ ] Constructing `OpenAIClient(...)` performs no network I/O.
- [ ] `pytest --collect-only -q` exits 0 with no collection errors; `python -c "from pharmagpt_vn.api.main import app"` exits 0 offline.
- [ ] Heuristic refusal classifier flags an obvious out-of-scope query (`should_refuse` True) and keeps a clinical drug query `clinical_safe`.
- [ ] `get_retriever()` returns a `HybridRetriever` with non-`None` `_embedder` and `_backend`, OR an explicit deferral note is present in code and this SPEC.
- [ ] New tests are offline (no network/Qdrant/Redis/model download) and the suite passes the 80% coverage gate.

## Ambiguity Report
| Dimension          | Score | Min  | Status | Notes |
|--------------------|-------|------|--------|-------|
| Goal Clarity       | 0.93  | 0.75 | PASS   | Single measurable goal: real embeddings endpoint + named blocking stubs, all offline-testable. |
| Boundary Clarity   | 0.91  | 0.70 | PASS   | Dense-only vs sparse, fakes vs real weights, and which mature subsystems are untouched are all explicit. |
| Constraint Clarity | 0.90  | 0.65 | PASS   | Offline, 80% coverage, lazy heavy imports, backward-compatible signatures, lint/type gates stated. |
| Acceptance Criteria| 0.92  | 0.70 | PASS   | Every requirement has a concrete pass/fail probe with exact dim/order/import checks. |
| **Ambiguity**      | 0.08  | ≤0.20| PASS   | Requirements derived from verified call sites; residual ambiguity only in retriever-wiring depth (bounded by explicit deferral clause). |

## Interview Log
| Round | Perspective     | Question summary | Decision locked |
|-------|-----------------|------------------|-----------------|
| 1     | Researcher      | Is the embedder real and what is the correct dense dim? | `BGEM3Embedder.encode()` exists and returns `EmbeddingPair.dense`; BGE-M3 dense dim = 1024 (qdrant_store.py:8). Endpoint must return dense vectors of `len == dim`. |
| 2     | Simplifier      | Should `/v1/embeddings` also return sparse/ColBERT vectors? | No. Response schema is `list[list[float]]` (dense only). Sparse output deferred — keep scope to dense to avoid a schema change. |
| 3     | Boundary Keeper | The `models` package is missing and blocks all imports — in or out of scope? | In scope as a hard blocker (R4/R5): the embeddings work and the test suite cannot run without it. Contract is fully fixed by existing call sites; do not redesign the interface. |
| 4     | Failure Analyst | What silently fails closed today, and how do we test without infra? | Refusal placeholder always returns `clinical_safe`; wired retriever always returns `[]`. Replace placeholder with a deterministic heuristic and wire (or explicitly defer) the retriever; verify everything with injected fakes and `dependency_overrides` — zero network, Qdrant, Redis, or model download. |

---
*Phase: pharmagpt-embeddings*
*Spec created: 2026-06-09*
