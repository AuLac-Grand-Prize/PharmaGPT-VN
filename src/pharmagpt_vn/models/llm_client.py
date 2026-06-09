"""Provider-agnostic LLM generation contract (Plan §3.6).

Every component that calls an LLM — `ChatService`, `DisambiguationService`,
`HyDEGenerator` — depends only on this module, never on a concrete provider.
That keeps the orchestration layer testable with a tiny stub (see the unit
tests) and lets the deployed provider be swapped via `api/dependencies.py`.

The surface here is intentionally minimal and frozen, because eight call sites
already depend on these exact shapes:

* ``GenerationRequest(prompt, temperature, max_tokens)``
* ``GenerationResult(text, prompt_tokens, completion_tokens, finish_reason)``
* ``LLMClient.generate(req) -> GenerationResult`` (awaitable)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Sampling defaults mirror `core/config.py` (`default_temperature` / `default_max_tokens`)
# so a bare `GenerationRequest(prompt=...)` behaves like the app's configured default.
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 512


@dataclass(frozen=True)
class GenerationRequest:
    """A single, stateless completion request.

    `prompt` is the fully-rendered prompt string (callers do their own
    templating). `temperature` / `max_tokens` carry sampling controls; both
    default so callers that don't care can omit them.
    """

    prompt: str
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS


@dataclass(frozen=True)
class GenerationResult:
    """The model's reply plus token accounting.

    `finish_reason` defaults to ``"stop"`` so stubs and providers that don't
    surface a reason still satisfy downstream code (`ChatResult.finish_reason`,
    tracing spans) that always reads the field.
    """

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"


@runtime_checkable
class LLMClient(Protocol):
    """Anything that can turn a `GenerationRequest` into a `GenerationResult`.

    Implementations must be awaitable; the orchestration layer always
    `await`s `generate`.
    """

    async def generate(self, req: GenerationRequest) -> GenerationResult: ...


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "GenerationRequest",
    "GenerationResult",
    "LLMClient",
]
