"""LLM-client package — the generation interface consumed across the engine.

`llm_client` defines the provider-agnostic contract (`GenerationRequest`,
`GenerationResult`, `LLMClient` Protocol); `openai_client` implements it against
any OpenAI-compatible `/v1/chat/completions` endpoint. The HTTP dependency
(`httpx`) is imported lazily inside `OpenAIClient.generate()` so importing this
package — and the FastAPI app — never triggers network setup.
"""

from pharmagpt_vn.models.llm_client import (
    GenerationRequest,
    GenerationResult,
    LLMClient,
)
from pharmagpt_vn.models.openai_client import OpenAIClient

__all__ = [
    "GenerationRequest",
    "GenerationResult",
    "LLMClient",
    "OpenAIClient",
]
