"""OpenAI-compatible LLM client (Plan §3.6).

Talks to any endpoint exposing `POST {base_url}/chat/completions` with the
OpenAI request/response shape — the project's self-hosted vLLM/SGLang gateway,
OpenAI proper, OpenRouter, etc. The concrete provider is chosen at the edge in
`api/dependencies.py`; everything downstream only sees `LLMClient`.

Design constraints honoured here:

* **No I/O at construction.** ``__init__`` only stores config — building an
  ``OpenAIClient`` never opens a socket, so importing the FastAPI app and
  running offline tests touch no network.
* **Lazy, dependency-light HTTP.** ``httpx`` is imported inside ``generate`` so
  merely importing this module doesn't require the HTTP stack to be present.
"""

from __future__ import annotations

from pharmagpt_vn.models.llm_client import GenerationRequest, GenerationResult

# Per-request network budget (seconds). Generous because clinical generations
# with long contexts can run for a while on the self-hosted endpoint.
DEFAULT_TIMEOUT_S = 60.0


class OpenAIClient:
    """`LLMClient` over an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        # Normalise once so callers may pass either ".../v1" or ".../v1/".
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def generate(self, req: GenerationRequest) -> GenerationResult:
        # Imported lazily: keeps module import dependency-light and ensures no
        # network machinery is created until an actual call is made.
        import httpx

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": req.prompt}],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
        return _parse_completion(data)


def _parse_completion(data: dict) -> GenerationResult:
    """Map an OpenAI chat-completion response onto `GenerationResult`.

    Tolerant of missing fields: a provider that omits `usage` or `finish_reason`
    still yields a well-formed result (zero tokens, ``"stop"``) rather than
    raising — generation must never crash on a benign schema gap.
    """
    choices = data.get("choices") or []
    first = choices[0] if choices else {}
    message = first.get("message") or {}
    text = message.get("content") or ""
    finish_reason = first.get("finish_reason") or "stop"
    usage = data.get("usage") or {}
    return GenerationResult(
        text=text,
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        finish_reason=finish_reason,
    )


__all__ = ["OpenAIClient"]
