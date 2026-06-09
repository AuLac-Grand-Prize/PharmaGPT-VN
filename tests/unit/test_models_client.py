"""pharmagpt_vn.models — LLM-client contract + OpenAI-compatible implementation.

Verifies the package the rest of the engine imports: dataclass shapes/defaults,
no network at `OpenAIClient` construction, and that `generate()` builds the
correct OpenAI request and parses the response — all with httpx mocked, no
real socket.
"""

from __future__ import annotations

import pytest

from pharmagpt_vn.models.llm_client import (
    GenerationRequest,
    GenerationResult,
    LLMClient,
)
from pharmagpt_vn.models.openai_client import OpenAIClient, _parse_completion


def test_generation_request_defaults() -> None:
    r = GenerationRequest(prompt="hello")
    assert r.prompt == "hello"
    assert isinstance(r.temperature, float)
    assert isinstance(r.max_tokens, int)


def test_generation_result_defaults_finish_reason_stop() -> None:
    g = GenerationResult(text="hi")
    assert g.text == "hi"
    assert g.prompt_tokens == 0
    assert g.completion_tokens == 0
    assert g.finish_reason == "stop"


def test_openai_client_construction_does_no_io() -> None:
    # Constructing must not open a socket. We assert via behaviour: it returns
    # immediately and stores normalised config. (No network mocking needed —
    # any real call here would hang/raise; construction returns instantly.)
    c = OpenAIClient(base_url="http://x/v1/", api_key="k", model="m")
    assert c.base_url == "http://x/v1"  # trailing slash trimmed
    assert c.api_key == "k"
    assert c.model == "m"


def test_openai_client_satisfies_llm_client_protocol() -> None:
    c = OpenAIClient(base_url="http://x/v1", api_key="k", model="m")
    assert isinstance(c, LLMClient)  # runtime_checkable Protocol


def test_parse_completion_maps_fields() -> None:
    data = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Câu trả lời."},
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7},
    }
    result = _parse_completion(data)
    assert result.text == "Câu trả lời."
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 7
    assert result.finish_reason == "length"


def test_parse_completion_tolerates_missing_fields() -> None:
    result = _parse_completion({})
    assert result.text == ""
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
    assert result.finish_reason == "stop"


@pytest.mark.asyncio
async def test_generate_posts_openai_payload_and_parses(monkeypatch) -> None:
    captured: dict = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {"content": "OK [REF:1]."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            }

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            captured["client_kwargs"] = kwargs

        async def __aenter__(self) -> _FakeAsyncClient:
            return self

        async def __aexit__(self, *exc) -> None:
            return None

        async def post(self, url: str, json: dict, headers: dict) -> _FakeResponse:
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _FakeResponse()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    client = OpenAIClient(base_url="http://llm/v1", api_key="secret", model="gpt-x")
    result = await client.generate(
        GenerationRequest(prompt="liều metformin?", temperature=0.1, max_tokens=128)
    )

    # Request shape: OpenAI chat-completions on the right URL.
    assert captured["url"] == "http://llm/v1/chat/completions"
    assert captured["json"]["model"] == "gpt-x"
    assert captured["json"]["messages"] == [
        {"role": "user", "content": "liều metformin?"}
    ]
    assert captured["json"]["temperature"] == 0.1
    assert captured["json"]["max_tokens"] == 128
    assert captured["headers"]["Authorization"] == "Bearer secret"

    # Response parsed onto GenerationResult.
    assert result.text == "OK [REF:1]."
    assert result.prompt_tokens == 5
    assert result.completion_tokens == 3
    assert result.finish_reason == "stop"
