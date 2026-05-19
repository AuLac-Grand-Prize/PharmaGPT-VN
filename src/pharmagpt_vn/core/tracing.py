"""Pipeline tracing — zero-dep, Phoenix/Langfuse-ready Protocol.

ChatService stages (classify → retrieve → rerank → CRAG → generate → validate)
are wrapped in spans so we can see where latency goes and where retrieval
quality dies. Default is a `NoopTracer` so tests/dev pay nothing; production
wires `StructuredLogTracer` or a vendor-specific adapter (Phoenix, Langfuse).

Design
------
- `Tracer.start_span(name, **attrs)` returns a context-manager `Span` that
  records start time, end time, and arbitrary attributes via `set_attribute`.
- `Span` is async-safe (only stores state; no await needed in __aenter__).
- Backends are swap-in: implement `Tracer.start_span` and you're done.

This is intentionally a thin shim — the goal is to keep the pipeline call sites
ergonomic, not to reimplement OpenTelemetry. A real vendor adapter forwards to
its own span machinery inside `start_span`.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger("pharmagpt_vn.trace")


@dataclass
class Span:
    name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: float = field(default_factory=time.perf_counter)
    ended_at: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None
    status: str = "ok"

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_status(self, status: str) -> None:
        self.status = status

    def end(self) -> None:
        if self.ended_at is None:
            self.ended_at = time.perf_counter()

    @property
    def duration_ms(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.perf_counter()
        return (end - self.started_at) * 1000.0


class Tracer(Protocol):
    @contextmanager
    def start_span(self, name: str, **attrs: Any) -> Iterator[Span]: ...


class NoopTracer:
    """Default — discards all spans, zero overhead in hot paths."""

    @contextmanager
    def start_span(self, name: str, **attrs: Any) -> Iterator[Span]:
        span = Span(name=name, attributes=dict(attrs))
        try:
            yield span
        finally:
            span.end()


class StructuredLogTracer:
    """Writes one structured JSON log line per finished span.

    Practical for shipping spans to ELK / Loki / Datadog without a vendor SDK.
    """

    def __init__(self, logger_: logging.Logger | None = None) -> None:
        self._logger = logger_ or logger

    @contextmanager
    def start_span(self, name: str, **attrs: Any) -> Iterator[Span]:
        span = Span(name=name, attributes=dict(attrs))
        try:
            yield span
        except Exception:
            span.set_status("error")
            raise
        finally:
            span.end()
            self._logger.info(
                json.dumps(
                    {
                        "span": span.name,
                        "span_id": span.span_id,
                        "parent_id": span.parent_id,
                        "duration_ms": round(span.duration_ms, 2),
                        "status": span.status,
                        "attributes": _safe_attrs(span.attributes),
                    },
                    ensure_ascii=False,
                )
            )


class InMemoryTracer:
    """Captures every span in a list — useful for tests and harness eval."""

    def __init__(self) -> None:
        self.spans: list[Span] = []

    @contextmanager
    def start_span(self, name: str, **attrs: Any) -> Iterator[Span]:
        span = Span(name=name, attributes=dict(attrs))
        self.spans.append(span)
        try:
            yield span
        except Exception:
            span.set_status("error")
            raise
        finally:
            span.end()


def _safe_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Coerce non-serializable values so the log line never silently fails."""
    out: dict[str, Any] = {}
    for k, v in attrs.items():
        try:
            json.dumps(v, ensure_ascii=False)
            out[k] = v
        except TypeError:
            out[k] = repr(v)
    return out


__all__ = [
    "InMemoryTracer",
    "NoopTracer",
    "Span",
    "StructuredLogTracer",
    "Tracer",
]
