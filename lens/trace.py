"""The tracing SDK.

Design decisions worth defending:

* **A trace is a run, and runs contain spans.** One user question produces one
  run; every model call, retrieval, and tool invocation inside it becomes a span.
  This is the same shape as OpenTelemetry, so the mental model transfers.

* **Safety by default.** If you call `lens.llm()` without opening a trace, the SDK
  still records it as an implicit single-step run instead of dropping it. Losing
  the one call you forgot to wrap is worse than an extra row.

* **Span context lives in a ContextVar.** Parallel branches (async retrieval fan-out)
  therefore attribute their children to the right parent instead of to whichever
  coroutine finished last.

* **Batched writes.** Spans collect in memory and flush once when the run closes.
  A twenty-step agent does not pay twenty fsyncs.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

from . import pricing
from .config import canonical_name, default_model, resolve_provider
from .db import Store, new_id
from .providers import ChatResult, get_provider

MAX_PAYLOAD_CHARS = 8000

_current_run: ContextVar["Run | None"] = ContextVar("lens_current_run", default=None)
_current_spans: ContextVar[tuple["Span", ...]] = ContextVar("lens_span_stack", default=())


def _clip(value: Any, limit: int = MAX_PAYLOAD_CHARS) -> Any:
    """Keep traces readable without letting a 200KB prompt bloat the database."""
    if value is None:
        return None
    text = value if isinstance(value, str) else repr(value)
    if len(text) <= limit:
        return value if isinstance(value, str) else text
    return text[:limit] + f"...[truncated {len(text) - limit} chars]"


@dataclass
class Span:
    id: str
    run_id: str
    name: str
    kind: str = "custom"
    parent_id: str | None = None
    input: Any = None
    output: Any = None
    attributes: dict = field(default_factory=dict)
    status: str = "ok"
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    duration_ms: float | None = None
    model: str | None = None
    provider: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    _t0: float = field(default_factory=time.perf_counter, repr=False)

    def set(self, **attrs: Any) -> "Span":
        self.attributes.update(attrs)
        return self

    def record_llm(self, result: ChatResult, provider: str) -> "Span":
        self.model = result.model
        self.provider = provider
        self.prompt_tokens = result.prompt_tokens
        self.completion_tokens = result.completion_tokens
        self.output = _clip(result.text)

        cost = pricing.cost_usd(provider, result.model, result.prompt_tokens, result.completion_tokens)
        if cost is None:
            self.attributes["pricing"] = "unknown"
        else:
            self.cost_usd = cost
            price = pricing.lookup(provider, result.model)
            self.attributes["pricing"] = {
                "matched": price.matched if price else None,
                "snapshot": price.snapshot if price else None,
            }
        if result.finish_reason:
            self.attributes["finish_reason"] = result.finish_reason
        return self

    def close(self, output: Any = None, error: str | None = None) -> None:
        if output is not None:
            self.output = _clip(output)
        if error:
            self.status = "error"
            self.error = error
        self.duration_ms = round((time.perf_counter() - self._t0) * 1000, 2)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "error": self.error,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "model": self.model,
            "provider": self.provider,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
            "input": _clip(self.input),
            "output": _clip(self.output),
            "attributes": self.attributes,
        }


@dataclass
class Run:
    id: str
    project: str
    name: str
    input: Any = None
    output: Any = None
    label: str | None = None
    origin: str = "app"          # "app" for real traffic, "eval" for harness runs
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    status: str = "ok"
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    duration_ms: float | None = None
    spans: list[Span] = field(default_factory=list)

    _t0: float = field(default_factory=time.perf_counter, repr=False)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(s.cost_usd for s in self.spans), 8)

    @property
    def total_tokens(self) -> int:
        return sum(s.prompt_tokens + s.completion_tokens for s in self.spans)

    @property
    def current_span(self) -> Span | None:
        stack = _current_spans.get()
        return stack[-1] if stack else None

    def close(self, error: str | None = None) -> None:
        if error:
            self.status = "error"
            self.error = error
        self.ended_at = time.time()
        self.duration_ms = round((time.perf_counter() - self._t0) * 1000, 2)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project": self.project,
            "name": self.name,
            "input": _clip(self.input),
            "output": _clip(self.output),
            "status": self.status,
            "error": self.error,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "label": self.label,
            "origin": self.origin,
            "tags": self.tags,
            "metadata": self.metadata,
        }


class Lens:
    def __init__(
        self,
        project: str = "default",
        *,
        store: Store | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        self.project = project
        self.store = store or Store()
        self.provider = canonical_name(provider or resolve_provider().name)
        self.model = model or default_model(self.provider)

    # ------------------------------------------------------------ lifecycle

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        input: Any = None,
        label: str | None = None,
        origin: str = "app",
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> Iterator[Run]:
        run = Run(
            id=new_id("run"),
            project=self.project,
            name=name,
            input=input,
            label=label,
            origin=origin,
            tags=tags or [],
            metadata=metadata or {},
        )
        run_token = _current_run.set(run)
        span_token = _current_spans.set(())
        error: str | None = None
        try:
            yield run
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            stack = _current_spans.get()
            for span in reversed(stack):
                span.close(error=error if span.status == "ok" else None)
            run.close(error=error)
            _flush(self.store, run)
            _current_spans.reset(span_token)
            _current_run.reset(run_token)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: str = "custom",
        input: Any = None,
        **attributes: Any,
    ) -> Iterator[Span]:
        run = _current_run.get()
        implicit = run is None
        if run is None:
            run = Run(id=new_id("run"), project=self.project, name=name, input=input)
            run_token = _current_run.set(run)
            span_token = _current_spans.set(())
        else:
            run_token = span_token = None  # type: ignore[assignment]

        parent = run.current_span
        span = Span(
            id=new_id("sp"),
            run_id=run.id,
            name=name,
            kind=kind,
            parent_id=parent.id if parent else None,
            input=input,
            attributes=dict(attributes),
        )
        stack_token = _current_spans.set(_current_spans.get() + (span,))
        error: str | None = None
        try:
            yield span
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.close(error=error)
            run.spans.append(span)
            _current_spans.reset(stack_token)
            if implicit:
                run.close()
                _flush(self.store, run)
                _current_spans.reset(span_token)  # type: ignore[arg-type]
                _current_run.reset(run_token)  # type: ignore[arg-type]

    # ------------------------------------------------------------------- llm

    def llm(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        provider: str | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        provider_name = canonical_name(provider or self.provider)
        model_name = model or self.model
        span_name = name or f"llm:{model_name.split('/')[-1]}"

        with self.span(span_name, kind="llm", input=messages[-1].get("content") if messages else None) as span:
            span.set(messages=len(messages))
            client = get_provider(provider_name)
            result = client.chat(model_name, messages, **kwargs)
            span.record_llm(result, provider_name)
        return result

    def __call__(self, messages: list[dict], **kwargs: Any) -> ChatResult:
        return self.llm(messages, **kwargs)


def _flush(store: Store, run: Run) -> None:
    store.save_run(run.to_dict())
    for span in run.spans:
        store.save_span(span.to_dict())
