"""Langfuse integration (observability) — ported from Athena.

Usage:
  - init_langfuse() must run once at app startup (main.py lifespan) when enabled.
  - observe(**kwargs) is a drop-in for langfuse.observe that becomes an
    identity decorator when tracing is disabled — so decorated code paths
    cost nothing and change nothing when Langfuse is off.
  - manual_span(...) is a context manager that opens a span and becomes a
    no-op when tracing is disabled; pass trace_context=... for explicit
    parenting across LangGraph thread hops, or omit it to parent to the
    CURRENT Langfuse span in this thread. get_trace_context() captures the
    current {"trace_id", "parent_span_id"} for forwarding via graph
    state/config — this is how step:{id} spans nest INSIDE the plan span.
  - TracingProvider (app/providers/tracing.py) wraps the ModelProvider so
    every LLM call is recorded as a generation.
  - truncate(v, n) caps inputs/outputs before they reach Langfuse.

RIP adaptation: Settings carries plain str keys (no SecretStr), and there is
no tenant/auth layer — session_id is the corpus_id (one corpus = one
conversation), user_id is the fixed local user.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langfuse import (
    Langfuse,
)
from langfuse import (
    get_client as _get_client,
)
from langfuse import (
    observe as _langfuse_observe,
)
from langfuse import (
    propagate_attributes as _propagate_attributes,
)

from rip_maf.core.config import settings

_client: Langfuse | None = None


def init_langfuse() -> None:
    """Configure the singleton client the observe() decorators use.

    Also exports LANGFUSE_* to the environment: the OTLP exporter reads
    host + credentials from env vars, not from the client instance.
    """
    global _client
    if not settings.langfuse_enabled:
        _client = None
        return
    os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    _client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


def observe(**kwargs):
    """langfuse.observe when enabled, identity decorator otherwise."""
    if settings.langfuse_enabled:
        return _langfuse_observe(**kwargs)

    def identity(func):
        return func

    return identity


def flush() -> None:
    if _client is not None:
        _client.flush()


class _DisabledObservation:
    """Stand-in for a real observation when tracing is off.

    Exposes a minimal `.id` and no-op `.update()` so callers that use
    `manual_span` uniformly never need to branch on whether tracing is
    enabled.
    """

    id: str | None = None

    def update(self, **kwargs: Any) -> None:
        return None


def tracing_enabled() -> bool:
    """True only when tracing is on AND the client has been initialized."""
    return settings.langfuse_enabled and _client is not None


def get_trace_context() -> dict[str, str] | None:
    """Capture the current Langfuse trace context for explicit parenting.

    Returns {"trace_id", "parent_span_id"} for the current observation,
    or None when tracing is disabled / no active span. The dict is
    JSON-serializable, so it can ride in LangGraph state/config across
    threads where contextvars alone would be lost (outer nodes scheduled
    on pool threads). Pass it as `trace_context` to `manual_span` to
    parent a new span under the captured one.
    """
    if not tracing_enabled():
        return None
    assert _client is not None, "tracing_enabled() guarantees a configured client"
    try:
        trace_id = _client.get_current_trace_id()
        parent_span_id = _client.get_current_observation_id()
    except Exception:  # noqa: BLE001 - tracing must never break runs  # pragma: no cover - defensive
        return None
    if not trace_id or not parent_span_id:
        return None
    return {"trace_id": trace_id, "parent_span_id": parent_span_id}


@contextmanager
def manual_span(
    name: str,
    *,
    as_type: Any = "span",
    input: Any = None,
    output: Any = None,
    metadata: Any = None,
    trace_context: dict[str, str] | None = None,
    **extra: Any,
) -> Iterator[_DisabledObservation]:
    """Open a span as the *current* observation, ending it on block exit.

    Parenting is explicit when `trace_context` is given (survives thread
    hops / LangGraph pool scheduling); otherwise it parents to whatever
    Langfuse span is current in this thread. The run worker opens the
    trace root; the orchestrator captures its context into config, the
    plan node captures the plan span context into graph state, and step
    spans parent explicitly under the plan span — so `step:{id}` nest
    inside `plan` in Langfuse instead of sitting beside it as siblings
    under `run`.
    """
    if not tracing_enabled():
        yield _DisabledObservation()
        return
    assert _client is not None, "tracing_enabled() guarantees a configured client"
    with _client.start_as_current_observation(
        name=name,
        as_type=as_type,
        input=input,
        output=output,
        metadata=metadata,
        end_on_exit=True,
        trace_context=trace_context,
        **extra,
    ) as obs:
        yield obs  # type: ignore[misc]


def _norm_meta(metadata: dict[str, Any] | None) -> dict[str, str] | None:
    """Normalize request metadata to Langfuse v4 expectations.

    v4 propagate_attributes() requires metadata as dict[str, str] with values
    capped at 200 characters; extras are dropped (with a warning) instead of
    failing.
    """
    if not metadata:
        return None
    out: dict[str, str] = {}
    for k, v in metadata.items():
        if not isinstance(k, str):
            continue
        if v is None:
            continue
        if not isinstance(v, str):
            v = str(v)
        out[k] = v[:200]
    return out or None


@contextmanager
def request_attributes(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    trace_name: str | None = None,
) -> Iterator[None]:
    """Attach correlating attributes to the current trace and ALL child spans.

    Wraps Langfuse's v4 propagate_attributes() so run-scoped attributes land
    on every observation in the trace, not just the root. Becomes a no-op
    when tracing is disabled.
    """
    if not tracing_enabled():
        yield
        return
    with _propagate_attributes(
        session_id=session_id,
        user_id=user_id,
        metadata=_norm_meta(metadata),
        tags=tags,
        trace_name=trace_name,
        environment=settings.langfuse_environment or None,
        version=settings.langfuse_release or None,
        as_baggage=True,
    ):
        yield


def update_generation(**kwargs: Any) -> None:
    """Update the current generation observation (model, usage_details, I/O).

    No-op when tracing is disabled; silently survives pre-init.
    """
    if not tracing_enabled():
        return
    _get_client().update_current_generation(**kwargs)


@contextmanager
def manual_generation(
    name: str,
    *,
    input: Any = None,
    model: str | None = None,
    model_parameters: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Open a generation observation WITHOUT becoming the current one.

    Unlike manual_span, this does NOT touch the current-observation
    contextvar — the generation is parented to the current trace but never
    becomes current itself. A TOKEN STREAM yields control back to its
    consumer between chunks, and anything the consumer does in between must
    not accidentally nest under a half-finished generation.

    The caller holds the returned observation and MUST end it (the
    contextmanager does that on block exit, including on exception). No-op
    (_DisabledObservation) when tracing is disabled.
    """
    if not tracing_enabled():
        yield _DisabledObservation()
        return
    assert _client is not None, "tracing_enabled() guarantees a configured client"
    obs = _client.start_observation(
        name=name,
        as_type="generation",
        input=input,
        model=model,
        model_parameters=model_parameters,
        metadata=metadata,
    )
    try:
        yield obs
    finally:
        obs.end()


def truncate(value: Any, n: int = 2000) -> str | None:
    """Flatten `value` to a bounded string for Langfuse metadata.

    Prevents huge outputs / long inputs from bloating traces.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value[:n]
    try:
        text = json.dumps(value, default=str)
    except Exception:  # noqa: BLE001 - truncate fallback must never raise  # pragma: no cover - extremely defensive
        text = str(value)
    return text[:n]
