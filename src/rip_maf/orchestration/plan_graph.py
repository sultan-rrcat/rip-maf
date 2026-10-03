"""Inner plan graph — the per-request LangGraph DAG.

Builds and runs a StateGraph from a VALIDATED Plan at runtime:

    START ──► step with no depends_on ──► ... ──► sinks ──► END
    (edges = depends_on; LangGraph runs independent siblings of a
     super-step in parallel)

Per-step node bodies are FACTORIES: each closure captures its PlanStep plus
run-level constants (trace id, corpus id, memory context, fallback
message) — things that cannot change while the graph runs. The graph STATE
carries only what actually flows between nodes: `step_results`, keyed by
step_id behind a merge reducer so parallel siblings write disjoint keys.

Semantics:
  - {{step_id}} placeholder resolution BEFORE the step executes
  - fallback message injection for AGENT steps lacking `message` only
    (tool steps never receive it — e.g. plot.chart ignores prose)
  - scoped short-term-memory `context`: terminal prose AGENT steps only
    (intermediates get task + upstream outputs; the Planner threads
    follow-up references into subtask messages). TOOL steps never get it.
  - `corpus_id` injection into every TOOL step input (Q30): the run's
    corpus id comes from Run.corpus_id, never the LLM — the closure
    overwrites any planner-emitted value
  - status-driven retry loop (agents never raise; LangGraph RetryPolicy is
    exception-driven and therefore does NOT apply to our contract)
  - per-step wall-clock timeout (reports FAILURE; cannot preempt a hung
    worker — pre-existing thread limitation; the span lifecycle lives in
    the node thread so a late background finish can no longer flip the
    span or emit late events, and the orphan aborts between attempts)
  - per-executor deadlines: agent (LLM) steps get the provider budget
    (`ollama_timeout_ms`), deterministic tool steps stay on the tight
    default — one global 30s starves multi-paragraph writes on a 14B model
  - no approval gate (local single-user: tools execute directly)

RIP port: approvals and correlation helpers removed; step lifecycle is
reported through the opaque `on_event` callback as plain dicts (the run
worker maps these onto SSE in Phase 4). Each step opens a `step:{id}`
Langfuse span parented explicitly under the plan span via
`parent_span_ctx` (engine forwards state["plan_span_ctx"]) so steps nest
INSIDE `plan` even across thread hops.
"""

from __future__ import annotations

import contextvars
import logging
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rip_maf.agents.base import DelegationRequest, StepStatus
from rip_maf.agents.registry import AgentRegistry
from rip_maf.core.config import settings
from rip_maf.observability.langfuse import manual_span, truncate
from rip_maf.orchestration.plan import Plan, PlanStep
from rip_maf.orchestration.results import ExecutionResult, StepResult
from rip_maf.tools.registry import ToolRegistry

logger = logging.getLogger("orchestration.plan_graph")

# Matches {{step_id}} placeholders inside a step's input values.
_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_-]+)\s*\}\}")

#: Machine outputs that never need conversation context even when terminal
#: (a lone numbers/chunks step is shown via the aggregator's anti-blank
#: fallback, but the step itself runs on task + upstream data alone).
_MACHINE_OUTPUT_TYPES = frozenset({"chunks", "numbers"})

# Athena default kept as a code constant (no new Settings key): attempts
# beyond the first only help flaky steps; honest failure follows.
_DEFAULT_MAX_RETRIES = 2


def _trunc(text: str | None, limit: int = 2000) -> str | None:
    if text is None or len(text) <= limit:
        return text
    return text[:limit] + "…"


def is_cancelled(cancel_event: threading.Event | None) -> bool:
    """Cooperative cancellation check.

    Threads cannot preempt each other, so cancellation is polled: at the top
    of every step node (between steps) and before every attempt inside the
    retry loop. A cancelled run short-circuits the remaining steps to a
    FAILURE("run cancelled") result.
    """
    return cancel_event is not None and cancel_event.is_set()


def merge_dicts(left: dict, right: dict) -> dict:
    # Module-level ON PURPOSE: LangGraph resolves Annotated reducers via
    # typing.get_type_hints against the module namespace.
    return {**left, **right}


class PlanGraphState(TypedDict):
    """State of the inner plan graph.

    Only data that CHANGES while the graph runs lives here. Sibling steps in
    a super-step each write their OWN key (disjoint writes), so the merge
    reducer is order-free. Downstream nodes read upstream results from this
    dict — the O(1) by-step_id lookup.
    """

    step_results: Annotated[dict[str, StepResult], merge_dicts]


def _outputs_from(results: dict[str, StepResult]) -> dict[str, str]:
    """Successful outputs by step_id — the placeholder-resolution source."""
    return {
        sid: r.output
        for sid, r in results.items()
        if r.status is StepStatus.SUCCESS and r.output is not None
    }


def _try_numeric(s: str) -> int | float | str:
    """Try to parse a string as a number; return the original string on failure.

    This enables tool steps (like plot.chart) to receive numeric values from
    upstream step outputs (e.g. a counting agent returning "29").
    """
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


#: Fail-closed marker substituted for placeholders whose upstream step did
#: not succeed (timed out / failed / unknown). Matches the rag.query empty
#: output so grounded prompts ("Say 'not in the documents' when the chunks
#: are empty") trigger instead of leaking literal "{{id}}" to the LLM
#: (trace cfbaa9c3: summarize ran reasoning on "{{1}} {{2}}" after two
#: overview timeouts and asked the user to paste chunks).
_EMPTY_CHUNKS_MARKER = "(no chunks retrieved)"


def _resolve_value(value: object, outputs: dict[str, str]) -> object:
    if isinstance(value, str):
        # Check if the entire string is a single placeholder (e.g. "{{1}}").
        # If so, resolve and try numeric conversion — enables tool steps to
        # receive numeric values from upstream step outputs.
        m = _PLACEHOLDER.fullmatch(value.strip())
        if m:
            resolved = outputs.get(m.group(1), _EMPTY_CHUNKS_MARKER)
            return _try_numeric(resolved)
        # Embedded placeholder (e.g. "Result is {{1}}") — string substitution
        # only; unknown/failed refs become the empty-chunks marker so the
        # downstream LLM never sees raw "{{id}}" internals.
        return _PLACEHOLDER.sub(
            lambda m: outputs.get(m.group(1), _EMPTY_CHUNKS_MARKER),
            value,
        )
    if isinstance(value, dict):
        return {k: _resolve_value(v, outputs) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_value(v, outputs) for v in value]
    return value


def _resolve_input(raw: dict, outputs: dict[str, str]) -> dict:
    return {k: _resolve_value(v, outputs) for k, v in raw.items()}


def _invoke_with_wall_clock(
    step: PlanStep,
    ctx: contextvars.Context,
    body: Callable[[], StepResult],
    timeout_ms: int,
    cancel_event: threading.Event | None = None,
    expired: threading.Event | None = None,
) -> StepResult:
    """Run one step's full body under a wall-clock deadline.

    A timeout REPORTS FAILURE but cannot preempt the hung worker
    (pre-existing thread limitation). shutdown(wait=False) matters: a
    context-managed executor would block the node until the hung body
    finished, silently turning the reported timeout into a longer stall.
    Cooperative cancel: checked before submit and again on result.
    On timeout `expired` is set so the orphaned background body stays
    side-effect free (no span updates, no late deltas/events — the span
    lifecycle lives in the node thread, not the body).
    """
    if is_cancelled(cancel_event):
        return _cancelled_result(step)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future: Future = executor.submit(ctx.run, body)
        return future.result(timeout=timeout_ms / 1000)
    except FuturesTimeoutError:
        if expired is not None:
            expired.set()
        return StepResult(
            step_id=step.step_id,
            agent_id=step.executor_id,
            status=StepStatus.FAILURE,
            error="step timed out",
        )
    except Exception as e:  # noqa: BLE001 - fail-honest boundary
        return StepResult(
            step_id=step.step_id,
            agent_id=step.executor_id,
            status=StepStatus.FAILURE,
            error=str(e),
        )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _cancelled_result(step: PlanStep) -> StepResult:
    return StepResult(
        step_id=step.step_id,
        agent_id=step.executor_id,
        status=StepStatus.FAILURE,
        error="run cancelled",
    )


def _run_step_body(
    step: PlanStep,
    trace_id: str,
    resolved_input: dict,
    registry: AgentRegistry,
    max_retries: int,
    timeout_ms: int,
    cancel_event: threading.Event | None = None,
    on_delta: Callable[[str], None] | None = None,
    tool_registry: ToolRegistry | None = None,
    expired: threading.Event | None = None,
) -> StepResult:
    """Delegation + status-driven retry loop.

    Agent steps delegate to the AgentRegistry; tool steps dispatch through
    the tool executor (no approval gate — local single-user). All paths
    share the retry loop and the StepResult shape (StepResult.agent_id
    carries the executor identity — agent_id or tool_id).
    `expired` aborts the orphaned background body between attempts after
    the wall clock already reported a timeout (sparing the Ollama server).
    """
    import json as _json

    from rip_maf.tools.executor import execute_tool as _execute_tool

    is_tool = bool(step.tool_id)
    executor_id = step.tool_id or step.agent_id
    agent = None if is_tool else registry.get(step.agent_id)
    agent_request = (
        None
        if is_tool
        else DelegationRequest(
            step_id=step.step_id,
            trace_id=trace_id,
            input=resolved_input,
            timeout_ms=timeout_ms,
            on_delta=on_delta,
        )
    )

    response = None
    tool_data: dict = {}
    for attempt in range(max_retries + 1):
        if is_cancelled(cancel_event):
            return _cancelled_result(step)
        if expired is not None and expired.is_set():
            return StepResult(
                step_id=step.step_id,
                agent_id=executor_id,
                status=StepStatus.FAILURE,
                error="step timed out",
            )
        if is_tool:
            assert step.tool_id is not None  # narrowed by is_tool
            tools = tool_registry or ToolRegistry()
            tool_resp = _execute_tool(
                tools,
                step.tool_id,
                resolved_input,
                step_id=step.step_id,
                trace_id=trace_id,
                timeout_ms=timeout_ms,
            )
            status = StepStatus.SUCCESS if tool_resp.ok else StepStatus.FAILURE
            output = tool_resp.output
            if output is None and tool_resp.data:
                output = _json.dumps(tool_resp.data)[:4000]
            # Keep the raw tool payload for SSE sources/artifacts downstream.
            # Only the final attempt's data is kept.
            tool_data = dict(tool_resp.data) if tool_resp.data else {}
            from rip_maf.agents.base import DelegationResponse as _DelegationResponse

            response = _DelegationResponse(
                step_id=step.step_id,
                status=status,
                output=output,
                error=tool_resp.error,
            )
        else:
            assert agent is not None and agent_request is not None
            response = agent.execute(agent_request)  # never raises; failure status on error
        if response.status is StepStatus.SUCCESS:
            break  # success -> no more retries

        logger.warning(
            "step %s attempt=%d status=%s",
            step.step_id,
            attempt + 1,
            response.status.value,
        )

    if response is None:  # safeguard
        return StepResult(
            step_id=step.step_id,
            agent_id=executor_id,
            status=StepStatus.FAILURE,
            error="No response returned from step execution",
        )

    logger.info(
        "step %s executor=%s status=%s trace=%s",
        step.step_id,
        executor_id,
        response.status.value,
        trace_id,
    )
    return StepResult(
        step_id=step.step_id,
        agent_id=executor_id,
        status=response.status,
        output=response.output,
        error=response.error,
        needs_clarification=response.needs_clarification,
        data=tool_data if is_tool and response.status is StepStatus.SUCCESS else {},
    )


def _step_timeout_ms(step: PlanStep, default_ms: int) -> int:
    """Per-step wall-clock deadline.

    Agent steps stream long text out of a single local Ollama server
    (observed 25–77s for a 5-MCQ write on qwen2.5:14b); they get the
    provider budget (`ollama_timeout_ms`, 120s). Deterministic tool steps
    (rag.query, plot.chart, …) stay on the tight default (30s) — except
    `rag.query` in `overview` mode, which reranks a 3x candidate pool for
    stratification (trace cfbaa9c3: two parallel overviews both hit 30s).
    Overview shards get double the default so summarize/quiz fan-outs
    survive BGE rerank on CPU.
    """
    if step.tool_id:
        if step.tool_id == "rag.query":
            try:
                mode = str((step.input or {}).get("mode", "")).strip().lower()
            except (AttributeError, TypeError):
                mode = ""
            if mode == "overview":
                return default_ms * 2
        return default_ms
    return settings.ollama_timeout_ms


def _make_step_node(
    step: PlanStep,
    *,
    registry: AgentRegistry,
    tool_registry: ToolRegistry | None = None,
    trace_id: str,
    corpus_id: str | None,
    context: str | None,
    fallback_message: str | None,
    max_retries: int,
    timeout_ms: int,
    node_ctx: contextvars.Context,
    on_event: Callable[[dict], None] | None = None,
    cancel_event: threading.Event | None = None,
    is_terminal: bool = False,
    parent_span_ctx: dict[str, str] | None = None,
) -> Callable[[PlanGraphState], dict]:
    """Node factory: closure captures everything constant for this run.

    The returned node takes the graph state (upstream step_results so far),
    resolves its input against it, delegates, and returns its OWN key as a
    partial state update. `is_terminal` marks sink steps (nothing depends on
    them) — only terminal prose agent steps receive memory `context`.
    """
    _eot = (step.expected_output_type or "text").lower()
    _visibility = (
        "hide"
        if _eot in ("chunks", "numbers") or step.executor_id == "corpus.inspect"
        else "show"
    )

    def node(state: PlanGraphState) -> dict:
        # Cancellation fires HERE, in the node body proper — before the
        # executor hop, so no thread is even spawned for a cancelled step.
        if is_cancelled(cancel_event):
            return {"step_results": {step.step_id: _cancelled_result(step)}}
        if on_event is not None:
            on_event(
                {
                    "type": "step_started",
                    "step_id": step.step_id,
                    "executor_id": step.executor_id,
                    "expected_output_type": step.expected_output_type,
                    "visibility": _visibility,
                }
            )

        # Resolve placeholders BEFORE executing so the step sees the
        # resolved, self-descriptive input. Cheap and synchronous: done
        # HERE in the node thread (not the background body) so the span
        # input is accurate even when the body times out.
        # Fail-visible: a dependent step with no placeholder runs ungrounded
        # (the validator rejects this shape, but log here as backstop for
        # plans predating validation or bypassing it in tests).
        if step.depends_on:
            _refs = _PLACEHOLDER.findall(str(step.input))
            if not _refs:
                logger.warning(
                    "step %s depends on %s but input carries no {{id}} "
                    "placeholder — executing ungrounded",
                    step.step_id, sorted(step.depends_on),
                )
        resolved_input = _resolve_input(
            step.input, _outputs_from(state["step_results"])
        )
        is_tool = bool(step.tool_id)
        if is_tool:
            # Deterministic utilities run on schema inputs alone:
            # drop any planner-emitted chatter keys (the run's
            # corpus_id is injected below). The rag.query `message`
            # alias is a first-class schema field, so it stays.
            resolved_input.pop("context", None)
            resolved_input.pop("history", None)
        else:
            if "message" not in resolved_input and fallback_message:
                resolved_input["message"] = fallback_message
            eot = (step.expected_output_type or "text").lower()
            if (
                context
                and "context" not in resolved_input
                and is_terminal
                and eot not in _MACHINE_OUTPUT_TYPES
            ):
                resolved_input["context"] = context
        if step.expected_output_type and "expected_output_type" not in resolved_input:
            resolved_input["expected_output_type"] = step.expected_output_type
        if step.tool_id and corpus_id is not None:
            # Run-scoped truth wins over anything the planner emitted.
            resolved_input["corpus_id"] = corpus_id

        step_timeout = _step_timeout_ms(step, timeout_ms)
        expired = threading.Event()

        def _emit_delta(text: str) -> None:
            # Live-only deltas from an already-timed-out body would stream
            # into chat AFTER the failure was recorded — suppress them.
            if expired.is_set():
                return
            if on_event is not None:
                on_event(
                    {"type": "delta", "step_id": step.step_id, "content": text}
                )

        delta_sink = _emit_delta if on_event is not None else None

        def body() -> StepResult:
            # Side-effect free by contract: no span updates, no step_completed
            # events. If the wall clock already fired, the result is discarded
            # and only compute is wasted — never a trace lie. Abort between
            # attempts when possible to spare the single Ollama server.
            if expired.is_set() or is_cancelled(cancel_event):
                return _cancelled_result(step) if is_cancelled(cancel_event) else StepResult(
                    step_id=step.step_id,
                    agent_id=step.executor_id,
                    status=StepStatus.FAILURE,
                    error="step timed out",
                )
            return _run_step_body(
                step, trace_id, resolved_input, registry, max_retries, step_timeout,
                cancel_event, delta_sink, tool_registry, expired,
            )

        # Span lifecycle lives HERE in the node thread (not the background
        # body): exactly one update, exactly one end — at success OR at the
        # wall-clock timeout. A late background finish can no longer flip a
        # timed-out span to success. The background thread inherits the span
        # via a context copied INSIDE it so generations still nest under it.
        with manual_span(
            f"step:{step.step_id}", as_type="span",
            input=truncate(resolved_input, 2000),
            metadata={
                "executor": step.executor_id,
                "expected_output_type": step.expected_output_type,
                "visibility": _visibility,
            },
            trace_context=parent_span_ctx,
        ) as step_obs:
            run_ctx = contextvars.copy_context()
            result = _invoke_with_wall_clock(
                step, run_ctx, body, step_timeout, cancel_event, expired
            )
            step_obs.update(output={
                "status": result.status.value,
                "output": truncate(result.output, 2000),
                "error": truncate(result.error, 500),
            })
        if on_event is not None:
            on_event(
                {
                    "type": "step_completed",
                    "step_id": step.step_id,
                    "status": result.status.value,
                    "output": _trunc(result.output),
                    "expected_output_type": step.expected_output_type,
                    "visibility": _visibility,
                }
            )
        return {"step_results": {step.step_id: result}}

    return node


def build_plan_graph(
    plan: Plan,
    registry: AgentRegistry,
    *,
    tool_registry: ToolRegistry | None = None,
    trace_id: str,
    corpus_id: str | None = None,
    context: str | None,
    fallback_message: str | None,
    max_retries: int | None = None,
    timeout_ms: int | None = None,
    on_event: Callable[[dict], None] | None = None,
    cancel_event: threading.Event | None = None,
    parent_span_ctx: dict[str, str] | None = None,
) -> tuple[CompiledStateGraph, dict[str, contextvars.Context]]:
    """Compile a StateGraph from a validated Plan — per request, at runtime.

    Returns (compiled_graph, node_contexts): the caller streams the graph;
    node_contexts holds one contextvars snapshot per node.
    """
    retries = max_retries if max_retries is not None else _DEFAULT_MAX_RETRIES
    deadline = timeout_ms if timeout_ms is not None else settings.default_timeout_ms

    step_ids = {s.step_id for s in plan.steps}
    for s in plan.steps:
        unknown = set(s.depends_on) - step_ids
        if unknown:
            raise ValueError(
                f"step {s.step_id} depends on unknown step(s): {sorted(unknown)}"
            )

    # One contextvars copy per node, made HERE (calling thread, sequentially —
    # a Context cannot be entered concurrently, and copy_context() only copies
    # the CURRENT thread's context). Kept for the return contract; span
    # propagation is explicit (parent_span_ctx) plus a fresh copy taken
    # INSIDE each step span so generations nest under it.
    base_ctx = contextvars.copy_context()
    node_ctxs = {s.step_id: base_ctx.run(contextvars.copy_context) for s in plan.steps}

    # Sink steps (nothing depends on them) are terminal: only terminal
    # prose agent steps receive memory context (scoped-context contract).
    depended_upon = {dep for s in plan.steps for dep in s.depends_on}
    terminal_ids = {s.step_id for s in plan.steps} - depended_upon

    graph = StateGraph(PlanGraphState)
    for s in plan.steps:
        graph.add_node(
            s.step_id,
            _make_step_node(
                s,
                registry=registry,
                tool_registry=tool_registry,
                trace_id=trace_id,
                corpus_id=corpus_id,
                context=context,
                fallback_message=fallback_message,
                max_retries=retries,
                timeout_ms=deadline,
                node_ctx=node_ctxs[s.step_id],
                on_event=on_event,
                cancel_event=cancel_event,
                is_terminal=s.step_id in terminal_ids,
                parent_span_ctx=parent_span_ctx,
            ),
            # Explicit input_schema: LangGraph's add_node is generically typed
            # over the node's input; mypy cannot solve that inference from a
            # plain Callable, though the runtime accepts this exact shape.
            input_schema=PlanGraphState,  # type: ignore[call-overload]
        )
    has_dependents = {dep for s in plan.steps for dep in s.depends_on}
    for s in plan.steps:
        if s.depends_on:
            for dep in s.depends_on:
                graph.add_edge(dep, s.step_id)
        else:
            graph.add_edge(START, s.step_id)
        if s.step_id not in has_dependents:
            graph.add_edge(s.step_id, END)
    return graph.compile(), node_ctxs


def run_plan_graph(
    plan: Plan,
    registry: AgentRegistry,
    *,
    tool_registry: ToolRegistry | None = None,
    trace_id: str,
    corpus_id: str | None = None,
    context: str | None = None,
    fallback_message: str | None = None,
    on_step_completed: Callable[[StepResult], None] | None = None,
    max_retries: int | None = None,
    timeout_ms: int | None = None,
    on_event: Callable[[dict], None] | None = None,
    cancel_event: threading.Event | None = None,
    parent_span_ctx: dict[str, str] | None = None,
) -> ExecutionResult:
    """Build + stream the plan graph, emitting events as super-steps complete.

    Streaming (not monolithic invoke) is load-bearing: callers receive step
    events AS they finish. Within a parallel super-step, LangGraph may batch
    sibling updates — cross-super-step ordering (upstream before downstream)
    is guaranteed, intra-wave order is not.

    `parent_span_ctx` is the current attempt's plan-span context: step spans
    parent explicitly under it so they nest INSIDE `plan` in Langfuse.
    """
    if plan.is_trivial():
        return ExecutionResult(trace_id=trace_id, step_results=[])

    started = time.monotonic()
    graph, _node_ctxs = build_plan_graph(
        plan,
        registry,
        tool_registry=tool_registry,
        trace_id=trace_id,
        corpus_id=corpus_id,
        context=context,
        fallback_message=fallback_message,
        max_retries=max_retries,
        timeout_ms=timeout_ms,
        on_event=on_event,
        cancel_event=cancel_event,
        parent_span_ctx=parent_span_ctx,
    )

    results: dict[str, StepResult] = {}
    for chunk in graph.stream({"step_results": {}}, stream_mode="updates"):
        for node_update in chunk.values():
            for step_id, step_result in node_update["step_results"].items():
                results[step_id] = step_result
                if on_step_completed is not None:
                    on_step_completed(step_result)

    logger.info(
        "executed plan=%s steps=%d trace=%s in %dms",
        plan.plan_id,
        len(plan.steps),
        trace_id,
        int((time.monotonic() - started) * 1000),
    )
    return ExecutionResult(
        trace_id=trace_id,
        step_results=[results[s.step_id] for s in plan.steps],
    )
