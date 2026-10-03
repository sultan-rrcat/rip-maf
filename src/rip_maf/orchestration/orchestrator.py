"""Orchestrator — thin façade over the LangGraph orchestration graph.

Q28 locked signature: run(request_text, corpus_id, on_event, context,
cancel_event). corpus_id is run-scoped truth: passed to every tool call
(especially rag.query) by the engine, never LLM-generated. context is
MemoryContext.as_prompt() text.

Owns exactly what should NOT live inside the graph:
  - trace-id minting (request correlation)
  - the per-request RunnableConfig: corpus_id, on_event callback,
    cancel_event
  - the OrchestrationError contract: a plan/validation failure routes to the
    graph's error terminal and is raised HERE, after the graph completes
    (fail-honest)

RIP port: multi-tenancy, quotas, approval store, reflection budget, and
Langfuse/audit scaffolding removed.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable

from pydantic import BaseModel, Field

from rip_maf.agents.registry import AgentRegistry
from rip_maf.orchestration.aggregator import Aggregator
from rip_maf.orchestration.engine import OrchestrationState, build_orchestration_graph
from rip_maf.orchestration.plan import Plan
from rip_maf.orchestration.planner import Planner
from rip_maf.orchestration.results import StepResult
from rip_maf.orchestration.validator import PlanValidator
from rip_maf.tools.registry import ToolRegistry

logger = logging.getLogger("orchestration")


class OrchestrationError(Exception):
    """Raised when planning/validation fails before any step executes."""


class OrchestrationResult(BaseModel):
    trace_id: str
    plan_id: str
    goal: str
    step_results: list[StepResult]
    summary: str | None = None
    status: str = "failed"
    plan_incomplete: bool = True
    conflicts: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    shown: list[str] = Field(default_factory=list)
    hidden: list[str] = Field(default_factory=list)
    visibility: dict[str, str] = Field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return all(r.status.value == "success" for r in self.step_results)


class Orchestrator:
    def __init__(
        self,
        planner: Planner,
        validator: PlanValidator,
        aggregator: Aggregator,
        registry: AgentRegistry,
        tool_registry: ToolRegistry | None = None,
    ):
        self._planner = planner
        self._validator = validator
        self._aggregator = aggregator
        self._registry = registry
        self._tool_registry = tool_registry or ToolRegistry()
        self._graph = build_orchestration_graph(
            planner,
            validator,
            registry,
            aggregator,
            tool_registry=self._tool_registry,
        )

    def run(
        self,
        request_text: str,
        corpus_id: str,
        on_event: Callable[[dict], None] | None = None,
        context: str | None = None,
        cancel_event: threading.Event | None = None,
        corpus_context: str | None = None,
    ) -> OrchestrationResult:
        logger.info("run started corpus=%s text=%.120s", corpus_id, request_text)
        trace_id = str(uuid.uuid4())

        # Per-request, per-thread dependencies ride in config (never state):
        # corpus_id for tool injection, on_event for step lifecycle,
        # cancel_event (nodes poll it cooperatively — threads cannot preempt
        # each other) — and thread_id, which keys the checkpointer's
        # snapshots for THIS run.
        # trace_context is the Langfuse run-span context captured HERE in the
        # worker thread (inside manager's `run` span): outer nodes use it for
        # explicit parenting so plan/aggregate land under `run` even when
        # LangGraph schedules them on pool threads.
        from rip_maf.observability.langfuse import get_trace_context

        config = {
            "configurable": {
                "thread_id": trace_id,
                "corpus_id": corpus_id,
                "on_event": on_event,
                "cancel_event": cancel_event,
                "trace_context": get_trace_context(),
            }
        }
        # Same add_node-style overload limitation as plan_graph.py: the
        # generic Pregel.invoke input cannot be matched from a plain dict
        # literal, though the runtime accepts this exact shape.
        final: OrchestrationState = self._graph.invoke(
            {
                "request_text": request_text,
                "corpus_id": corpus_id,
                "context": context,
                "corpus_context": corpus_context,
                "trace_id": trace_id,
                "plan": None,
                "plan_error": None,
                "step_results": {},
                "aggregation": None,
                "route_intent": None,
                "plan_span_ctx": None,
            },
            config=config,  # type: ignore[call-overload]
        )

        if final["plan_error"]:
            # L3 general fallback: no L2 deterministic builder applied
            # (unknown intent, non-deterministic shape, routing miss).
            # Try the ReAct loop before failing honestly — cancellations
            # skip it and raise immediately.
            if cancel_event is not None and cancel_event.is_set():
                logger.warning("orchestration aborted: %s", final["plan_error"])
                raise OrchestrationError(final["plan_error"])
            try:
                from rip_maf.observability.langfuse import (
                    get_trace_context as _get_tc,
                )
                from rip_maf.observability.langfuse import (
                    manual_span as _manual_span,
                )
                from rip_maf.observability.langfuse import (
                    truncate as _truncate,
                )
                from rip_maf.orchestration.react import run_react

                # Trace-only sibling: the `react` span parents explicitly
                # under `run` (via the worker-thread context captured here),
                # so Langfuse reads run → react → react:iter-N → step:rN.
                # Disabled path is a no-op; failures still fall through to
                # the original honest error below.
                with _manual_span(
                    "react",
                    as_type="span",
                    input={
                        "request": _truncate(request_text, 2000),
                        "plan_error": _truncate(final["plan_error"], 500),
                    },
                    trace_context=_get_tc(),
                ) as react_obs:
                    react = run_react(
                        request_text,
                        self._planner.provider,
                        self._registry,
                        self._tool_registry,
                        trace_id=trace_id,
                        corpus_id=corpus_id,
                        context=context,
                        corpus_context=corpus_context,
                        cancel_event=cancel_event,
                        on_event=on_event,
                        parent_span_ctx=_get_tc(),
                        route_intent=final.get("route_intent"),
                    )
                    aggregation = self._aggregator.aggregate(react.plan, react.result)
                    react_obs.update(output={
                        "status": aggregation.status,
                        "steps": len(react.plan.steps),
                        "iterations": len(react.result.step_results),
                        "summary": _truncate(aggregation.summary, 2000),
                    })
                    if aggregation.status != "failed":
                        logger.info(
                            "react fallback recovered plan=%s steps=%d trace=%s",
                            react.plan.plan_id, len(react.plan.steps), trace_id,
                        )
                        return OrchestrationResult(
                            trace_id=trace_id,
                            plan_id=react.plan.plan_id,
                            goal=react.plan.goal,
                            step_results=list(react.result.step_results),
                            summary=aggregation.summary,
                            status=aggregation.status,
                            plan_incomplete=aggregation.plan_incomplete,
                            conflicts=aggregation.conflicts,
                            needs_clarification=aggregation.needs_clarification,
                            shown=list(aggregation.shown),
                            hidden=list(aggregation.hidden),
                            visibility=dict(aggregation.visibility),
                        )
            except OrchestrationError:
                raise
            except Exception as e:  # noqa: BLE001 - react miss → original honest error
                logger.warning("react fallback failed: %s", e)
            logger.warning("orchestration aborted: %s", final["plan_error"])
            raise OrchestrationError(final["plan_error"])

        plan: Plan | None = final["plan"]
        aggregation = final["aggregation"]
        if plan is None or aggregation is None:
            # Unreachable via graph routing; fail-honest rather than None-deref.
            raise OrchestrationError("orchestration graph finished without a plan")

        logger.info(
            "orchestrated plan=%s steps=%d status=%s trace=%s",
            plan.plan_id,
            len(plan.steps),
            aggregation.status,
            trace_id,
        )
        return OrchestrationResult(
            trace_id=trace_id,
            plan_id=plan.plan_id,
            goal=plan.goal,
            step_results=[final["step_results"][s.step_id] for s in plan.steps],
            summary=aggregation.summary,
            status=aggregation.status,
            plan_incomplete=aggregation.plan_incomplete,
            conflicts=aggregation.conflicts,
            needs_clarification=aggregation.needs_clarification,
            shown=list(aggregation.shown),
            hidden=list(aggregation.hidden),
            visibility=dict(aggregation.visibility),
        )
