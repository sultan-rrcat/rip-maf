"""Outer orchestration graph — the LangGraph engine.

The whole round trip is one compiled StateGraph:

    START ──► plan ──(builder hit?)──► execute ──► aggregate ──► END
                      │ (miss → plan_error → END; the orchestrator
                      │  runs L3 ReAct before failing honest)

Planning is L1 → L2 → L3:

- plan node: L1 Router (sole dispatcher, every request via LLM) + L2
  deterministic builders + Validator. Builder misses (unknown intent,
  non-deterministic shapes, unresolvable converts, >5 files) return
  plan_error so the orchestrator runs L3 ReAct. Router failures fail
  open the same way. No mega-prompt, no planner recall.
- execute node: builds + streams the per-request inner plan graph
  (plan_graph.run_plan_graph), forwarding step events to on_event
- aggregate node: deterministic Aggregator (Q36, no LLM), no retry —
  partial/failed runs surface honestly; clarifications are the answer.

Per-request state that must NOT live in graph state:
- `on_event` (per-request callback) and `corpus_id` (run-scoped truth
  injected into tool inputs) travel via RunnableConfig["configurable"] —
  config is per-invocation and never checkpointed, unlike state.

Langfuse parenting (explicit, thread-safe): the orchestrator captures the
run-span context into config["configurable"]["trace_context"]; plan and
aggregate spans parent explicitly under it. The plan node captures its own
span context into state["plan_span_ctx"]; the execute node forwards it to
the inner plan graph so `step:{id}` spans nest INSIDE `plan`
(run → plan → step, aggregate sibling of plan). Plain-string IDs only —
checkpoint-safe, no contextvars dependence across LangGraph threads.

RIP port: approvals, reflection, and correlation helpers removed;
cooperative cancel kept.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rip_maf.agents.registry import AgentRegistry
from rip_maf.observability.langfuse import get_trace_context, manual_span, truncate
from rip_maf.orchestration.aggregator import AggregationResult, Aggregator
from rip_maf.orchestration.builders import build as build_layered_plan
from rip_maf.orchestration.plan import Plan, PlanStep
from rip_maf.orchestration.plan_graph import run_plan_graph
from rip_maf.orchestration.planner import Planner
from rip_maf.orchestration.results import ExecutionResult, StepResult
from rip_maf.orchestration.router import Router
from rip_maf.orchestration.validator import PlanValidationError, PlanValidator
from rip_maf.tools.registry import ToolRegistry

logger = logging.getLogger("orchestration.engine")

_ON_EVENT = "on_event"
_CORPUS_ID = "corpus_id"
_CANCEL_EVENT = "cancel_event"
#: Captured run-span context ({"trace_id", "parent_span_id"}) injected by
#: the orchestrator so outer nodes parent under `run` even when LangGraph
#: schedules them on pool threads without the worker's contextvars.
_TRACE_CTX = "trace_context"


def _cancel_event(config: RunnableConfig) -> threading.Event | None:
    """The run's cooperative cancel flag (None when run without one)."""
    return _configurable(config).get(_CANCEL_EVENT)


def _cancelled(config: RunnableConfig) -> bool:
    event = _cancel_event(config)
    return event is not None and event.is_set()


class OrchestrationState(TypedDict):
    """Outer graph state — serializable, checkpoint-safe."""

    request_text: str
    corpus_id: str | None
    context: str | None
    corpus_context: str | None
    trace_id: str
    plan: Plan | None
    plan_error: str | None
    step_results: dict[str, StepResult]
    aggregation: AggregationResult | None
    #: The L1 router's verdict, carried out of the plan node so the
    #: orchestrator can hand it to L3 ReAct (ADR-035): ReAct executes the
    #: routed intent instead of re-deciding doc-vs-non-doc from scratch.
    route_intent: str | None
    #: Explicit Langfuse parent for `step:{id}` spans: the plan-span context
    #: ({"trace_id", "parent_span_id"}), captured inside the plan span.
    #: None when tracing is off or the plan failed. Lets steps nest INSIDE
    #: `plan` instead of sitting beside it under `run`.
    plan_span_ctx: dict[str, str] | None


def _configurable(config: RunnableConfig) -> dict:
    return config.get("configurable") or {}


def _make_plan_node(
    planner: Planner, validator: PlanValidator, registry: AgentRegistry
) -> Callable:
    def plan_node(state: OrchestrationState, config: RunnableConfig) -> dict:
        # Cooperative cancel: a cancelled run never spends another LLM call
        # on planning; the error terminal unwinds the graph.
        if _cancelled(config):
            return {"plan": None, "plan_error": "run cancelled"}
        run_ctx = _configurable(config).get(_TRACE_CTX)
        # L1 (sole dispatcher) → L2 (deterministic builders). Any miss
        # (unknown intent, non-deterministic shape, unresolvable convert,
        # >5 files, validation failure, router error) returns plan_error so
        # the orchestrator runs L3 ReAct before failing honest.
        # Routing runs BEFORE the plan span opens: the router span is a
        # strict temporal-predecessor sibling (run → router → plan) in the
        # trace timeline, even though both live in this node (Option A: no
        # graph or state changes). Disabled path is a no-op.
        route_info: dict = {
            "layer": "L3-react",
            "intent": "unknown",
            "routed_by": "none",
            "confidence": 0.0,
            "queries": [],
        }
        with manual_span(
            "router",
            as_type="span",
            input={"request": truncate(state["request_text"], 2000)},
            trace_context=run_ctx,
        ) as router_obs:
            try:
                route = Router(planner.provider).route(
                    state["request_text"],
                    context=state.get("context"),
                    corpus_context=state.get("corpus_context"),
                )
                route_info = {
                    "layer": "L2-builder",
                    "intent": route.intent.value,
                    "routed_by": route.routed_by,
                    "confidence": route.confidence,
                }
                router_obs.update(output={
                    "intent": route.intent.value,
                    "confidence": route.confidence,
                    "routed_by": route.routed_by,
                })
                candidate = build_layered_plan(
                    state["request_text"],
                    route,
                    state.get("corpus_context"),
                )
                if candidate is None:
                    logger.info(
                        "no builder for intent=%s, delegating to L3 ReAct",
                        route.intent.value,
                    )
                    return {
                        "plan": None,
                        "plan_error": (
                            f"no deterministic builder for intent "
                            f"{route.intent.value} — L3 ReAct required"
                        ),
                        "route_intent": route.intent.value,
                        "plan_span_ctx": None,
                    }
            except Exception as e:  # noqa: BLE001 - miss fails open to L3 ReAct
                router_obs.update(output={"error": str(e)[:500]})
                logger.info("router/builder miss, delegating to L3 ReAct: %s", e)
                return {
                    "plan": None,
                    "plan_error": f"routing failed ({e}) — L3 ReAct required",
                    "route_intent": route_info["intent"],
                    "plan_span_ctx": None,
                }
        with manual_span(
            "plan", as_type="span",
            input={"request": truncate(state["request_text"], 2000)},
            trace_context=run_ctx,
        ) as plan_obs:
            try:
                plan = candidate
                validator.validate(plan)
                if plan.is_trivial():
                    # Defensive fallback: builders always emit steps, but an
                    # empty plan would execute zero steps and the
                    # deterministic aggregator would report failed ("No steps
                    # were executed."). Route to one conversational
                    # reasoning step instead.
                    fallback_id = "reasoning"
                    try:
                        registry.get(fallback_id)
                    except KeyError:
                        manifest = registry.manifest()
                        if not manifest:
                            raise PlanValidationError(
                                "trivial plan has no steps and no agents registered"
                            )
                        fallback_id = manifest[0]["agent_id"]
                    plan = Plan(
                        plan_id=plan.plan_id,
                        goal=plan.goal,
                        steps=[
                            PlanStep(
                                step_id="1",
                                agent_id=fallback_id,
                                input={"message": state["request_text"]},
                                expected_output_type="text",
                            )
                        ],
                    )
                    validator.validate(plan)
                    logger.info(
                        "trivial plan repaired plan=%s fallback_agent=%s",
                        plan.plan_id, fallback_id,
                    )
            except (ValueError, PlanValidationError) as e:
                err = str(e)
                plan_obs.update(output={
                    "error": err[:500],
                    "layer": route_info["layer"],
                    "intent": route_info["intent"],
                    "routed_by": route_info["routed_by"],
                })
                logger.info("builder plan rejected, delegating to L3 ReAct: %s", err)
                return {
                    "plan": None,
                    "plan_error": f"builder plan rejected ({err}) — L3 ReAct required",
                    "route_intent": route_info["intent"],
                    "plan_span_ctx": None,
                }
            plan_obs.update(output={
                "goal": truncate(plan.goal, 500),
                "steps": len(plan.steps),
                "executors": [s.executor_id for s in plan.steps],
                "layer": route_info["layer"],
                "intent": route_info["intent"],
                "routed_by": route_info["routed_by"],
                "confidence": route_info["confidence"],
            })
            # Capture the plan-span context while it is current so the
            # execute node can parent step:{id} spans explicitly under it.
            # Stored in graph state (plain strings) — survives checkpointing
            # and thread hops where contextvars would be lost.
            plan_span_ctx = get_trace_context()
        # The worker persists + streams this as the SSE `plan` event. Emitted
        # here (plan-time, before any step runs) so live subscribers and the
        # persisted replay both see run_started -> plan -> step_* in order.
        # Additive and None-safe: callers without on_event see no change.
        # `attempt` stays 1 forever (no recall): the frontend uses it to
        # detect retried plans, and a constant keeps old clients working.
        on_event = _configurable(config).get(_ON_EVENT)
        if callable(on_event):
            on_event(
                {
                    "type": "plan",
                    "plan_id": plan.plan_id,
                    "goal": plan.goal,
                    "attempt": 1,
                    "route": {
                        "intent": route_info["intent"],
                        "routed_by": route_info["routed_by"],
                        "confidence": route_info["confidence"],
                    },
                    "steps": [
                        {
                            "step_id": s.step_id,
                            "executor": s.executor_id,
                            "depends_on": list(s.depends_on),
                            "expected_output_type": s.expected_output_type,
                        }
                        for s in plan.steps
                    ],
                }
            )
        return {
            "plan": plan,
            "plan_error": None,
            "route_intent": route_info["intent"],
            "plan_span_ctx": plan_span_ctx,
        }

    return plan_node


def _make_execute_node(
    registry: AgentRegistry, tool_registry: ToolRegistry | None = None
) -> Callable:
    def execute_node(state: OrchestrationState, config: RunnableConfig) -> dict:
        plan = state["plan"]
        if plan is None:  # unreachable via routing; defensive
            return {}

        on_event = _configurable(config).get(_ON_EVENT)
        if not callable(on_event):
            on_event = None

        exec_result = run_plan_graph(
            plan,
            registry,
            tool_registry=tool_registry or ToolRegistry(),
            trace_id=state["trace_id"],
            corpus_id=state["corpus_id"],
            context=state["context"],
            fallback_message=state["request_text"],
            on_event=on_event,
            cancel_event=_cancel_event(config),
            parent_span_ctx=state.get("plan_span_ctx"),
        )
        results = {r.step_id: r for r in exec_result.step_results}
        return {"step_results": results}

    return execute_node


def _make_aggregate_node(aggregator: Aggregator) -> Callable:
    def aggregate_node(state: OrchestrationState, config: RunnableConfig) -> dict:
        plan = state["plan"]
        if plan is None:  # unreachable via routing; defensive
            return {}

        exec_result = ExecutionResult(
            trace_id=state["trace_id"],
            step_results=[state["step_results"][s.step_id] for s in plan.steps],
        )
        run_ctx = _configurable(config).get(_TRACE_CTX)
        with manual_span(
            "aggregate", as_type="span", input={"goal": truncate(plan.goal, 500)},
            trace_context=run_ctx,
        ) as agg_obs:
            agg = aggregator.aggregate(plan, exec_result)
            agg_obs.update(output={
                "status": agg.status,
                "summary": truncate(agg.summary, 2000),
                "shown": list(agg.shown),
                "hidden": list(agg.hidden),
                "visibility": dict(agg.visibility),
            })
        return {"aggregation": agg}

    return aggregate_node


def _route_after_plan(state: OrchestrationState) -> str:
    if state["plan_error"] or state["plan"] is None:
        return END
    return "execute"


def _route_after_aggregate(state: OrchestrationState) -> str:
    return END


def build_orchestration_graph(
    planner: Planner,
    validator: PlanValidator,
    registry: AgentRegistry,
    aggregator: Aggregator,
    *,
    tool_registry: ToolRegistry | None = None,
) -> CompiledStateGraph:
    """Compile the outer graph once per Orchestrator (process-wide)."""
    graph = StateGraph(OrchestrationState)
    graph.add_node(
        "plan", _make_plan_node(planner, validator, registry),
        input_schema=OrchestrationState,  # type: ignore[call-overload]
    )
    graph.add_node(
        "execute", _make_execute_node(registry, tool_registry),
        input_schema=OrchestrationState,  # type: ignore[call-overload]
    )
    graph.add_node(
        "aggregate", _make_aggregate_node(aggregator),
        input_schema=OrchestrationState,  # type: ignore[call-overload]
    )
    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan", _route_after_plan,
        {"execute": "execute", END: END},
    )
    graph.add_edge("execute", "aggregate")
    graph.add_edge("aggregate", END)
    # Checkpointed state: every super-step writes a snapshot keyed by
    # thread_id (= trace_id, set by the façade). MemorySaver is in-process —
    # run durability across processes comes from Postgres run_events (Phase 4);
    # per-request isolation is via the trace id.
    return graph.compile(checkpointer=MemorySaver())
