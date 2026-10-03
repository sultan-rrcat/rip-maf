"""L3 ReAct engine — general fallback when no L2 deterministic builder applies.

Unlike upfront DAG planning, ReAct interleaves thought → action → observation:
each iteration proposes exactly ONE step, executes it immediately, and appends
the observation to the scratchpad. No placeholder wiring is ever emitted, so
the ecd93eb4 ungrounded-fan-in class cannot occur — inputs are inlined.

Route-aware (ADR-035): the L1 router's verdict and the corpus's corpus
state ride in on `run()` and are stated in the prompt, so ReAct *executes* a
decided intent instead of re-deriving doc-vs-non-doc from scratch on every
iteration. Two deterministic steering rules follow from the corpus state:
- empty/processing corpus: a proposed `rag.query` is substituted by a
  general-knowledge `reasoning` step (it executes, so the repeat guards
  engage) instead of being refused as a wasted iteration;
- unknown corpus (inventory unavailable): the prompt mandates
  `corpus.inspect` as the first step — the one case where probing
  carries information the prompt does not already hold.

Bounded: max 6 iterations, cooperative cancel, per-step timeouts inherited
from run_plan_graph. Returns a (Plan, ExecutionResult) pair so the standard
deterministic Aggregator stays the single answer-assembly path. A loop that
ends with nothing executed still answers from the request verbatim (same
shape as the `chat` builder) rather than surfacing a routing error.

The loop lives on `ReActEngine` (composition-time deps in the constructor,
per-request args on `run()`); the idle-turn guard lives in `idle_guard`;
corpus tri-stating lives in `corpus`. Module-level `run_react()` remains
as a thin wrapper so existing callers are unaffected.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections.abc import Callable

from rip_maf.agents.base import StepStatus
from rip_maf.agents.registry import AgentRegistry
from rip_maf.core.config import settings
from rip_maf.observability.langfuse import (
    get_trace_context as _get_trace_context,
)
from rip_maf.observability.langfuse import (
    manual_span as _manual_span,
)
from rip_maf.observability.langfuse import (
    truncate as _truncate,
)
from rip_maf.orchestration.corpus import get_corpus_state
from rip_maf.orchestration.idle_guard import IdleGuard
from rip_maf.orchestration.intents import DOC_INTENTS, REACT_ONLY_INTENTS
from rip_maf.orchestration.plan import Plan, PlanStep
from rip_maf.orchestration.plan_graph import run_plan_graph
from rip_maf.orchestration.results import ExecutionResult, StepResult
from rip_maf.providers.base import ModelProvider
from rip_maf.tools.registry import ToolRegistry

logger = logging.getLogger("orchestration.react")

MAX_REACT_ITERATIONS = 6

#: Intents whose first step must produce numbers before any chart. Derived
#: from the taxonomy (both plot intents) so a new REACT_ONLY intent cannot
#: silently skip the rule.
_NUMBERS_FIRST_INTENTS = frozenset(i.value for i in REACT_ONLY_INTENTS)

#: Doc-intent values — a document question on an unknown inventory needs
#: corpus.inspect before anything else (ADR-027 freshness probe).
_DOC_INTENT_VALUES = frozenset(i.value for i in DOC_INTENTS)

#: Corpus state → what ReAct may and may not do (prompt input, ADR-035).
_CORPUS_ROUTE_LINES = {
    "ready": "the corpus HAS ready documents — rag.query can answer document questions",
    "empty": "the corpus is EMPTY (nothing uploaded) — rag.query returns nothing and is never run",
    "processing": (
        "the corpus's files are still uploading/processing — rag.query returns "
        "nothing right now and is never run"
    ),
    "unknown": (
        "the file inventory is unavailable — your FIRST step must be "
        "corpus.inspect {} (it takes no input) to see what is uploaded"
    ),
}

REACT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "executor": {"type": "string"},
        "input": {"type": "object"},
        "is_final": {"type": "boolean"},
        "answer": {"type": "string"},
    },
    "required": ["thought", "executor", "is_final"],
}

_TOOL_OUTPUT_TYPES = {
    "rag.query": "chunks",
    "plot.chart": "chart",
    "doc.generate": "document",
    "doc.convert": "document",
    "image.generate": "document",
}

def _tool_examples(tool_ids: list[str], tools: ToolRegistry) -> str:
    """One copyable call per tool, taken from the tool that owns it.

    ReAct used to carry a hand-written hint table for all seven tools; four
    copies of the plot.chart shape drifted until the model could not produce
    a two-series chart (trace c9b02eef). The shape shown to the model is now
    the tool's own `input_example`, and a rejection is the tool's own
    `explain_invalid()` — so a rule lives in exactly one place: the tool.
    """
    parts: list[str] = []
    for tool_id in tool_ids:
        if tool_id not in tools:
            continue
        example = tools.get(tool_id).input_example
        if example:
            parts.append(f"- {tool_id}: {example}")
    return "\n".join(parts)


def _plot_rules() -> str:
    """plot.chart policy. The tool's SHAPE lives in `input_example` and is
    rendered once by `_tool_examples`; only policy belongs here."""
    return (
        "Bar/line charts MUST use plot.chart with literal numbers from "
        "observations (or a prior reasoning step). Every plot.chart should "
        "name its metric in a short "
        "'title' (e.g. 'mAP@50-95: FASDD_CV vs AgniNetra'). Each plot.chart "
        "must cover a DIFFERENT metric — never re-plot numbers already "
        "charted."
    )


def _normalize_react_input(executor: str, action_input: dict) -> dict:
    """Unwrap common ReAct model slips into flat tool/agent inputs.

    - {"agent": {"message": ...}} → top-level "message" (observed live
      for rag.query in the same run).
    - stray {"tool_id": ...} inside input → dropped (executor already
      selects the tool; the key only confuses required-field checks).
      - rag.query message→query alias (mirrors RagQueryTool.execute). The
      alias is MOVED, not copied: tool key sets are closed, so leaving
      `message` behind after mapping it would be rejected as litter.
    Pure function — safe to unit test without Ollama/DB.
    """
    normalized = dict(action_input)
    nested = normalized.get("agent")
    if isinstance(nested, dict) and isinstance(nested.get("message"), str):
        if not str(normalized.get("message", "")).strip():
            normalized["message"] = nested["message"]
        normalized.pop("agent", None)
    normalized.pop("tool_id", None)
    alias_target = {
        "rag.query": "query",
    }.get(executor)
    if alias_target is not None:
        message = str(normalized.get("message", "") or "").strip()
        if message and not str(normalized.get(alias_target, "") or "").strip():
            normalized[alias_target] = message
        normalized.pop("message", None)
    return normalized


def _validate_react_input(
    executor: str, action_input: dict, tools: ToolRegistry
) -> str | None:
    """Return None when valid, else the tool's own correct-shape hint.
    The pre-flight check exists for iteration economics, not correctness:
    `execute_tool` validates too, but a failing step is retried twice inside
    `run_plan_graph`, so a malformed call would cost three executions and
    three identical errors in the scratchpad. Caught here it costs one idle
    turn and nothing else (ADR-031).
    The rules are NOT re-implemented here -- `tools.get(executor)` owns them
    (schema + cross-field + example). This function only decides *whether* to
    reject, using the tool that will actually run.
    """
    if executor not in tools:
        return None  # agents (the message check lives at the call site)
    return tools.get(executor).explain_invalid(action_input)


def _output_type(executor: str, is_final: bool) -> str:
    if is_final:
        return "answer"
    tool_type = _TOOL_OUTPUT_TYPES.get(executor)
    if tool_type is not None:
        return tool_type
    # A non-final AGENT step is a scratchpad observation, not an answer.
    # Typed "text" it was SHOWn next to the final synthesis, duplicating it
    # and leaking ASCII-art redraws of a chart the tool would have drawn
    # (trace c9b02eef's summary led with an ASCII India/China plot). Hidden
    # type instead: the aggregator's anti-blank fallback still surfaces it
    # when it is the only output.
    return "observation"


#: Fields the ReAct model may (wrongly) put the final answer into when it
#: sets is_final=true but leaves `answer` empty (trace 35e8fbd9: the full
#: table rode inside input.content of a malformed doc.generate call — the
#: loop discarded a correct answer because it only read raw["answer"]).
_ANSWER_FALLBACK_FIELDS = ("answer", "content", "message", "text", "body", "output")


def _fallback_answer_text(action_input: dict) -> str:
    """Best-effort recovery of a final answer stranded in `input` fields."""
    for field in _ANSWER_FALLBACK_FIELDS:
        value = action_input.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _action_signature(executor: str, action_input: dict) -> str:
    """Stable id for a proposed action — repeats of a failed action loop out."""
    try:
        return executor + "\0" + json.dumps(action_input, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return executor + "\0" + str(sorted(action_input))


def _plot_data_key(action_input: dict) -> tuple | None:
    """Normalized data identity for a plot.chart proposal.

    Same chart data under cosmetic tweaks (retitled, relabeled — trace
    07fb4f59 r5/r6 re-plotted [229, 135] with different labels) renders the
    same bars, so the frontend would show the same plot twice. The key
    covers chart_type + the numeric data only, ignoring title/labels/legend
    cosmetics. Returns None when the numbers cannot be read (validation
    owns that shape — this is a dedupe helper, not a validator).
    """
    try:
        chart_type = str(action_input.get("chart_type", "")).strip()
        series = action_input.get("series")
        if isinstance(series, list) and series:
            parts = []
            for entry in series:
                if not isinstance(entry, dict):
                    return None
                numbers = tuple(float(v) for v in (entry.get("values") or []))
                parts.append(numbers)
            return (chart_type, tuple(parts))
        values = action_input.get("values")
        if not isinstance(values, list):
            return None
        flat: list[float] = []
        for v in values:
            if isinstance(v, (list, dict)):
                return None
            flat.append(float(v))
        return (chart_type, tuple(flat))
    except (TypeError, ValueError):
        return None


def _chart_observation(action_input: dict) -> str:
    """Short scratchpad line for a successful chart — never raw SVG.

    Raw SVG observations (multi-KB) flood the scratchpad/synthesis context
    and teach the model nothing; the chart itself travels via the SSE
    artifacts event. Trace 07fb4f59's r-final redrew the charts as ASCII
    blocks because all it could see was SVG soup.
    """
    title = str(action_input.get("title", "") or "").strip()
    labels = action_input.get("labels")
    if title:
        return f"chart generated: {title}"
    if isinstance(labels, list) and labels:
        return f"chart generated for labels {labels}"
    return "chart generated"


def _synthesis_evidence_line(result) -> str:
    """One evidence line for the final-synthesis prompt.

    Chart successes collapse to a one-liner (the SVG bytes already travel
    via Artifacts; pasting them here only invites ASCII redraws like trace
    07fb4f59's r-final). Everything else keeps its truncated text.
    """
    if result.agent_id == "plot.chart" or (result.output or "").lstrip().startswith(
        "<svg"
    ):
        return f"[{result.step_id} (plot.chart)] chart already generated and shown in Artifacts"
    return f"[{result.step_id} ({result.agent_id})]\n{(result.output or '')[:1500]}"


#: Request keywords that need breadth (stratified sample), not topical rank.
#: The ReAct model defaults rag.query to mode=specific; without this hint a
#: "summarize the docs" loop retrieves REFERENCES sections repeatedly
#: (trace cfbaa9c3) instead of overview content.
_OVERVIEW_HINTS = (
    "summar",
    "overview",
    "compare",
    "contrast",
    "quiz",
    "overall",
    "main topics",
    "key points",
)


def _default_react_mode(request_text: str, action_input: dict) -> dict:
    """Default broad asks to overview retrieval (stratified one-per-H1).

    Pure helper: returns a copy with mode=overview only when the model
    left it unset and the request needs breadth. Single-fact QA keeps
    the specific default. The input dict is never mutated.
    """
    if str(action_input.get("mode", "")).strip():
        return action_input
    lowered = (request_text or "").lower()
    if any(h in lowered for h in _OVERVIEW_HINTS):
        updated = dict(action_input)
        updated["mode"] = "overview"
        return updated
    return action_input


class ReactResult:
    def __init__(self, plan: Plan, result: ExecutionResult):
        self.plan = plan
        self.result = result


def _route_block(route_intent: str | None, corpus_state: str) -> str:
    """Prompt block stating what L1 already decided and what the corpus allows.

    The 3B model on the reported trace re-litigated "should I search the
    documents?" on every iteration and answered differently each time. The
    router's verdict is a fact by the time ReAct runs — state it, and state
    the one deterministic rule that follows from the corpus state.
    """
    lines: list[str] = []
    if route_intent and route_intent != "unknown":
        lines.append(
            f"Router intent: {route_intent} — serve THIS intent; do not "
            f"re-classify the request."
        )
    lines.append(f"Corpus: {_CORPUS_ROUTE_LINES.get(corpus_state, _CORPUS_ROUTE_LINES['unknown'])}")
    if route_intent in _NUMBERS_FIRST_INTENTS:
        lines.append(
            "No observation holds numbers yet unless the scratchpad shows some: "
            "your FIRST step must be a reasoning step recalling the figures, "
            "then plot.chart with those literal numbers."
        )
    if corpus_state in ("empty", "processing"):
        lines.append(
            "On an empty corpus a proposed rag.query is replaced by a "
            "general-knowledge reasoning step — do not retry rag.query."
        )
    return "Route:\n" + "\n".join(lines)


class ReActEngine:
    """Thought → action → observation loop with a single `run()` interface.

    Composition-time dependencies (provider, registries, trace id) are
    bound in the constructor; per-request values ride on `run()`. The
    idle-turn guard is an `IdleGuard` — each non-progress turn records
    one idle tick, each executed step resets it.
    """

    def __init__(
        self,
        provider: ModelProvider,
        agents: AgentRegistry,
        tools: ToolRegistry,
        trace_id: str,
    ):
        self._provider = provider
        self._agents = agents
        self._tools = tools
        self._trace_id = trace_id
        self._known_agents = {a["agent_id"] for a in agents.manifest()}
        self._known_tools = {t["tool_id"] for t in tools.manifest()}
        self._agent_ids = sorted(self._known_agents)
        self._tool_ids = sorted(self._known_tools)
        self._model = settings.ollama_default_model

    def run(
        self,
        request_text: str,
        *,
        corpus_id: str | None,
        context: str | None = None,
        corpus_context: str | None = None,
        route_intent: str | None = None,
        max_iterations: int = MAX_REACT_ITERATIONS,
        timeout_ms: int | None = None,
        cancel_event: threading.Event | None = None,
        on_event: Callable[[dict], None] | None = None,
        parent_span_ctx: dict[str, str] | None = None,
    ) -> ReactResult:
        """Run the thought → action → observation loop to answer request_text.

        `route_intent` is the L1 router's verdict (ADR-035) — it only sharpens
        the prompt; the loop still fails honest if the intent cannot be served.
        `parent_span_ctx` is the `react`-span context opened by the caller
        (orchestrator) — per-iteration spans parent explicitly under it so the
        trace reads `run → react → react:iter-N → step:rN` even across
        LangGraph pool-thread hops. None = parent to whatever is current
        (or no-op when tracing is off); existing callers are unaffected.
        """
        known_agents = self._known_agents
        known_tools = self._known_tools
        agent_ids = self._agent_ids
        tool_ids = self._tool_ids

        scratchpad: list[str] = []
        steps: list[PlanStep] = []
        step_results: list[StepResult] = []
        # Signatures of FAILED executions (retried the same broken action)
        # twice with the same docker error and rag.query twice with the same
        # empty result). A proposed repeat becomes an idle turn — no execution,
        # budget preserved for a different executor or a final answer.
        failed_actions: dict[str, str] = {}
        # Signatures of SUCCESSFUL executions (trace 07fb4f59 plotted the
        # identical Avg-Tokens chart twice, r2 then r3 verbatim). An exact
        # repeat becomes an idle turn — the chart/answer already exists.
        seen_actions: set[str] = set()
        # Normalized chart data already plotted (chart_type + numbers, ignoring
        # title/labels cosmetics — trace 07fb4f59 r5/r6 re-plotted [229, 135]
        # under different labels). Re-plotting the same data renders the same
        # bars, so the frontend would show the same plot twice.
        plotted_data: set[tuple] = set()
        # Corpus tri-state drives both the prompt and one deterministic
        # substitution below: on empty/processing, a proposed rag.query is
        # replaced by a general-knowledge reasoning step rather than refused
        # (a refusal burned an iteration and, charged to the idle budget,
        # could end the run with zero progress — trace c9b02eef).
        corpus_state = get_corpus_state(corpus_context)
        corpus_empty = corpus_state in ("empty", "processing")
        route_block = _route_block(route_intent, corpus_state)
        # Whether corpus.inspect has already run — the only case where
        # probing is worth an iteration is an unavailable inventory.
        inspected = False
        # Consecutive turns that produced no observation (provider errors,
        # unknown executors, empty answers). Caps garbage-loops against a
        # degraded model; any executed step or final answer resets it.
        guard = IdleGuard()

        # Explicit parent for `react:iter-N` spans: the `react`-span context
        # opened by the caller (orchestrator), or whatever is current when this
        # function is invoked directly (tests, ad-hoc). Plain strings — no
        # contextvars dependence across the thread hops below. No-op when
        # tracing is off.
        react_ctx = (
            parent_span_ctx if parent_span_ctx is not None else _get_trace_context()
        )
        final_answered = False
        for iteration in range(1, max_iterations + 1):
            with _manual_span(
                f"react:iter-{iteration}",
                as_type="span",
                input={
                    "iteration": iteration,
                    "request": _truncate(request_text, 500),
                },
                trace_context=react_ctx,
            ) as iter_obs:
                if cancel_event is not None and cancel_event.is_set():
                    iter_obs.update(output={"status": "cancelled"})
                    step_results.append(
                        StepResult(
                            step_id=f"r{iteration}",
                            agent_id="react",
                            status=StepStatus.FAILURE,
                            error="run cancelled",
                        )
                    )
                    break
                history = "\n".join(scratchpad) if scratchpad else "(no actions yet)"
                system_prompt = (
                    "You are a ReAct agent. Answer the user request one step at a time.\n"
                    f"Agents: {agent_ids}\nTools: {tool_ids}\n"
                    "Each turn return thought (what you learned / what remains), "
                    "executor (exactly one agent_id or tool_id for the NEXT single "
                    "step), input, is_final (true only when answering now), and "
                    "answer (the final answer when is_final).\n"
                    "Input shape (FLAT object, never nested under 'agent'):\n"
                    '- agent executor: {"message": "..."} with observations '
                    "inlined verbatim — never reference steps by number.\n"
                    "- tool executor: its FLAT schema fields, exactly as the "
                    "examples below show. A key the tool does not list is "
                    "rejected, not ignored.\n"
                    + _tool_examples(tool_ids, self._tools)
                    + "\n"
                    'WRONG: {"agent": {"message": "..."}} for a tool — '
                    "the tool reads top-level fields, so this fails with "
                    "'query'/'code' required. RIGHT: {\"query\": \"...\"}.\n"
                    "Prefer rag.query first when documents are available. "
                    "For summarize/compare/quiz or 'overall content' asks use "
                    "rag.query mode='overview' (stratified one-per-section "
                    "sample); single-fact QA keeps the specific default. "
                    "When the corpus has no documents and no observation "
                    "holds numbers, recall approximate figures with a reasoning "
                    "step first (state they are approximate), then plot.chart. "
                    + _plot_rules()
                    + " "
                    + route_block
                    + " "
                    + f"Corpus documents:\n{corpus_context or '(no documents)'}"
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Request: {request_text}\n\nScratchpad:\n{history}",
                    },
                ]
                if context:
                    messages.insert(
                        1,
                        {
                            "role": "system",
                            "content": f"Conversation context:\n{context}",
                        },
                    )
                try:
                    raw = self._provider.generate_structured(
                        model=self._model,
                        messages=messages,
                        schema=REACT_SCHEMA,
                        temperature=0,
                    )
                except Exception as e:  # noqa: BLE001 - failed iteration is an observation
                    if guard.record_idle():
                        # Provider itself is down — fail fast instead of burning the
                        # remaining iterations on identical errors.
                        iter_obs.update(
                            output={
                                "status": "failed",
                                "error": _truncate(
                                    f"react planner unavailable: {e}", 500
                                ),
                            }
                        )
                        step_results.append(
                            StepResult(
                                step_id=f"r{iteration}",
                                agent_id="react",
                                status=StepStatus.FAILURE,
                                error=f"react planner unavailable: {e}",
                            )
                        )
                        break
                    scratchpad.append(
                        f"planner error: {e}; propose a simpler next step."
                    )
                    iter_obs.update(
                        output={
                            "status": "retry",
                            "error": _truncate(str(e), 500),
                        }
                    )
                    continue
                executor = str(raw.get("executor", "")).strip()
                is_final = bool(raw.get("is_final", False))
                thought_in = _truncate(str(raw.get("thought", "")), 300)
                if is_final:
                    answer = raw.get("answer") or ""
                    if not str(answer).strip():
                        raw_input = raw.get("input", {}) or {}
                        answer = _fallback_answer_text(
                            dict(raw_input) if isinstance(raw_input, dict) else {}
                        )
                    if not str(answer).strip():
                        if guard.record_idle():
                            iter_obs.update(
                                output={
                                    "status": "failed",
                                    "error": "is_final with empty answer",
                                }
                            )
                            break
                        scratchpad.append(
                            "is_final was true but answer was empty. Either put the "
                            "final answer text in the 'answer' field with "
                            "is_final=true, or set is_final=false and execute a "
                            "tool step (e.g. doc.generate with title+sections)."
                        )
                        iter_obs.update(output={"status": "retry", "is_final": True})
                        continue
                    step_id = f"r{iteration}"
                    steps.append(
                        PlanStep(
                            step_id=step_id,
                            agent_id="reasoning",
                            input={"message": str(answer)},
                            expected_output_type="answer",
                        )
                    )
                    step_results.append(
                        StepResult(
                            step_id=step_id,
                            agent_id="reasoning",
                            status=StepStatus.SUCCESS,
                            output=str(answer),
                        )
                    )
                    final_answered = True
                    guard.record_progress()
                    iter_obs.update(
                        output={
                            "status": "success",
                            "thought": thought_in,
                            "executor": "reasoning",
                            "is_final": True,
                            "answer": _truncate(str(answer), 2000),
                        }
                    )
                    break
                if executor not in known_agents and executor not in known_tools:
                    if guard.record_idle():
                        iter_obs.update(
                            output={
                                "status": "failed",
                                "error": _truncate(
                                    f"unknown executor {executor!r}", 500
                                ),
                            }
                        )
                        break
                    scratchpad.append(
                        f"unknown executor {executor!r}; use one of {agent_ids + tool_ids}."
                    )
                    iter_obs.update(
                        output={
                            "status": "retry",
                            "thought": thought_in,
                            "executor": executor,
                        }
                    )
                    continue
                if (
                    not inspected
                    and corpus_state == "unknown"
                    and route_intent in _DOC_INTENT_VALUES
                    and executor != "corpus.inspect"
                    and not guard.record_blocked()
                ):
                    # The inventory failed to load, so the prompt cannot say
                    # what is uploaded — probing is the only way to learn it
                    # (ADR-027 freshness probe). Charged to the block budget,
                    # not the idle budget: the proposal itself was valid.
                    scratchpad.append(
                        "the file inventory is unavailable, so call "
                        "corpus.inspect {} first to see which documents are "
                        "uploaded and their status."
                    )
                    iter_obs.update(
                        output={
                            "status": "retry",
                            "thought": thought_in,
                            "executor": executor,
                            "error": "corpus.inspect required first",
                        }
                    )
                    continue
                raw_input = raw.get("input", {}) or {}
                action_input = _normalize_react_input(
                    executor, dict(raw_input) if isinstance(raw_input, dict) else {}
                )
                substituted_from: str | None = None
                if corpus_empty and executor == "rag.query":
                    # Deterministic substitution (ADR-035): retrieval on an
                    # empty corpus can only return "(no chunks retrieved)",
                    # so run the general-knowledge step the model actually
                    # needs instead of refusing. It executes, so the
                    # observation reaches the scratchpad and a repeat hits
                    # the seen/failed guards rather than looping forever.
                    # Runs BEFORE input validation on purpose: a malformed
                    # rag.query on an empty corpus would otherwise burn an
                    # idle turn correcting a call that was never going to run.
                    substituted_from = executor
                    executor = "reasoning"
                    action_input = {
                        "message": (
                            "This corpus has no ready documents, so no "
                            "retrieval was run. Answer the request from your "
                            "own knowledge (state clearly when a figure is "
                            f"approximate). Request: {request_text}"
                        )
                    }
                    scratchpad.append(
                        "note: rag.query was skipped — this corpus has no "
                        "ready documents; the step below answers from general "
                        "knowledge instead."
                    )
                elif executor == "rag.query":
                    action_input = _default_react_mode(request_text, action_input)
                if executor in known_tools:
                    hint = _validate_react_input(executor, action_input, self._tools)
                    if hint is not None:
                        if guard.record_idle():
                            iter_obs.update(
                                output={
                                    "status": "failed",
                                    "error": _truncate(hint, 500),
                                }
                            )
                            break
                        scratchpad.append(f"{hint}; retry with corrected flat input.")
                        iter_obs.update(
                            output={
                                "status": "retry",
                                "thought": thought_in,
                                "executor": executor,
                                "error": _truncate(hint, 500),
                            }
                        )
                        continue
                if (
                    executor in known_agents
                    and not str(action_input.get("message", "")).strip()
                ):
                    if guard.record_idle():
                        iter_obs.update(
                            output={
                                "status": "failed",
                                "error": f"agent {executor} missing input.message",
                            }
                        )
                        break
                    scratchpad.append(
                        f"agent {executor} needs input.message; retry with it."
                    )
                    iter_obs.update(
                        output={
                            "status": "retry",
                            "thought": thought_in,
                            "executor": executor,
                        }
                    )
                    continue
                sig = _action_signature(executor, action_input)
                if sig in failed_actions:
                    if guard.record_idle():
                        iter_obs.update(
                            output={
                                "status": "failed",
                                "error": _truncate(
                                    f"{executor} already failed: {failed_actions[sig]}",
                                    500,
                                ),
                            }
                        )
                        break
                    scratchpad.append(
                        f"{executor} already failed ({failed_actions[sig]}); pick "
                        "a different executor or answer from what you have."
                    )
                    iter_obs.update(
                        output={
                            "status": "retry",
                            "thought": thought_in,
                            "executor": executor,
                            "error": _truncate(f"repeat of failed {executor}", 500),
                        }
                    )
                    continue
                if sig in seen_actions:
                    if guard.record_idle():
                        iter_obs.update(
                            output={
                                "status": "failed",
                                "error": _truncate(
                                    f"{executor} already did this exact step", 500
                                ),
                            }
                        )
                        break
                    scratchpad.append(
                        f"{executor} with these exact inputs already succeeded; "
                        "do something different (a new metric, or answer from "
                        "what you have)."
                    )
                    iter_obs.update(
                        output={
                            "status": "retry",
                            "thought": thought_in,
                            "executor": executor,
                            "error": _truncate(f"repeat of successful {executor}", 500),
                        }
                    )
                    continue
                if executor == "plot.chart":
                    data_key = _plot_data_key(action_input)
                    if data_key is not None and data_key in plotted_data:
                        if guard.record_idle():
                            iter_obs.update(
                                output={
                                    "status": "failed",
                                    "error": "chart data already plotted",
                                }
                            )
                            break
                        scratchpad.append(
                            "these numbers are already plotted in an earlier "
                            "chart; plot a DIFFERENT metric or answer from what "
                            "you have — never re-plot the same data."
                        )
                        iter_obs.update(
                            output={
                                "status": "retry",
                                "thought": thought_in,
                                "executor": executor,
                                "error": "chart data already plotted",
                            }
                        )
                        continue
                guard.record_progress()
                step_id = f"r{iteration}"
                is_tool = executor in known_tools
                mini = Plan(
                    plan_id=f"react-{iteration}",
                    goal=request_text,
                    steps=[
                        PlanStep(
                            step_id=step_id,
                            **(
                                {"tool_id": executor}
                                if is_tool
                                else {"agent_id": executor}
                            ),
                            input=action_input,
                            expected_output_type=_output_type(executor, False),
                        )
                    ],
                )
                # Step spans parent explicitly under THIS iteration span so the
                # trace reads react → react:iter-N → step:rN even though the
                # inner plan graph schedules nodes on pool threads.
                step_parent = _get_trace_context()
                try:
                    exec_result = run_plan_graph(
                        mini,
                        self._agents,
                        tool_registry=self._tools,
                        trace_id=self._trace_id,
                        corpus_id=corpus_id,
                        context=None,
                        fallback_message=request_text,
                        timeout_ms=timeout_ms or settings.default_timeout_ms,
                        on_event=on_event,
                        cancel_event=cancel_event,
                        parent_span_ctx=step_parent,
                    )
                except Exception as e:  # noqa: BLE001 - execution error is an observation
                    scratchpad.append(
                        f"step {step_id} ({executor}) raised {e}; try another."
                    )
                    iter_obs.update(
                        output={
                            "status": "retry",
                            "thought": thought_in,
                            "executor": executor,
                            "error": _truncate(str(e), 500),
                        }
                    )
                    continue
                outcome = (
                    exec_result.step_results[0] if exec_result.step_results else None
                )
                if outcome is None:
                    scratchpad.append(f"step {step_id} produced nothing; try another.")
                    iter_obs.update(
                        output={
                            "status": "retry",
                            "thought": thought_in,
                            "executor": executor,
                        }
                    )
                    continue
                steps.append(mini.steps[0])
                step_results.append(outcome)
                guard.record_progress()  # an executed step is progress, even on tool failure
                seen_actions.add(sig)
                if executor == "corpus.inspect" and outcome.status is StepStatus.SUCCESS:
                    inspected = True
                thought = str(raw.get("thought", ""))[:300]
                if outcome.status is StepStatus.SUCCESS:
                    if is_tool and executor == "plot.chart":
                        data_key = _plot_data_key(action_input)
                        if data_key is not None:
                            plotted_data.add(data_key)
                        observation = _chart_observation(action_input)
                    else:
                        observation = (outcome.output or "")[:1500]
                    scratchpad.append(
                        f"step {step_id} ({executor}) thought: {thought} "
                        f"observation: {observation}"
                    )
                    iter_obs.update(
                        output={
                            "status": "success",
                            "thought": thought_in,
                            "executor": executor,
                            "substituted_from": substituted_from,
                            "observation": _truncate(observation, 2000),
                        }
                    )
                else:
                    failed_actions[sig] = outcome.error or "unknown error"
                    scratchpad.append(
                        f"step {step_id} ({executor}) failed: {outcome.error}; try another."
                    )
                    iter_obs.update(
                        output={
                            "status": outcome.status.value,
                            "thought": thought_in,
                            "executor": executor,
                            "substituted_from": substituted_from,
                            "error": _truncate(outcome.error, 500),
                        }
                    )
        if not final_answered:
            synth = self._synthesize_final_answer(
                request_text,
                step_results,
                context=context,
                corpus_id=corpus_id,
                timeout_ms=timeout_ms,
                cancel_event=cancel_event,
                on_event=on_event,
            )
            if synth is not None:
                synth_step, synth_outcome = synth
                steps.append(synth_step)
                step_results.append(synth_outcome)
                final_answered = True
        if not final_answered and not any(
            r.status is StepStatus.SUCCESS for r in step_results
        ):
            # Last resort (ADR-035): the loop executed nothing that worked, so
            # answer the request verbatim with one reasoning step — the same
            # shape as the `chat`/`knowledge_qa` builders. Trace c9b02eef
            # ended a well-formed request as a run failure carrying the
            # routing error "no deterministic builder for intent
            # plot_standalone"; the routing was right and the user deserves
            # an answer, not a taxonomy message.
            last = self._last_resort_answer(
                request_text,
                context=context,
                corpus_id=corpus_id,
                timeout_ms=timeout_ms,
                cancel_event=cancel_event,
                on_event=on_event,
            )
            if last is not None:
                steps.append(last[0])
                step_results.append(last[1])
                final_answered = True
        if route_intent in _NUMBERS_FIRST_INTENTS and not any(
            r.agent_id == "plot.chart" and r.status is StepStatus.SUCCESS
            for r in step_results
        ):
            # The user asked for a chart and got prose. Report it: a plot
            # intent whose malformed plot.chart attempts were all dropped
            # as idle turns leaves no step result, so without this the run
            # reports plain success and the missing artifact is invisible
            # (trace c9b02eef: "success" with an ASCII-art redraw instead of
            # a chart). No plan step — nothing executed.
            step_results.append(
                StepResult(
                    step_id="r-chart",
                    agent_id="plot.chart",
                    status=StepStatus.FAILURE,
                    error=(
                        "no chart was generated for this plot request "
                        "(every plot.chart proposal was rejected before "
                        "execution)"
                    ),
                )
            )
        plan = Plan(
            plan_id=str(uuid.uuid4()),
            goal=request_text,
            steps=steps
            or [
                PlanStep(
                    step_id="r0",
                    agent_id="reasoning",
                    input={"message": request_text},
                    expected_output_type="text",
                )
            ],
        )
        if not step_results:
            step_results = [
                StepResult(
                    step_id="r0",
                    agent_id="react",
                    status=StepStatus.FAILURE,
                    error="react loop produced no steps",
                )
            ]
        return ReactResult(
            plan, ExecutionResult(trace_id=self._trace_id, step_results=step_results)
        )

    def _synthesize_final_answer(
        self,
        request_text: str,
        step_results: list[StepResult],
        *,
        context: str | None,
        corpus_id: str | None,
        timeout_ms: int | None,
        cancel_event: threading.Event | None,
        on_event: Callable[[dict], None] | None,
    ) -> tuple[PlanStep, StepResult] | None:
        """Grounded final answer when the loop exhausts iterations without is_final.

        (Trace cfbaa9c3: six rag.query observations, final summary was a
        truncated raw chunk dump via the aggregator anti-blank fallback.)
        Synthesizes one grounded answer from the successful observations so
        the user gets prose covering every retrieved document instead of
        raw chunks. Returns None when there is nothing to synthesize from.
        """
        successes = [
            r
            for r in step_results
            if r.status is StepStatus.SUCCESS and (r.output or "").strip()
        ]
        if successes and not (cancel_event is not None and cancel_event.is_set()):
            synth_id = (
                "reasoning"
                if "reasoning" in self._known_agents
                else (self._agent_ids[0] if self._agent_ids else "")
            )
            if synth_id:
                evidence = "\n\n".join(
                    _synthesis_evidence_line(r) for r in successes[-4:]
                )
                synth_message = (
                    f"Synthesize the final answer to the request using ONLY "
                    f"these observations. Cover every document below; do not "
                    f"ask the user to upload or paste anything. Charts are "
                    f"already rendered in Artifacts — describe each chart's "
                    f"takeaway and give a summary table, but NEVER redraw "
                    f"charts as ASCII/text blocks. "
                    f"Request: {request_text}\n\nObservations:\n{evidence}"
                )
                try:
                    synth_plan = Plan(
                        plan_id=f"react-final-{uuid.uuid4().hex[:8]}",
                        goal=request_text,
                        steps=[
                            PlanStep(
                                step_id="r-final",
                                agent_id=synth_id,
                                input={"message": synth_message},
                                expected_output_type="answer",
                            )
                        ],
                    )
                    synth_result = run_plan_graph(
                        synth_plan,
                        self._agents,
                        tool_registry=self._tools,
                        trace_id=self._trace_id,
                        corpus_id=corpus_id,
                        context=context,
                        fallback_message=request_text,
                        timeout_ms=timeout_ms or settings.default_timeout_ms,
                        on_event=on_event,
                        cancel_event=cancel_event,
                        parent_span_ctx=_get_trace_context(),
                    )
                    synth_outcome = (
                        synth_result.step_results[0]
                        if synth_result.step_results
                        else None
                    )
                    if (
                        synth_outcome is not None
                        and synth_outcome.status is StepStatus.SUCCESS
                    ):
                        return synth_plan.steps[0], synth_outcome
                except Exception as e:  # noqa: BLE001 - synthesis miss keeps raw steps
                    logger.warning("react final synthesis failed: %s", e)
        return None

    def _last_resort_answer(
        self,
        request_text: str,
        *,
        context: str | None,
        corpus_id: str | None,
        timeout_ms: int | None,
        cancel_event: threading.Event | None,
        on_event: Callable[[dict], None] | None,
    ) -> tuple[PlanStep, StepResult] | None:
        """One ungrounded reasoning answer when the loop produced nothing.

        Deliberately NOT document-grounded: there are no observations to
        ground on, and the alternative (the aggregator's all-failed path)
        shows the user the taxonomy's internal error text. Carries the
        conversation context so the answer can still use prior turns.
        Returns None on cancel or provider failure — the caller then fails
        honest.
        """
        if cancel_event is not None and cancel_event.is_set():
            return None
        agent_id = (
            "reasoning"
            if "reasoning" in self._known_agents
            else (self._agent_ids[0] if self._agent_ids else "")
        )
        if not agent_id:
            return None
        plan = Plan(
            plan_id=f"react-lastresort-{uuid.uuid4().hex[:8]}",
            goal=request_text,
            steps=[
                PlanStep(
                    step_id="r-lastresort",
                    agent_id=agent_id,
                    input={"message": request_text},
                    expected_output_type="answer",
                )
            ],
        )
        try:
            result = run_plan_graph(
                plan,
                self._agents,
                tool_registry=self._tools,
                trace_id=self._trace_id,
                corpus_id=corpus_id,
                context=context,
                fallback_message=request_text,
                timeout_ms=timeout_ms or settings.default_timeout_ms,
                on_event=on_event,
                cancel_event=cancel_event,
                parent_span_ctx=_get_trace_context(),
            )
        except Exception as e:  # noqa: BLE001 - honest failure below
            logger.warning("react last-resort answer failed: %s", e)
            return None
        outcome = result.step_results[0] if result.step_results else None
        if outcome is None or outcome.status is not StepStatus.SUCCESS:
            return None
        return plan.steps[0], outcome


def run_react(
    request_text: str,
    provider: ModelProvider,
    agents: AgentRegistry,
    tools: ToolRegistry,
    *,
    trace_id: str,
    corpus_id: str | None,
    context: str | None = None,
    corpus_context: str | None = None,
    route_intent: str | None = None,
    max_iterations: int = MAX_REACT_ITERATIONS,
    timeout_ms: int | None = None,
    cancel_event: threading.Event | None = None,
    on_event: Callable[[dict], None] | None = None,
    parent_span_ctx: dict[str, str] | None = None,
) -> ReactResult:
    """Run the thought → action → observation loop to answer request_text.

    Thin wrapper over `ReActEngine` — preserved so existing callers
    (orchestrator, tests) are unaffected. `parent_span_ctx` is the
    `react`-span context opened by the caller (orchestrator);
    `route_intent` is the L1 router's verdict (ADR-035).
    """
    engine = ReActEngine(provider, agents, tools, trace_id)
    return engine.run(
        request_text,
        corpus_id=corpus_id,
        context=context,
        corpus_context=corpus_context,
        route_intent=route_intent,
        max_iterations=max_iterations,
        timeout_ms=timeout_ms,
        cancel_event=cancel_event,
        on_event=on_event,
        parent_span_ctx=parent_span_ctx,
    )
