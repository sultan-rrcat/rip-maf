"""Layered planning chain tests: L2 builder hit, L3 ReAct fallback.

L1 router (sole dispatcher, every request via LLM) → L2 deterministic
builders → L3 ReAct. Uses the real default registries (reasoning agent +
rag.query tool) with a fake provider/RAG — no Ollama, no DB.
"""

from __future__ import annotations

import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from rip_maf.agents.registry import get_default_agent_registry
from rip_maf.orchestration.aggregator import Aggregator
from rip_maf.orchestration.orchestrator import Orchestrator
from rip_maf.orchestration.plan import Plan, PlanStep
from rip_maf.orchestration.planner import Planner
from rip_maf.orchestration.validator import PlanValidationError, PlanValidator
from rip_maf.providers.base import ModelProvider
from rip_maf.tools.registry import get_default_tool_registry


class FakeRAG:
    def retrieve_context(
        self, corpus_id, query, top_k=4, file_id=None, file_name=None, mode="specific"
    ):
        return {
            "query": query,
            "mode": mode,
            "results": [
                {
                    "content": f"chunk for {query}",
                    "source": "f.pdf",
                    "section": "H1",
                    "rerank_score": 0.9,
                }
            ],
        }


class FakeLayeredProvider(ModelProvider):
    def __init__(self, text="layered answer", queued=None):
        self.text = text
        self.queued = list(queued) if queued else []
        self.models: list[str] = []
        self.prompts: list = []
        self.structured_calls = 0

    def generate(self, model, messages, *, temperature=0.2, max_tokens=None):
        self.models.append(model)
        return self.text

    def generate_stream(self, model, messages, *, temperature=0.2, max_tokens=None):
        self.models.append(model)
        yield self.text

    def generate_structured(self, model, messages, schema, *, temperature=0.0):
        self.models.append(model)
        self.prompts.append(messages)
        self.structured_calls += 1
        return dict(self.queued.pop(0))

    def embed(self, model: str, text: str) -> list[float]:
        raise NotImplementedError("test fake")

    def list_available_models(self) -> list[dict]:
        return [{"id": "fake"}]


def _orchestrator(provider):
    agents = get_default_agent_registry(provider)
    tools = get_default_tool_registry(rag=FakeRAG())
    return Orchestrator(
        Planner(provider, agents, tools),
        PlanValidator(agents, tools),
        Aggregator(), agents, tools,
    )


COMPARE_REQUEST = "compare both report and rank them based on complexity"


def test_compare_request_uses_builder_plan_without_mega_call() -> None:
    provider = FakeLayeredProvider(queued=[
        {
            "intent": "compare_multi",
            "queries": ["fire report sections", "faultbook sections"],
            "confidence": 0.9,
        },
    ])
    orch = _orchestrator(provider)
    result = orch.run(COMPARE_REQUEST, "nb-layered")
    assert result.status == "success"
    assert result.summary == "layered answer"
    assert len(result.step_results) == 3  # 2×rag.query + reasoning fan-in
    assert result.goal == COMPARE_REQUEST
    # Only the router structured call ran — the mega-prompt planner was
    # never asked (its queued payload would raise IndexError if consumed).
    assert provider.structured_calls == 1
    assert "intent router" in provider.prompts[0][0]["content"]
    assert result.shown == ["3"] and result.hidden == ["1", "2"]


def test_unknown_intent_falls_back_to_react() -> None:
    provider = FakeLayeredProvider(queued=[
        {"intent": "unknown", "queries": [], "confidence": 0.0},
        {"thought": "answer directly", "executor": "reasoning",
         "input": {}, "is_final": True, "answer": "react answer"},
    ])
    result = _orchestrator(provider).run(
        "a vague long request with no clear shape at all here", "nb-1"
    )
    assert result.status == "success" and result.summary == "react answer"
    assert provider.structured_calls == 2  # router + react turn


def test_greeting_goes_through_router_to_builder() -> None:
    provider = FakeLayeredProvider(queued=[
        {"intent": "chat", "queries": [], "confidence": 0.95},
    ])
    result = _orchestrator(provider).run("hi", "nb-1")
    assert result.status == "success" and result.summary == "layered answer"
    assert len(result.step_results) == 1
    assert provider.structured_calls == 1  # L1 router, then L2 builder (no LLM)


def _prose_validator():
    provider = FakeLayeredProvider()
    agents = get_default_agent_registry(provider)
    return PlanValidator(agents, get_default_tool_registry(rag=FakeRAG()))


def _rag_step(step_id: str) -> PlanStep:
    return PlanStep(
        step_id=step_id, tool_id="rag.query",
        input={"query": "x"}, expected_output_type="chunks",
    )


def test_prose_step_ref_without_placeholder_rejected() -> None:
    # Exact ecd93eb4 shape: prose names steps 1/2, no {{1}} {{2}}, no edges.
    import pytest

    plan = Plan(
        plan_id="p", goal="compare both reports",
        steps=[
            _rag_step("1"),
            _rag_step("2"),
            PlanStep(
                step_id="3", agent_id="reasoning",
                input={"message": (
                    "Using the retrieved chunks from step 1 (Fire report) "
                    "and step 2 (Faultbook report), compare and rank them."
                )},
                expected_output_type="answer",
            ),
        ],
    )
    with pytest.raises(PlanValidationError, match="placeholder"):
        _prose_validator().validate(plan)


def test_prose_step_ref_with_placeholders_passes() -> None:
    plan = Plan(
        plan_id="p", goal="compare both reports",
        steps=[
            _rag_step("1"),
            _rag_step("2"),
            PlanStep(
                step_id="3", agent_id="reasoning",
                input={"message": "Using {{1}} and {{2}}, compare step outputs."},
                depends_on=["1", "2"], expected_output_type="answer",
            ),
        ],
    )
    assert _prose_validator().validate(plan) is plan


def test_benign_step_prose_without_rag_siblings_passes() -> None:
    plan = Plan(
        plan_id="p", goal="g",
        steps=[
            PlanStep(
                step_id="1", agent_id="reasoning",
                input={"message": "follow step 1 of the checklist below"},
                expected_output_type="text",
            )
        ],
    )
    assert _prose_validator().validate(plan) is plan


def _react_orchestrator(provider):
    agents = get_default_agent_registry(provider)
    tools = get_default_tool_registry(rag=FakeRAG())
    return agents, tools


# --- ADR-035: route-aware ReAct -----------------------------------------


def _probe_prompts():
    """Run one final-answer turn and return the prompts the model saw."""
    from rip_maf.orchestration.react import run_react

    seen: list = []

    class _ProbeProvider(FakeLayeredProvider):
        def generate_structured(self, model, messages, schema, *, temperature=0.0):
            seen.append(messages)
            return {"thought": "done", "executor": "reasoning",
                    "input": {}, "is_final": True, "answer": "ok"}

    provider = _ProbeProvider()
    agents, tools = _react_orchestrator(provider)
    run_react("plot this", provider, agents, tools,
              trace_id="t", corpus_id="nb-1", corpus_context="(no documents)",
              route_intent="plot_standalone")
    return [m[0]["content"] for m in seen]


def test_react_prompt_carries_router_intent_and_corpus_state() -> None:
    # Trace c9b02eef: the loop re-litigated "should I search the documents?"
    # every iteration because it was never told what L1 decided.
    system = _probe_prompts()[0]
    assert "Router intent: plot_standalone" in system
    assert "corpus is EMPTY" in system
    assert "do not re-classify" in system


def test_react_plot_intent_demands_numbers_first() -> None:
    system = _probe_prompts()[0]
    assert "your FIRST step must be a reasoning step recalling the figures" in system
    assert "do not retry rag.query" in system


def test_react_route_block_absent_without_a_route() -> None:
    from rip_maf.orchestration.react import run_react

    seen: list = []

    class _ProbeProvider(FakeLayeredProvider):
        def generate_structured(self, model, messages, schema, *, temperature=0.0):
            seen.append(messages)
            return {"thought": "done", "executor": "reasoning",
                    "input": {}, "is_final": True, "answer": "ok"}

    provider = _ProbeProvider()
    agents, tools = _react_orchestrator(provider)
    run_react("q", provider, agents, tools, trace_id="t", corpus_id="nb-1")
    system = seen[0][0]["content"]
    assert "Router intent:" not in system
    # The corpus line is still stated — it is what the steering rules key on.
    assert "Corpus:" in system


def test_react_ready_corpus_keeps_rag_query_legal() -> None:
    from rip_maf.orchestration.react import run_react

    seen: list = []

    class _ProbeProvider(FakeLayeredProvider):
        def generate_structured(self, model, messages, schema, *, temperature=0.0):
            seen.append(messages)
            return {"thought": "done", "executor": "reasoning",
                    "input": {}, "is_final": True, "answer": "ok"}

    provider = _ProbeProvider()
    agents, tools = _react_orchestrator(provider)
    run_react("summarize", provider, agents, tools, trace_id="t",
              corpus_id="nb-1",
              corpus_context="1 file(s): a.pdf [ready] id=aaa",
              route_intent="summarize")
    system = seen[0][0]["content"]
    assert "HAS ready documents" in system
    assert "never run" not in system


def test_react_requires_inspect_only_when_inventory_unavailable() -> None:
    # The targeted version of "inspect first": probing pays off only when
    # the prompt genuinely cannot say what is uploaded.
    from rip_maf.orchestration.react import run_react

    provider = FakeLayeredProvider(queued=[
        {"thought": "look around", "executor": "reasoning",
         "input": {"message": "guess"}, "is_final": False},
        {"thought": "listing", "executor": "corpus.inspect",
         "input": {}, "is_final": False},
        {"thought": "done", "executor": "reasoning", "input": {},
         "is_final": True, "answer": "listed"},
    ])
    agents, tools = _react_orchestrator(provider)
    outcome = run_react(
        "summarize the docs", provider, agents, tools,
        trace_id="t", corpus_id="nb-1",
        corpus_context=None,  # inventory unavailable
        route_intent="summarize",
    )
    # r1 was turned away with a hint (no step), r2 inspected, r3 answered.
    assert [s.step_id for s in outcome.plan.steps] == ["r2", "r3"]
    assert outcome.plan.steps[0].tool_id == "corpus.inspect"
    assert "file inventory is unavailable" in provider.prompts[1][-1]["content"]


def test_react_does_not_force_inspect_when_snapshot_is_known() -> None:
    from rip_maf.orchestration.react import run_react

    provider = FakeLayeredProvider(queued=[
        {"thought": "answer", "executor": "reasoning", "input": {},
         "is_final": True, "answer": "direct"},
    ])
    agents, tools = _react_orchestrator(provider)
    outcome = run_react(
        "summarize the docs", provider, agents, tools,
        trace_id="t", corpus_id="nb-1",
        corpus_context="1 file(s): a.pdf [ready] id=aaa",
        route_intent="summarize",
    )
    assert [s.step_id for s in outcome.plan.steps] == ["r1"]
    assert provider.structured_calls == 1


def test_react_last_resort_answers_when_nothing_executed() -> None:
    # A request ReAct cannot act on still gets an answer instead of the
    # routing error ("no deterministic builder for intent ...").
    from rip_maf.orchestration.react import run_react

    provider = FakeLayeredProvider(queued=[
        {"thought": "bad pick", "executor": "ghost", "input": {}, "is_final": False},
        {"thought": "bad pick again", "executor": "ghost", "input": {}, "is_final": False},
    ])
    agents, tools = _react_orchestrator(provider)
    outcome = run_react(
        "explain gradient descent", provider, agents, tools,
        trace_id="t", corpus_id="nb-1",
        corpus_context="(no documents)",
        route_intent="knowledge_qa",
    )
    assert provider.structured_calls == 2  # loop still fails fast
    assert outcome.result.step_results[-1].status.value == "success"
    assert outcome.result.step_results[-1].output == "layered answer"


def test_plot_intent_without_a_chart_is_reported() -> None:
    # Trace c9b02eef: the user asked for a chart, every plot.chart proposal
    # was rejected before execution (idle turns, so no step result), and the
    # run still reported plain success. The missing artifact must be visible.
    from rip_maf.orchestration.react import run_react

    provider = FakeLayeredProvider(queued=[
        {"thought": "bad pick", "executor": "ghost", "input": {}, "is_final": False},
        {"thought": "bad pick again", "executor": "ghost", "input": {}, "is_final": False},
    ])
    agents, tools = _react_orchestrator(provider)
    outcome = run_react(
        "plot india vs china gdp", provider, agents, tools,
        trace_id="t", corpus_id="nb-1",
        corpus_context="(no documents)",
        route_intent="plot_standalone",
    )
    missing = [r for r in outcome.result.step_results if r.step_id == "r-chart"]
    assert missing and missing[0].status.value == "failure"
    assert "no chart was generated" in (missing[0].error or "")
    # No plan step is invented for something that never executed.
    assert "r-chart" not in [s.step_id for s in outcome.plan.steps]


def test_plot_intent_with_a_chart_reports_no_chart_failure() -> None:
    from rip_maf.orchestration.react import run_react

    provider = FakeLayeredProvider(queued=[
        {"thought": "draw it", "executor": "plot.chart",
         "input": {"chart_type": "line", "labels": ["2015", "2016"],
                   "values": [7.2, 7.1], "title": "GDP growth %"},
         "is_final": False},
        {"thought": "done", "executor": "reasoning", "input": {},
         "is_final": True, "answer": "charted"},
    ])
    agents, tools = _react_orchestrator(provider)
    outcome = run_react(
        "plot india gdp", provider, agents, tools,
        trace_id="t", corpus_id="nb-1",
        corpus_context="(no documents)",
        route_intent="plot_standalone",
    )
    assert not [r for r in outcome.result.step_results if r.step_id == "r-chart"]


def test_react_last_resort_not_used_after_a_real_step() -> None:
    # A loop that executed something keeps its own steps — the last resort
    # is for zero-progress runs only.
    from rip_maf.orchestration.react import run_react

    provider = FakeLayeredProvider(queued=[
        {"thought": "answer", "executor": "reasoning", "input": {},
         "is_final": True, "answer": "real answer"},
    ])
    agents, tools = _react_orchestrator(provider)
    outcome = run_react(
        "q", provider, agents, tools,
        trace_id="t", corpus_id="nb-1",
    )
    assert [s.step_id for s in outcome.plan.steps] == ["r1"]


def test_react_plot_invented_keys_are_rejected_with_the_fix() -> None:
    # Trace c9b02eef iters 2-3: the model expressed "India vs China" as
    # series_labels + a parallel china_values array, twice. The rule and the
    # fix both live in PlotChartTool now (they used to be copied into
    # react_engine.py four times and drift).
    from rip_maf.tools.plot_chart import PlotChartTool

    hint = PlotChartTool().explain_invalid({
        "chart_type": "line",
        "labels": ["2015", "2016", "2017"],
        "values": [7.2, 7.1, 7.0],
        "series_labels": ["India"],
        "china_values": [6.9, 6.5, 6.0],
        "title": "GDP growth %: India vs China",
    })
    assert hint is not None
    assert "series_labels" in hint and "china_values" in hint
    assert "'series'" in hint
    # Copyable example — prose rules were not enough for the 3B model.
    assert '{"label": "India", "values": [7.2, 7.1, 7.0]}' in hint
    assert "ONE 'series' array over shared" in hint


def test_react_plot_parallel_array_alone_is_rejected() -> None:
    # Without the closed key set this would have PASSED as a single-series
    # call and silently charted India only — a dropped series, which is the
    # ADR-027 hallucination class.
    from rip_maf.tools.plot_chart import PlotChartTool

    hint = PlotChartTool().explain_invalid({
        "chart_type": "line",
        "labels": ["2015", "2016"],
        "values": [7.2, 7.1],
        "china_values": [6.9, 6.5],
    })
    assert hint is not None and "china_values" in hint


def test_react_plot_correct_shapes_still_pass() -> None:
    from rip_maf.tools.plot_chart import PlotChartTool

    tool = PlotChartTool()
    assert tool.explain_invalid({
        "chart_type": "line", "labels": ["2015", "2016"],
        "values": [7.2, 7.1], "title": "GDP growth %",
    }) is None
    assert tool.explain_invalid({
        "chart_type": "line", "labels": ["2015", "2016"],
        "series": [
            {"label": "India", "values": [7.2, 7.1]},
            {"label": "China", "values": [6.9, 6.5]},
        ],
        "title": "GDP growth %: India vs China",
    }) is None


def test_react_preflight_uses_the_tool_contract_not_a_copy() -> None:
    # The ReAct pre-flight must be the tool's own verdict, reachable through
    # the registry — that is the whole point of Tool.validate_input.
    from rip_maf.orchestration.react import _validate_react_input
    from rip_maf.tools.plot_chart import PlotChartTool
    from rip_maf.tools.registry import get_default_tool_registry

    tools = get_default_tool_registry()
    bad = {"chart_type": "line", "labels": ["a"], "values": [1],
           "series_labels": ["x"]}
    assert _validate_react_input("plot.chart", bad, tools) == (
        PlotChartTool().explain_invalid(bad)
    )
    good = {"chart_type": "line", "labels": ["a"], "values": [1], "title": "t"}
    assert _validate_react_input("plot.chart", good, tools) is None
    # Agents are not in the tool registry; their message check is elsewhere.
    assert _validate_react_input("reasoning", {}, tools) is None


def test_react_prompt_shows_the_multi_series_shape() -> None:
    from rip_maf.orchestration.react import run_react

    seen: list = []

    class _ProbeProvider(FakeLayeredProvider):
        def generate_structured(self, model, messages, schema, *, temperature=0.0):
            seen.append(messages)
            return {"thought": "done", "executor": "reasoning",
                    "input": {}, "is_final": True, "answer": "ok"}

    provider = _ProbeProvider()
    agents, tools = _react_orchestrator(provider)
    run_react("plot this", provider, agents, tools,
              trace_id="t", corpus_id="nb-1")
    system = seen[0][0]["content"]
    assert '{"label": "China", "values": [6.9, 6.5, 6.0]}' in system
    assert "no 'series_labels' key" in system


def test_react_intermediate_reasoning_is_hidden_from_the_answer() -> None:
    # A non-final agent step is scratchpad. Shown as "text" it duplicated
    # the final synthesis and led the summary with an ASCII-art redraw.
    from rip_maf.orchestration.react import run_react

    provider = FakeLayeredProvider(queued=[
        {"thought": "recall figures", "executor": "reasoning",
         "input": {"message": "numbers please"}, "is_final": False},
        {"thought": "done", "executor": "reasoning", "input": {},
         "is_final": True, "answer": "final answer"},
    ])
    agents, tools = _react_orchestrator(provider)
    outcome = run_react("q", provider, agents, tools,
                        trace_id="t", corpus_id="nb-1")
    steps = {s.step_id: s for s in outcome.plan.steps}
    assert steps["r1"].expected_output_type == "observation"
    assert steps["r2"].expected_output_type == "answer"


def test_react_answers_after_tool_observation() -> None:
    from rip_maf.orchestration.react import run_react

    provider = FakeLayeredProvider(queued=[
        {"thought": "need docs", "executor": "rag.query",
         "input": {"query": "fire"}, "is_final": False},
        {"thought": "have chunks", "executor": "reasoning",
         "input": {}, "is_final": True, "answer": "react final"},
    ])
    agents, tools = _react_orchestrator(provider)
    outcome = run_react(
        "what do docs say?", provider, agents, tools,
        trace_id="t", corpus_id="nb-1",
    )
    assert outcome.result.step_results[-1].status.value == "success"
    assert outcome.result.step_results[-1].output == "react final"
    assert provider.structured_calls == 2
    # No placeholder wiring is ever emitted by the loop.
    assert "{{" not in str([s.input for s in outcome.plan.steps])


def test_react_rejects_unknown_executor_then_recovers() -> None:
    from rip_maf.orchestration.react import run_react

    provider = FakeLayeredProvider(queued=[
        {"thought": "bad pick", "executor": "ghost",
         "input": {}, "is_final": False},
        {"thought": "answer directly", "executor": "reasoning",
         "input": {}, "is_final": True, "answer": "recovered"},
    ])
    agents, tools = _react_orchestrator(provider)
    outcome = run_react(
        "answer this", provider, agents, tools,
        trace_id="t", corpus_id="nb-1",
    )
    assert outcome.result.step_results[-1].output == "recovered"
    assert provider.structured_calls == 2


def test_react_normalizes_nested_agent_input() -> None:
    from rip_maf.orchestration.react import (
        _normalize_react_input,
        _validate_react_input,
    )
    from rip_maf.tools.registry import get_default_tool_registry

    registry = get_default_tool_registry()

    # Trace c9e59039 iter-1 shape: tool input wrapped under "agent".
    out = _normalize_react_input(
        "rag.query", {"agent": {"message": "fire distribution"}}
    )
    assert out["query"] == "fire distribution"
    assert "agent" not in out
    assert _validate_react_input("rag.query", out, registry) is None

    # Stray tool_id key dropped (trace iter-3 shape) — it would now also be
    # rejected as an unknown key, which is why dropping it is load-bearing.
    out = _normalize_react_input(
        "doc.convert", {"tool_id": "doc.convert", "file_id": "f", "target_format": "md"}
    )
    assert "tool_id" not in out
    assert _validate_react_input("doc.convert", out, registry) is None


def test_react_validation_hints_without_executing() -> None:
    from rip_maf.orchestration.react import _validate_react_input, run_react
    from rip_maf.tools.registry import get_default_tool_registry

    registry = get_default_tool_registry()
    assert "query" in (_validate_react_input("rag.query", {}, registry) or "")
    assert "target_format" in (
        _validate_react_input("doc.convert", {"file_id": "f"}, registry) or ""
    )
    assert "labels" in (
        _validate_react_input("plot.chart", {"chart_type": "bar"}, registry) or ""
    )

    # Malformed first turn gets a retry hint WITHOUT burning a step;
    # normalized second turn executes and the loop finishes.
    provider = FakeLayeredProvider(queued=[
        {"thought": "bad shape", "executor": "rag.query",
         "input": {"agent": {"message": ""}}, "is_final": False},
        {"thought": "need docs", "executor": "rag.query",
         "input": {"query": "fire"}, "is_final": False},
        {"thought": "have chunks", "executor": "reasoning",
         "input": {}, "is_final": True, "answer": "react final"},
    ])
    agents, tools = _react_orchestrator(provider)
    outcome = run_react(
        "what do docs say?", provider, agents, tools,
        trace_id="t", corpus_id="nb-1",
    )
    assert outcome.result.step_results[-1].output == "react final"
    # Only ONE executed tool step (r2) + final answer: the malformed r1
    # never reached execution.
    assert [s.step_id for s in outcome.plan.steps] == ["r2", "r3"]


def test_react_recovers_final_answer_stranded_in_input() -> None:
    # Trace 35e8fbd9 iter-1: is_final=true, answer empty, the full answer
    # rode inside input.content of a malformed doc.generate call. The loop
    # must recover it instead of discarding a correct answer.
    from rip_maf.orchestration.react import run_react

    provider = FakeLayeredProvider(queued=[
        {"thought": "format as table", "executor": "doc.generate",
         "input": {"title": "T", "content": "| A | B |\n|---|---|"},
         "is_final": True},
    ])
    agents, tools = _react_orchestrator(provider)
    outcome = run_react(
        "in a table format", provider, agents, tools,
        trace_id="t", corpus_id="nb-1",
    )
    assert outcome.result.step_results[-1].status.value == "success"
    assert outcome.result.step_results[-1].output == "| A | B |\n|---|---|"
    assert provider.structured_calls == 1


def test_react_empty_final_answer_retries_with_actionable_hint() -> None:
    # Two consecutive is_final=true turns with no answer anywhere: the
    # first gets an actionable correction (not the old "retry." non-hint),
    # the second fails the loop honestly — and the last-resort step
    # (ADR-035) still answers the request rather than surfacing the
    # routing error.
    from rip_maf.orchestration.react import run_react

    provider = FakeLayeredProvider(queued=[
        {"thought": "done", "executor": "reasoning",
         "input": {}, "is_final": True},
        {"thought": "still done", "executor": "reasoning",
         "input": {}, "is_final": True},
    ])
    agents, tools = _react_orchestrator(provider)
    outcome = run_react(
        "answer this", provider, agents, tools,
        trace_id="t", corpus_id="nb-1",
    )
    # The loop itself failed (no executed step, no is_final)...
    assert provider.structured_calls == 2
    # ...but the run ends with the last-resort reasoning answer.
    assert [s.step_id for s in outcome.plan.steps] == ["r-lastresort"]
    assert outcome.result.step_results[-1].status.value == "success"
    assert outcome.result.step_results[-1].output == "layered answer"


def test_react_prompt_carries_all_tool_schemas() -> None:
    # Trace 35e8fbd9 iter-2: the model guessed doc.convert fields for
    # doc.generate twice — the prompt never showed doc.generate's schema.
    from rip_maf.orchestration.react import run_react

    provider = FakeLayeredProvider(queued=[
        {"thought": "answer", "executor": "reasoning",
         "input": {}, "is_final": True, "answer": "ok"},
    ])
    agents, tools = _react_orchestrator(provider)
    run_react("q", provider, agents, tools, trace_id="t", corpus_id="nb-1")
    system = provider.prompts[0][0]["content"]
    assert '"sections"' in system
    assert '"heading"' in system
    assert '"message"' in system  # image.generate


def test_react_repeat_of_failed_executor_is_idle_turn() -> None:
    # Trace: rag.query ×2 (same empty result) — the second identical
    # proposal must not execute.
    from rip_maf.orchestration.react import run_react
    from rip_maf.tools.base import Tool, ToolRequest, ToolResponse
    from rip_maf.tools.registry import ToolRegistry

    class _FailTool(Tool):
        tool_id = "fail.tool"
        name = "Fail"
        description = "always fails"
        effect_class = "read-only"  # type: ignore[assignment]

        def __init__(self):
            self.calls = 0

        def execute(self, request: ToolRequest) -> ToolResponse:
            self.calls += 1
            return ToolResponse(
                tool_id=self.tool_id, ok=False, output=None, error="boom"
            )

    provider = FakeLayeredProvider(queued=[
        {"thought": "try it", "executor": "fail.tool",
         "input": {}, "is_final": False},
        {"thought": "try it again", "executor": "fail.tool",
         "input": {}, "is_final": False},
        {"thought": "give up", "executor": "reasoning",
         "input": {}, "is_final": True, "answer": "failed over"},
    ])
    agents, _ = _react_orchestrator(provider)
    fail_tool = _FailTool()
    tools = ToolRegistry()
    tools.register(fail_tool)
    outcome = run_react(
        "do the thing", provider, agents, tools,
        trace_id="t", corpus_id="nb-1",
    )
    # One proposal executes (plan_graph retries a failed step 3×: 1 + 2
    # retries); the identical repeat never executes (6 calls without it).
    assert fail_tool.calls == 3
    assert outcome.result.step_results[-1].output == "failed over"
    # r1 executed, r2 idle (no step), r3 final answer.
    assert [s.step_id for s in outcome.plan.steps] == ["r1", "r3"]
    assert provider.structured_calls == 3


def test_react_substitutes_rag_query_on_empty_corpus() -> None:
    # Trace c9b02eef / 27dcf635: rag.query on "(no documents)" provably
    # returns "(no chunks retrieved)". ADR-035: it is now SUBSTITUTED by a
    # general-knowledge reasoning step that executes (so the observation
    # reaches the scratchpad), not refused as a wasted idle turn.
    from rip_maf.orchestration.react import run_react

    provider = FakeLayeredProvider(queued=[
        {"thought": "need data", "executor": "rag.query",
         "input": {"query": "gdp"}, "is_final": False},
        {"thought": "answer directly", "executor": "reasoning",
         "input": {}, "is_final": True, "answer": "no docs answer"},
    ])
    agents, tools = _react_orchestrator(provider)
    outcome = run_react(
        "plot gdp", provider, agents, tools,
        trace_id="t", corpus_id="nb-1",
        corpus_context="(no documents)",
    )
    # r1 executed as `reasoning` (the substitute), r2 answered.
    assert [s.step_id for s in outcome.plan.steps] == ["r1", "r2"]
    assert outcome.plan.steps[0].agent_id == "reasoning"
    assert outcome.plan.steps[0].tool_id is None
    assert "no ready documents" in outcome.plan.steps[0].input["message"]
    assert outcome.result.step_results[-1].output == "no docs answer"
    assert provider.structured_calls == 2


def test_react_substitution_precedes_input_validation() -> None:
    # A malformed rag.query on an empty corpus must NOT burn an idle turn
    # on a correct-shape hint for a call that was never going to run.
    from rip_maf.orchestration.react import run_react

    provider = FakeLayeredProvider(queued=[
        {"thought": "no query at all", "executor": "rag.query",
         "input": {}, "is_final": False},
        {"thought": "answer", "executor": "reasoning", "input": {},
         "is_final": True, "answer": "done"},
    ])
    agents, tools = _react_orchestrator(provider)
    outcome = run_react(
        "what is the trend?", provider, agents, tools,
        trace_id="t", corpus_id="nb-1",
        corpus_context="(no documents)",
        route_intent="knowledge_qa",
    )
    assert [s.step_id for s in outcome.plan.steps] == ["r1", "r2"]
    assert outcome.plan.steps[0].agent_id == "reasoning"
    assert provider.structured_calls == 2


def test_react_prompt_allows_parametric_numbers_without_docs() -> None:
    from rip_maf.orchestration.react import run_react

    seen: list = []

    class _ProbeProvider(FakeLayeredProvider):
        def generate_structured(self, model, messages, schema, *, temperature=0.0):
            seen.append(messages)
            return {"thought": "done", "executor": "reasoning",
                    "input": {}, "is_final": True, "answer": "ok"}

    agents, tools = _react_orchestrator(_ProbeProvider())
    run_react("plot gdp", _ProbeProvider(), agents, tools,
              trace_id="t", corpus_id="nb-1",
              corpus_context="(no documents)")
    system = seen[0][0]["content"]
    assert "recall approximate figures with a reasoning step first" in system


def test_react_prompt_states_flat_shapes_and_plot_preference() -> None:
    from rip_maf.orchestration.react import run_react

    seen: list = []

    class _ProbeProvider(FakeLayeredProvider):
        def generate_structured(self, model, messages, schema, *, temperature=0.0):
            seen.append(messages)
            return {"thought": "done", "executor": "reasoning",
                    "input": {}, "is_final": True, "answer": "ok"}

    agents, tools = _react_orchestrator(_ProbeProvider())
    run_react("plot this", _ProbeProvider(), agents, tools,
              trace_id="t", corpus_id="nb-1")
    system = seen[0][0]["content"]
    assert '{"query": "..."' in system
    assert "never nested under 'agent'" in system
    assert "plot.chart" in system
    assert "title" in system


def test_orchestrator_falls_back_to_react_on_builder_miss() -> None:
    provider = FakeLayeredProvider(queued=[
        {"intent": "unknown", "queries": [], "confidence": 0.0},
        {"thought": "need docs", "executor": "rag.query",
         "input": {"query": "x"}, "is_final": False},
        {"thought": "done", "executor": "reasoning",
         "input": {}, "is_final": True, "answer": "react rescued"},
    ])
    assert _orchestrator(provider).run("a long failing request here", "nb-1").summary == "react rescued"
    assert provider.structured_calls == 3  # router + 2 react turns


# --- Option A observability: router sibling span, plan enrichment, react spans.
#
# No Langfuse server needed: `manual_span` is monkeypatched with a recorder,
# so these tests prove the wiring (names, explicit parenting, outputs)
# without any tracing backend.


class _SpanRecorder:
    """Stand-in for `manual_span`: records (name, trace_context, updates)."""

    def __init__(self):
        self.spans: list[dict] = []

    def __call__(self, name, *, as_type="span", input=None, output=None,
                 metadata=None, trace_context=None, **extra):
        rec: dict = {
            "name": name,
            "trace_context": trace_context,
            "input": input,
            "output": None,
            "updates": [],
        }
        self.spans.append(rec)
        obs_id = f"span-{len(self.spans)}"

        class _Obs:
            id = obs_id

            def update(self, **kw):
                rec["updates"].append(kw)
                if "output" in kw:
                    rec["output"] = kw["output"]

        class _Ctx:
            def __enter__(self):
                return _Obs()

            def __exit__(self, *exc):
                return False

        return _Ctx()

    def by_name(self, name: str) -> list[dict]:
        return [s for s in self.spans if s["name"] == name]


def _compare_provider() -> FakeLayeredProvider:
    return FakeLayeredProvider(queued=[
        {
            "intent": "compare_multi",
            "queries": ["fire report sections", "faultbook sections"],
            "confidence": 0.9,
        },
    ])


def test_plan_event_carries_routing_fields() -> None:
    events: list[dict] = []
    result = _orchestrator(_compare_provider()).run(
        COMPARE_REQUEST, "nb-1", on_event=events.append,
    )
    assert result.status == "success"
    plans = [e for e in events if e.get("type") == "plan"]
    assert len(plans) == 1
    route = plans[0].get("route") or {}
    assert route.get("intent") == "compare_multi"
    assert route.get("routed_by") == "llm"
    assert route.get("confidence") == 0.9


def test_greeting_plan_event_llm_route() -> None:
    events: list[dict] = []
    provider = FakeLayeredProvider(queued=[
        {"intent": "chat", "queries": [], "confidence": 0.95},
    ])
    result = _orchestrator(provider).run(
        "hi", "nb-1", on_event=events.append,
    )
    assert result.status == "success"
    plans = [e for e in events if e.get("type") == "plan"]
    assert len(plans) == 1
    route = plans[0].get("route") or {}
    assert route.get("routed_by") == "llm"
    assert route.get("intent") == "chat"


def test_router_span_is_sibling_of_plan_under_run(monkeypatch) -> None:
    from rip_maf.orchestration import engine, plan_graph

    recorder = _SpanRecorder()
    monkeypatch.setattr(engine, "manual_span", recorder)
    # Hermetic step spans: when a real Langfuse client is primed (full
    # suite via app.main lifespan), the sentinel plan-span ctx below would
    # otherwise reach the real plan_graph.manual_span and raise on the
    # invalid IDs. Step-span parenting is not under test here.
    monkeypatch.setattr(plan_graph, "manual_span", recorder)
    # Distinct well-formed sentinel for the plan-span context: proves the
    # router span does NOT parent under `plan` (it must carry the run ctx
    # instead). Well-formed (dict with trace_id/parent_span_id) so the
    # real step spans downstream still parent correctly when tracing is on.
    _plan_ctx = {"trace_id": "T", "parent_span_id": "PLAN-SPAN"}
    monkeypatch.setattr(engine, "get_trace_context", lambda: dict(_plan_ctx))
    result = _orchestrator(_compare_provider()).run(COMPARE_REQUEST, "nb-1")
    assert result.status == "success"
    routers = recorder.by_name("router")
    plans = recorder.by_name("plan")
    assert len(routers) == 1 and len(plans) == 1
    # Sibling parenting: both spans carry the run ctx from config, never the
    # plan-span ctx captured later via get_trace_context().
    assert routers[0]["trace_context"] == plans[0]["trace_context"]
    assert routers[0]["trace_context"] != _plan_ctx
    assert routers[0]["output"]["intent"] == "compare_multi"
    assert routers[0]["output"]["routed_by"] == "llm"
    assert plans[0]["output"]["layer"] == "L2-builder"
    assert plans[0]["output"]["intent"] == "compare_multi"


def test_builder_miss_emits_no_plan_span_and_runs_react(monkeypatch) -> None:
    from rip_maf.orchestration import engine

    recorder = _SpanRecorder()
    monkeypatch.setattr(engine, "manual_span", recorder)
    provider = FakeLayeredProvider(queued=[
        {"intent": "unknown", "queries": [], "confidence": 0.0},
        {"thought": "answer directly", "executor": "reasoning",
         "input": {}, "is_final": True, "answer": "react answer"},
    ])
    result = _orchestrator(provider).run(
        "a vague long request with no clear shape at all here", "nb-1"
    )
    assert result.status == "success"
    # Builder miss delegates to L3 ReAct: no L2 plan span is emitted.
    assert recorder.by_name("plan") == []
    routers = recorder.by_name("router")
    assert len(routers) == 1
    assert routers[0]["output"]["intent"] == "unknown"


def test_react_iteration_spans(monkeypatch) -> None:
    import rip_maf.orchestration.react_engine as react_mod
    from rip_maf.orchestration.react import run_react

    recorder = _SpanRecorder()
    monkeypatch.setattr(react_mod, "_manual_span", recorder)
    provider = FakeLayeredProvider(queued=[
        {"thought": "need docs", "executor": "rag.query",
         "input": {"query": "fire"}, "is_final": False},
        {"thought": "have chunks", "executor": "reasoning",
         "input": {}, "is_final": True, "answer": "react final"},
    ])
    agents, tools = _react_orchestrator(provider)
    outcome = run_react(
        "what do docs say?", provider, agents, tools,
        trace_id="t", corpus_id="nb-1",
    )
    assert outcome.result.step_results[-1].output == "react final"
    iters = [s for s in recorder.spans if s["name"].startswith("react:iter-")]
    assert [s["name"] for s in iters] == ["react:iter-1", "react:iter-2"]
    assert iters[0]["output"]["status"] == "success"
    assert iters[0]["output"]["executor"] == "rag.query"
    assert iters[1]["output"]["is_final"] is True
    # Both iterations share the same explicit parent (the react ctx).
    assert iters[0]["trace_context"] == iters[1]["trace_context"]


def test_orchestrator_react_span(monkeypatch) -> None:
    import rip_maf.observability.langfuse as lf
    import rip_maf.orchestration.react_engine as react_mod

    recorder = _SpanRecorder()
    # Orchestrator imports manual_span lazily (picks up the lf patch);
    # react binds _manual_span at module import, so patch both with the
    # same recorder. engine/plan_graph keep the real no-op here.
    monkeypatch.setattr(lf, "manual_span", recorder)
    monkeypatch.setattr(react_mod, "_manual_span", recorder)
    provider = FakeLayeredProvider(queued=[
        {"intent": "unknown", "queries": [], "confidence": 0.0},
        {"thought": "need docs", "executor": "rag.query",
         "input": {"query": "x"}, "is_final": False},
        {"thought": "done", "executor": "reasoning",
         "input": {}, "is_final": True, "answer": "react rescued"},
    ])
    assert _orchestrator(provider).run("a long failing request here", "nb-1").summary == "react rescued"
    reacts = recorder.by_name("react")
    assert len(reacts) == 1
    assert reacts[0]["output"]["status"] == "success"
    assert reacts[0]["output"]["steps"] == 2
    iters = [s for s in recorder.spans if s["name"].startswith("react:iter-")]
    assert len(iters) == 2


def test_qa_empty_corpus_skips_rag_query() -> None:
    # Trace ea48cb30: "What is QLoRA?" with "(no documents)" must build a
    # single general-answer reasoning step — no rag.query execution, only
    # the router structured call.
    provider = FakeLayeredProvider(queued=[
        {"intent": "qa_single", "queries": [], "confidence": 0.95},
    ])
    result = _orchestrator(provider).run(
        "What is QLoRA?", "nb-empty", corpus_context="(no documents)",
    )
    assert result.status == "success"
    assert result.summary == "layered answer"
    assert len(result.step_results) == 1
    assert result.step_results[0].agent_id == "reasoning"
    assert provider.structured_calls == 1


def test_compare_empty_corpus_yields_clarification() -> None:
    provider = FakeLayeredProvider(queued=[
        {"intent": "compare_multi", "queries": [], "confidence": 0.9},
    ])
    result = _orchestrator(provider).run(
        "compare both reports in this empty corpus please",
        "nb-empty",
        corpus_context="(no documents)",
    )
    assert result.status == "success"
    assert len(result.step_results) == 1
    assert result.shown == ["1"] and result.hidden == []
    assert provider.structured_calls == 1


def test_react_broad_ask_defaults_rag_query_to_overview() -> None:
    # Trace cfbaa9c3: "summarize the docs" loop retrieved REFERENCES via
    # specific ranking instead of overview stratification.
    from rip_maf.orchestration.react import _default_react_mode, run_react

    assert _default_react_mode("summarize the docs", {"query": "x"})["mode"] == "overview"
    assert "mode" not in _default_react_mode("what is QLoRA?", {"query": "x"})
    assert _default_react_mode("summarize", {"query": "x", "mode": "specific"})["mode"] == "specific"

    provider = FakeLayeredProvider(
        text="synthesized",
        queued=[
            {"thought": "need docs", "executor": "rag.query",
             "input": {"query": "summarize"}, "is_final": False},
            {"thought": "have chunks", "executor": "reasoning",
             "input": {}, "is_final": True, "answer": "react final"},
        ],
    )
    agents, tools = _react_orchestrator(provider)
    outcome = run_react(
        "summarize the docs", provider, agents, tools,
        trace_id="t", corpus_id="nb-1",
    )
    rag_steps = [s for s in outcome.plan.steps if s.tool_id == "rag.query"]
    assert rag_steps and all(s.input.get("mode") == "overview" for s in rag_steps)


def test_react_synthesizes_answer_when_iterations_exhaust() -> None:
    # Trace cfbaa9c3 iter-6: no is_final, final summary was a truncated raw
    # chunk dump. The loop must synthesize prose from observations instead.
    from rip_maf.orchestration.aggregator import Aggregator
    from rip_maf.orchestration.react import run_react

    provider = FakeLayeredProvider(
        text="synthesized summary covering both docs",
        queued=[
            {"thought": "get more", "executor": "rag.query",
             "input": {"query": "summarize"}, "is_final": False},
            {"thought": "get even more", "executor": "rag.query",
             "input": {"query": "summarize again"}, "is_final": False},
        ],
    )
    agents, tools = _react_orchestrator(provider)
    outcome = run_react(
        "summarize the docs", provider, agents, tools,
        trace_id="t", corpus_id="nb-1", max_iterations=2,
    )
    assert outcome.result.step_results[-1].agent_id == "reasoning"
    assert outcome.result.step_results[-1].output == "synthesized summary covering both docs"
    agg = Aggregator().aggregate(outcome.plan, outcome.result)
    assert agg.summary == "synthesized summary covering both docs"
    assert agg.shown == ["r-final"]


def test_react_plot_nested_values_is_idle_hint() -> None:
    # Trace 07fb4f59 iters 1+4: nested `values` arrays (and an invented
    # `series_labels` key) burned executions on "'values' must all be
    # numbers". Malformed shapes must get a corrective hint WITHOUT
    # executing so the iteration budget survives for a fixed shape.
    from rip_maf.orchestration.react import _validate_react_input, run_react
    from rip_maf.tools.registry import get_default_tool_registry

    registry = get_default_tool_registry()
    hint = _validate_react_input(
        "plot.chart",
        {"chart_type": "bar", "labels": ["Documents", "Pages"],
         "values": [[229, 135], [66, 47.5]]},
        registry,
    )
    assert hint is not None and "flat" in hint and "series" in hint
    hint2 = _validate_react_input(
        "plot.chart",
        {"chart_type": "bar", "labels": ["A", "B"], "values": [1, 2],
         "series_labels": ["Documents"]},
        registry,
    )
    assert hint2 is not None and "series_labels" in hint2

    provider = FakeLayeredProvider(queued=[
        {"thought": "bad shape", "executor": "plot.chart",
         "input": {"chart_type": "bar", "labels": ["Documents", "Pages"],
                   "values": [[229, 135], [66, 47.5]]}, "is_final": False},
        {"thought": "fixed shape", "executor": "plot.chart",
         "input": {"chart_type": "bar", "labels": ["DocBench", "MMLongBench"],
                   "values": [229, 135], "title": "docs"}, "is_final": False},
        {"thought": "done", "executor": "reasoning",
         "input": {}, "is_final": True, "answer": "have chart"},
    ])
    agents, tools = _react_orchestrator(provider)
    outcome = run_react(
        "compare docs as bar graph", provider, agents, tools,
        trace_id="t", corpus_id="nb-1",
    )
    # Malformed r1 never executed: only the fixed r2 chart + final answer.
    assert [s.step_id for s in outcome.plan.steps] == ["r2", "r3"]
    assert outcome.result.step_results[-1].output == "have chart"


def test_react_plot_without_title_is_idle_hint() -> None:
    # Trace affdbbd4: 3 of 4 charts rendered untitled — the model was
    # never asked for one. A title-less proposal must get a corrective
    # hint WITHOUT executing so the retry carries the same data + title.
    # This rule is the tool's `required_for_model`: a strict contract the
    # model must satisfy, that the renderer itself tolerates (it derives a
    # title from the data — trace affdbbd4).
    from rip_maf.orchestration.react import _validate_react_input, run_react
    from rip_maf.tools.registry import get_default_tool_registry

    registry = get_default_tool_registry()
    hint = _validate_react_input(
        "plot.chart",
        {"chart_type": "bar", "labels": ["A", "B"], "values": [1, 2]},
        registry,
    )
    assert hint is not None and "title" in hint
    hint_series = _validate_react_input(
        "plot.chart",
        {"chart_type": "bar", "labels": ["A", "B"],
         "series": [{"label": "s", "values": [1, 2]}]},
        registry,
    )
    assert hint_series is not None and "title" in hint_series
    assert _validate_react_input(
        "plot.chart",
        {"chart_type": "bar", "labels": ["A", "B"], "values": [1, 2],
         "title": "t"},
        registry,
    ) is None
    # The renderer itself still accepts it (derived title) — the strictness
    # is model-facing only.
    from rip_maf.tools.plot_chart import PlotChartTool

    assert PlotChartTool().validate_input(
        {"chart_type": "bar", "labels": ["A", "B"], "values": [1, 2]}
    ) is None

    provider = FakeLayeredProvider(queued=[
        {"thought": "plot it", "executor": "plot.chart",
         "input": {"chart_type": "bar", "labels": ["A", "B"],
                   "values": [1, 2]}, "is_final": False},
        {"thought": "plot it titled", "executor": "plot.chart",
         "input": {"chart_type": "bar", "labels": ["A", "B"],
                   "values": [1, 2], "title": "A vs B"}, "is_final": False},
        {"thought": "done", "executor": "reasoning",
         "input": {}, "is_final": True, "answer": "have chart"},
    ])
    agents, tools = _react_orchestrator(provider)
    outcome = run_react(
        "plot this", provider, agents, tools,
        trace_id="t", corpus_id="nb-1",
    )
    # Untitled r1 never executed: titled r2 chart + final answer.
    assert [s.step_id for s in outcome.plan.steps] == ["r2", "r3"]
    assert outcome.result.step_results[-1].output == "have chart"


def test_react_exact_successful_repeat_is_idle() -> None:
    # Trace 07fb4f59 r2/r3: the identical Avg-Tokens chart executed twice.
    # An exact repeat of a success must idle, not re-execute.
    from rip_maf.orchestration.react import run_react

    chart = {"chart_type": "bar", "labels": ["A", "B"],
             "values": [46377, 21214], "title": "tokens"}
    provider = FakeLayeredProvider(queued=[
        {"thought": "plot tokens", "executor": "plot.chart",
         "input": dict(chart), "is_final": False},
        {"thought": "plot tokens again", "executor": "plot.chart",
         "input": dict(chart), "is_final": False},
        {"thought": "done", "executor": "reasoning",
         "input": {}, "is_final": True, "answer": "have chart"},
    ])
    agents, tools = _react_orchestrator(provider)
    outcome = run_react(
        "plot tokens as bar graph", provider, agents, tools,
        trace_id="t", corpus_id="nb-1",
    )
    assert [s.step_id for s in outcome.plan.steps] == ["r1", "r3"]
    assert outcome.result.step_results[-1].output == "have chart"


def test_react_replot_same_data_is_idle() -> None:
    # Trace 07fb4f59 r5/r6: [229, 135] re-plotted under different
    # labels/title renders the same bars. Same data (ignoring cosmetics)
    # must idle so the frontend never shows the same plot twice.
    from rip_maf.orchestration.react import _plot_data_key, run_react

    assert _plot_data_key(
        {"chart_type": "bar", "labels": ["A", "B"], "values": [229, 135],
         "title": "Number of Documents"}
    ) == _plot_data_key(
        {"chart_type": "bar", "labels": ["A (x)", "B (x)"], "values": [229, 135]}
    )
    assert _plot_data_key(
        {"chart_type": "bar", "labels": ["A", "B"], "values": [229, 135]}
    ) != _plot_data_key(
        {"chart_type": "bar", "labels": ["A", "B"], "values": [46377, 21214]}
    )

    provider = FakeLayeredProvider(queued=[
        {"thought": "plot docs", "executor": "plot.chart",
         "input": {"chart_type": "bar",
                   "labels": ["DocBench (documents)", "MMLongBench (documents)"],
                   "values": [229, 135], "title": "Number of Documents"},
         "is_final": False},
        {"thought": "plot docs again", "executor": "plot.chart",
          "input": {"chart_type": "bar", "labels": ["DocBench", "MMLongBench"],
                    "values": [229, 135], "title": "Doc counts (retitled)"},
          "is_final": False},
        {"thought": "done", "executor": "reasoning",
         "input": {}, "is_final": True, "answer": "have chart"},
    ])
    agents, tools = _react_orchestrator(provider)
    outcome = run_react(
        "plot docs as bar graph", provider, agents, tools,
        trace_id="t", corpus_id="nb-1",
    )
    assert [s.step_id for s in outcome.plan.steps] == ["r1", "r3"]


def test_react_synthesis_evidence_collapses_charts() -> None:
    # Trace 07fb4f59 r-final: raw SVG evidence invited an ASCII redraw.
    # Chart successes must collapse to a one-liner in synthesis evidence.
    from rip_maf.agents.base import StepStatus
    from rip_maf.orchestration.react import _synthesis_evidence_line
    from rip_maf.orchestration.results import StepResult

    chart = StepResult(
        step_id="r2", agent_id="plot.chart", status=StepStatus.SUCCESS,
        output="<svg xmlns='x'>...</svg>",
    )
    line = _synthesis_evidence_line(chart)
    assert "<svg" not in line and "Artifacts" in line
    text = StepResult(
        step_id="r1", agent_id="rag.query", status=StepStatus.SUCCESS,
        output="chunk text here",
    )
    assert "chunk text here" in _synthesis_evidence_line(text)
