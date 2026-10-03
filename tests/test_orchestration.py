"""Phase 3.2 — orchestration: Planner → Engine → Aggregator.

Fakes throughout (no Ollama, no DB): FakeProvider serves canned structured
plans, FakeAgents execute deterministically, FakeRAG stands in for VectorRAG
(real e2e awaits the Phase 6 torch repair). Run with
``pytest backend/tests/test_orchestration.py -q --noconftest`` until the
torch env is repaired — the shared conftest imports app.main, which needs
sentence_transformers.
"""

from __future__ import annotations

import threading

import pytest

from rip_maf.agents.base import (
    Agent,
    DelegationRequest,
    DelegationResponse,
    StepStatus,
)
from rip_maf.agents.registry import AgentRegistry
from rip_maf.orchestration.aggregator import Aggregator
from rip_maf.orchestration.memory import (
    WINDOW_SIZE,
    MemoryContext,
    build_history_context,
    estimate_tokens,
)
from rip_maf.orchestration.orchestrator import (
    OrchestrationError,
    Orchestrator,
)
from rip_maf.orchestration.plan import Plan, PlanStep
from rip_maf.orchestration.plan_graph import run_plan_graph
from rip_maf.orchestration.planner import Planner
from rip_maf.orchestration.results import ExecutionResult, StepResult
from rip_maf.orchestration.validator import PlanValidationError, PlanValidator
from rip_maf.providers.base import ModelProvider
from rip_maf.tools.base import Tool, ToolRequest, ToolResponse
from rip_maf.tools.registry import ToolRegistry, get_default_tool_registry


class FakeProvider(ModelProvider):
    """Canned ModelProvider; records the model every call resolves to."""

    def __init__(self, structured: dict | None = None, text: str = "ok",
                 queued: list[dict] | None = None):
        self.structured = structured or {"goal": "g", "steps": []}
        self.text = text
        self.models: list[str] = []
        # Sequential payloads for router/react tests; every structured
        # prompt kept so tests can assert on prompt content.
        self.queued = list(queued) if queued else None
        self.prompts: list = []

    def generate(self, model, messages, *, temperature=0.2, max_tokens=None):
        self.models.append(model)
        return self.text

    def generate_stream(self, model, messages, *, temperature=0.2, max_tokens=None):
        self.models.append(model)
        yield self.text

    def generate_structured(self, model, messages, schema, *, temperature=0.0):
        self.models.append(model)
        self.prompts.append(messages)
        if self.queued:
            return dict(self.queued.pop(0))
        return dict(self.structured)

    def embed(self, model: str, text: str) -> list[float]:
        raise NotImplementedError("test fake")

    def list_available_models(self) -> list[dict]:
        return [{"id": "fake-model", "display_name": "fake-model"}]

    def generate_image(self, prompt: str):
        raise NotImplementedError("test fake")


class FakeAgent(Agent):
    """Deterministic agent; optionally streams one delta chunk."""

    agent_id = "fake"
    name = "Fake"
    description = "test agent"

    def __init__(self, output: str = "done", stream: bool = False,
                 clarify: bool = False, fail: bool = False):
        self.output = output
        self.stream = stream
        self.clarify = clarify
        self.fail = fail
        self.seen: list[dict] = []

    def execute(self, request: DelegationRequest) -> DelegationResponse:
        self.seen.append(dict(request.input))
        if self.fail:
            return DelegationResponse(
                step_id=request.step_id, status=StepStatus.FAILURE,
                output=None, error="fake agent failure",
            )
        if self.stream and request.on_delta is not None:
            request.on_delta("chunk-")
        return DelegationResponse(
            step_id=request.step_id, status=StepStatus.SUCCESS, output=self.output,
            needs_clarification=self.clarify,
        )


class FakeTool(Tool):
    """Deterministic recording tool for input-scoping assertions."""

    tool_id = "fake.tool"
    name = "FakeTool"
    description = "test tool"

    def __init__(self):
        self.seen: list[dict] = []

    def execute(self, request: ToolRequest) -> ToolResponse:
        self.seen.append(dict(request.input))
        return ToolResponse(tool_id=self.tool_id, ok=True, output="tool-out")


class FakeRAG:
    def __init__(self):
        self.seen: list[tuple] = []

    def retrieve_context(
        self, corpus_id, query, top_k=4, file_id=None, file_name=None, mode="specific"
    ):
        self.seen.append((corpus_id, query, top_k, file_id, mode))
        return {
            "query": query,
            "results": [
                {
                    "content": "chunk-one",
                    "source": "f.pdf",
                    "section": "H1",
                    "rerank_score": 0.9,
                }
            ],
        }


def _registries(agent: Agent, rag=None) -> tuple[AgentRegistry, ToolRegistry]:
    agents = AgentRegistry()
    agents.register(agent)
    return agents, get_default_tool_registry(rag=rag)


def _ok(step_id: str, output: str, executor: str = "reasoning") -> StepResult:
    return StepResult(
        step_id=step_id, agent_id=executor, status=StepStatus.SUCCESS, output=output
    )


def _fail(step_id: str, error: str, executor: str = "reasoning") -> StepResult:
    return StepResult(
        step_id=step_id, agent_id=executor, status=StepStatus.FAILURE, error=error
    )


# --- Validator ---


class TestValidator:
    def test_valid_plan_passes(self):
        from unittest.mock import MagicMock

        from rip_maf.agents.registry import get_default_agent_registry

        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(
                    step_id="1", agent_id="reasoning",
                    input={"message": "hi"},
                )
            ],
        )
        agents = get_default_agent_registry(MagicMock())
        tools = get_default_tool_registry(rag=FakeRAG())
        assert PlanValidator(agents, tools).validate(plan) is plan

    def test_unknown_agent_rejected(self):
        agents, tools = _registries(FakeAgent())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[PlanStep(step_id="1", agent_id="nope", input={})],
        )
        with pytest.raises(PlanValidationError):
            PlanValidator(agents, tools).validate(plan)

    def test_both_or_neither_rejected(self):
        agents, tools = _registries(FakeAgent())
        both = Plan(
            plan_id="p", goal="g",
            steps=[PlanStep(step_id="1", agent_id="fake", tool_id="rag.query")],
        )
        with pytest.raises(PlanValidationError):
            PlanValidator(agents, tools).validate(both)

    def test_cycle_rejected(self):
        agents, tools = _registries(FakeAgent())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", agent_id="fake", input={}, depends_on=["2"]),
                PlanStep(step_id="2", agent_id="fake", input={}, depends_on=["1"]),
            ],
        )
        with pytest.raises(PlanValidationError, match="cycle"):
            PlanValidator(agents, tools).validate(plan)

    def test_budget_rejected(self):
        agents, tools = _registries(FakeAgent())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id=str(i), agent_id="fake", input={})
                for i in range(3)
            ],
        )
        with pytest.raises(PlanValidationError):
            PlanValidator(agents, tools, max_steps=2).validate(plan)

    def test_dependent_plot_without_placeholder_rejected(self):
        # Run 4efaec2b shape: dependent plot, hardcoded literals, no {{id}}.
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", tool_id="rag.query", input={"query": "x"},
                         expected_output_type="chunks"),
                PlanStep(step_id="2", agent_id="fake", input={"message": "summarize {{1}}"},
                         depends_on=["1"], expected_output_type="summary"),
                PlanStep(step_id="3", tool_id="plot.chart",
                         input={"chart_type": "bar", "labels": ["Class 0", "Class 1"],
                                "values": [50, 50]},
                         depends_on=["2"], expected_output_type="chart"),
            ],
        )
        with pytest.raises(PlanValidationError, match="placeholder"):
            PlanValidator(agents, tools).validate(plan)

    def test_plot_referencing_non_numbers_rejected(self):
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", tool_id="rag.query", input={"query": "x"},
                         expected_output_type="chunks"),
                PlanStep(step_id="2", tool_id="plot.chart",
                         input={"chart_type": "bar", "labels": ["a"], "values": ["{{1}}"]},
                         depends_on=["1"], expected_output_type="chart"),
            ],
        )
        with pytest.raises(PlanValidationError, match="numbers-producing"):
            PlanValidator(agents, tools).validate(plan)

    def test_garbled_placeholder_in_values_rejected(self):
        # Run 66fd4ec3 shape: text glued around placeholders.
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", agent_id="fake", input={"message": "numbers"},
                         expected_output_type="numbers"),
                PlanStep(step_id="2", tool_id="plot.chart",
                         input={"chart_type": "bar", "labels": ["a", "b"],
                                "values": [",{{1}}", ",{{1}}"]},
                         depends_on=["1"], expected_output_type="chart"),
            ],
        )
        with pytest.raises(PlanValidationError, match="lone"):
            PlanValidator(agents, tools).validate(plan)

    def test_multi_source_plot_values_rejected(self):
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", agent_id="fake", input={"message": "n1"},
                         expected_output_type="numbers"),
                PlanStep(step_id="2", agent_id="fake", input={"message": "n2"},
                         expected_output_type="numbers"),
                PlanStep(step_id="3", tool_id="plot.chart",
                         input={"chart_type": "bar", "labels": ["a", "b"],
                                "values": ["{{1}}", "{{2}}"]},
                         depends_on=["1", "2"], expected_output_type="chart"),
            ],
        )
        with pytest.raises(PlanValidationError, match="merging numbers step"):
            PlanValidator(agents, tools).validate(plan)

    def test_mixed_literal_and_single_placeholder_passes(self):
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", agent_id="fake", input={"message": "n"},
                         expected_output_type="numbers"),
                PlanStep(step_id="2", tool_id="plot.chart",
                         input={"chart_type": "bar", "labels": ["a", "b"],
                                "values": [0.5, "{{1}}"]},
                         depends_on=["1"], expected_output_type="chart"),
            ],
        )
        assert PlanValidator(agents, tools).validate(plan) is plan

    def test_standalone_plot_with_literals_passes(self):
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", tool_id="plot.chart",
                         input={"chart_type": "bar", "labels": ["a", "b"], "values": [1, 2]},
                         expected_output_type="chart"),
            ],
        )
        assert PlanValidator(agents, tools).validate(plan) is plan

    def test_grounded_plot_pattern_passes(self):
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", tool_id="rag.query", input={"query": "x"},
                         expected_output_type="chunks"),
                PlanStep(step_id="2", agent_id="fake", input={"message": "numbers {{1}}"},
                         depends_on=["1"], expected_output_type="numbers"),
                PlanStep(step_id="3", agent_id="fake", input={"message": "summarize {{1}}"},
                         depends_on=["1"], expected_output_type="answer"),
                PlanStep(step_id="4", tool_id="plot.chart",
                         input={"chart_type": "bar", "labels": ["Fire", "Smoke"],
                                "values": ["{{2}}"]},
                         depends_on=["2"], expected_output_type="chart"),
            ],
        )
        assert PlanValidator(agents, tools).validate(plan) is plan

    def test_multiseries_plot_one_placeholder_per_series_passes(self):
        # Trace 27dcf635 shape: parametric numbers per series + one plot.
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", agent_id="fake", input={"message": "usa"},
                         expected_output_type="numbers"),
                PlanStep(step_id="2", agent_id="fake", input={"message": "china"},
                         expected_output_type="numbers"),
                PlanStep(step_id="3", tool_id="plot.chart",
                         input={"chart_type": "line",
                                "labels": ["2000", "2010", "2020"],
                                "series": [{"label": "USA", "values": ["{{1}}"]},
                                           {"label": "China", "values": ["{{2}}"]}]},
                         depends_on=["1", "2"], expected_output_type="chart"),
            ],
        )
        assert PlanValidator(agents, tools).validate(plan) is plan

    def test_plot_values_and_series_conflict_rejected(self):
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", tool_id="plot.chart",
                         input={"chart_type": "bar", "labels": ["a"],
                                "values": [1],
                                "series": [{"label": "x", "values": [1]}]},
                         expected_output_type="chart"),
            ],
        )
        with pytest.raises(PlanValidationError, match="exactly one"):
            PlanValidator(agents, tools).validate(plan)

    def test_plot_missing_values_and_series_rejected(self):
        # Trace 27dcf635 attempt-1 shape: labels + title, no data at all.
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", tool_id="plot.chart",
                         input={"chart_type": "line", "labels": ["USA", "China"],
                                "title": "GDP"},
                         expected_output_type="chart"),
            ],
        )
        with pytest.raises(PlanValidationError, match="missing required input"):
            PlanValidator(agents, tools).validate(plan)

    def test_multiseries_two_refs_in_one_series_rejected(self):
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", agent_id="fake", input={"message": "n1"},
                         expected_output_type="numbers"),
                PlanStep(step_id="2", agent_id="fake", input={"message": "n2"},
                         expected_output_type="numbers"),
                PlanStep(step_id="3", tool_id="plot.chart",
                         input={"chart_type": "line", "labels": ["a", "b"],
                                "series": [{"label": "x",
                                            "values": ["{{1}}", "{{2}}"]}]},
                         depends_on=["1", "2"], expected_output_type="chart"),
            ],
        )
        with pytest.raises(PlanValidationError, match="one placeholder per series"):
            PlanValidator(agents, tools).validate(plan)

    def test_ungrounded_doc_generate_rejected(self):
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", tool_id="rag.query", input={"query": "x"},
                         expected_output_type="chunks"),
                PlanStep(step_id="2", tool_id="doc.generate",
                         input={"title": "t", "sections": []},
                         depends_on=["1"], expected_output_type="document"),
            ],
        )
        with pytest.raises(PlanValidationError, match="answer/summary/text"):
            PlanValidator(agents, tools).validate(plan)

    def test_ungrounded_reasoning_rejected(self):
        # Run 4ad8adfc shape: reasoning depends on rag.query but message
        # carries no {{1}} — executes ungrounded, asks to re-upload.
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", tool_id="rag.query", input={"query": "x"},
                         expected_output_type="chunks"),
                PlanStep(step_id="2", agent_id="fake",
                         input={"message": "write 5 beginner MCQs"},
                         depends_on=["1"], expected_output_type="answer"),
            ],
        )
        with pytest.raises(PlanValidationError, match="placeholder"):
            PlanValidator(agents, tools).validate(plan)

    def test_grounded_reasoning_passes(self):
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", tool_id="rag.query", input={"query": "x"},
                         expected_output_type="chunks"),
                PlanStep(step_id="2", agent_id="fake",
                         input={"message": "write 15 MCQs from {{1}}"},
                         depends_on=["1"], expected_output_type="answer"),
            ],
        )
        assert PlanValidator(agents, tools).validate(plan) is plan

    def test_triple_mcq_fanout_passes(self):
        # Cap raised to 5 for the current model/host: 3 parallel writers
        # (e.g. beginner/intermediate/senior) validate fine.
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="Create 15 MCQs",
            steps=[
                PlanStep(step_id="1", tool_id="rag.query", input={"query": "x"},
                         expected_output_type="chunks"),
                *[
                    PlanStep(step_id=str(i), agent_id="fake",
                             input={"message": f"write 5 MCQs from {{{{{1}}}}} level {i}"},
                             depends_on=["1"], expected_output_type="answer")
                    for i in (2, 3, 4)
                ],
            ],
        )
        assert PlanValidator(agents, tools).validate(plan) is plan

    def test_six_way_fanout_rejected(self):
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", tool_id="rag.query", input={"query": "x"},
                         expected_output_type="chunks"),
                *[
                    PlanStep(step_id=str(i), agent_id="fake",
                             input={"message": f"write part {{{{{1}}}}} ({i})"},
                             depends_on=["1"], expected_output_type="answer")
                    for i in range(2, 8)
                ],
            ],
        )
        with pytest.raises(PlanValidationError, match="fan out|sequentially"):
            PlanValidator(agents, tools).validate(plan)

    def test_single_mcq_shape_passes(self):
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="Create 15 MCQs",
            steps=[
                PlanStep(step_id="1", tool_id="rag.query", input={"query": "x"},
                         expected_output_type="chunks"),
                PlanStep(step_id="2", agent_id="fake",
                         input={"message": "write 15 MCQs (5/5/5) from {{1}}"},
                         depends_on=["1"], expected_output_type="answer"),
            ],
        )
        assert PlanValidator(agents, tools).validate(plan) is plan

    def test_doc_generate_without_title_rejected(self):
        # Run 4ad8adfc attempt-1 shape: doc.generate with message, no title.
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", tool_id="doc.generate",
                         input={"message": "write MCQs"},
                         expected_output_type="document"),
            ],
        )
        with pytest.raises(PlanValidationError, match="title"):
            PlanValidator(agents, tools).validate(plan)

    def test_rag_without_query_rejected(self):
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", tool_id="rag.query", input={},
                         expected_output_type="chunks"),
            ],
        )
        with pytest.raises(PlanValidationError, match="query"):
            PlanValidator(agents, tools).validate(plan)

    def test_rag_query_must_be_typed_chunks(self):
        # Raw chunks typed as text would leak into the answer bubble.
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", tool_id="rag.query", input={"query": "x"},
                         expected_output_type="text"),
            ],
        )
        with pytest.raises(PlanValidationError, match="chunks"):
            PlanValidator(agents, tools).validate(plan)


# --- Aggregator (Q36 deterministic rules) ---


class TestAggregator:
    def test_empty_is_failed(self):
        agg = Aggregator().aggregate(
            Plan(plan_id="p", goal="g"),
            ExecutionResult(trace_id="t", step_results=[]),
        )
        assert agg.status == "failed" and agg.plan_incomplete

    def test_single_success_verbatim(self):
        agg = Aggregator().aggregate(
            Plan(plan_id="p", goal="g"),
            ExecutionResult(trace_id="t", step_results=[_ok("1", "the answer")]),
        )
        assert (agg.status, agg.summary) == ("success", "the answer")

    def test_multiple_success_labeled_join(self):
        agg = Aggregator().aggregate(
            Plan(plan_id="p", goal="g"),
            ExecutionResult(
                trace_id="t",
                step_results=[_ok("1", "aaa"), _ok("2", "bbb", executor="coding")],
            ),
        )
        assert agg.status == "success"
        assert agg.summary == (
            "Step 1 (reasoning): aaa\n\nStep 2 (coding): bbb"
        )

    def test_visibility_map_mcq_shape(self):
        # Example F: chunks HIDE, answer SHOW — answer bubble is SHOW-only.
        plan = Plan(
            plan_id="p", goal="Create 15 MCQs",
            steps=[
                PlanStep(step_id="1", tool_id="rag.query", input={"query": "x"},
                         expected_output_type="chunks"),
                PlanStep(step_id="2", agent_id="fake",
                         input={"message": "write MCQs from {{1}}"},
                         depends_on=["1"], expected_output_type="answer"),
            ],
        )
        agg = Aggregator().aggregate(
            plan,
            ExecutionResult(
                trace_id="t",
                step_results=[
                    _ok("1", "raw chunks", executor="rag.query"),
                    _ok("2", "Q1..."),
                ],
            ),
        )
        assert agg.summary == "Q1..."
        assert agg.shown == ["2"] and agg.hidden == ["1"]
        assert agg.visibility == {"1": "hide", "2": "show"}

    def test_partial_status(self):
        agg = Aggregator().aggregate(
            Plan(plan_id="p", goal="g"),
            ExecutionResult(
                trace_id="t", step_results=[_ok("1", "aaa"), _fail("2", "boom")]
            ),
        )
        assert agg.status == "partial"
        assert agg.summary.startswith("aaa")
        assert "Step 2 (reasoning) failed: boom" in agg.summary

    def test_all_failed_joins_errors(self):
        agg = Aggregator().aggregate(
            Plan(plan_id="p", goal="g"),
            ExecutionResult(trace_id="t", step_results=[_fail("1", "boom")]),
        )
        assert agg.status == "failed"
        assert "Step 1 (reasoning) failed: boom" in agg.summary

    def test_clarification_verbatim(self):
        agg = Aggregator().aggregate(
            Plan(plan_id="p", goal="g"),
            ExecutionResult(
                trace_id="t",
                step_results=[
                    StepResult(
                        step_id="1", agent_id="reasoning",
                        status=StepStatus.SUCCESS, output="Which file?",
                        needs_clarification=True,
                    )
                ],
            ),
        )
        assert agg.summary == "Which file?" and agg.needs_clarification

    def test_answer_plus_chart_shows_text_and_placeholder(self):
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", tool_id="rag.query", input={},
                         expected_output_type="chunks"),
                PlanStep(step_id="2", agent_id="reasoning", input={},
                         depends_on=["1"], expected_output_type="answer"),
                PlanStep(step_id="3", tool_id="plot.chart", input={},
                         depends_on=["2"], expected_output_type="chart"),
            ],
        )
        agg = Aggregator().aggregate(
            plan,
            ExecutionResult(
                trace_id="t",
                step_results=[
                    _ok("1", "raw chunks", executor="rag.query"),
                    _ok("2", "Fire 62pct, Smoke 38pct"),
                    StepResult(
                        step_id="3", agent_id="plot.chart",
                        status=StepStatus.SUCCESS,
                        output="<svg>chart</svg>",
                    ),
                ],
            ),
        )
        assert agg.status == "success"
        assert "Fire 62pct" in agg.summary
        assert "Chart generated" in agg.summary
        assert "raw chunks" not in agg.summary

    def test_numbers_plus_chart_hides_numbers(self):
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", agent_id="reasoning", input={},
                         expected_output_type="numbers"),
                PlanStep(step_id="2", tool_id="plot.chart", input={},
                         depends_on=["1"], expected_output_type="chart"),
            ],
        )
        agg = Aggregator().aggregate(
            plan,
            ExecutionResult(
                trace_id="t",
                step_results=[
                    _ok("1", "88518, 52770"),
                    StepResult(
                        step_id="2", agent_id="plot.chart",
                        status=StepStatus.SUCCESS,
                        output="<SVG>chart</SVG>",
                    ),
                ],
            ),
        )
        assert agg.summary == "Chart generated — see Artifacts below."

    def test_all_hidden_plus_failure_shows_failures_only(self):
        # Run 66fd4ec3 shape: hidden numbers + failed plot must not dump
        # the raw numbers CSV as the visible answer.
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", agent_id="reasoning", input={},
                         expected_output_type="numbers"),
                PlanStep(step_id="2", tool_id="plot.chart", input={},
                         depends_on=["1"], expected_output_type="chart"),
            ],
        )
        agg = Aggregator().aggregate(
            plan,
            ExecutionResult(
                trace_id="t",
                step_results=[
                    _ok("1", "0.88, 0.87"),
                    _fail("2", "'values' must all be numbers", executor="plot.chart"),
                ],
            ),
        )
        assert agg.status == "partial"
        assert "0.88" not in agg.summary
        assert "Step 2 (plot.chart) failed" in agg.summary

    def test_answer_plus_failed_chart_keeps_answer(self):
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", agent_id="reasoning", input={},
                         expected_output_type="answer"),
                PlanStep(step_id="2", tool_id="plot.chart", input={},
                         depends_on=["1"], expected_output_type="chart"),
            ],
        )
        agg = Aggregator().aggregate(
            plan,
            ExecutionResult(
                trace_id="t",
                step_results=[
                    _ok("1", "the summary"),
                    _fail("2", "bad values", executor="plot.chart"),
                ],
            ),
        )
        assert agg.status == "partial"
        assert "the summary" in agg.summary
        assert "Step 2 (plot.chart) failed: bad values" in agg.summary


# --- Conversation context (stateless) ---


class TestHistoryContext:
    def test_empty_is_none(self):
        assert build_history_context([]) is None
        assert build_history_context([{"role": "user", "content": "  "}]) is None

    def test_renders_newest_window_verbatim(self):
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(WINDOW_SIZE + 3)]
        rendered = build_history_context(msgs)
        assert rendered is not None
        assert f"m{WINDOW_SIZE + 2}" in rendered
        assert "m0" not in rendered  # aged out of the window

    def test_estimate_tokens(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens("abcdefgh") == 2
        assert isinstance(MemoryContext().as_prompt(), str)


# --- Plan graph ---


class TestPlanGraph:
    def test_rag_query_gets_corpus_id(self):
        rag = FakeRAG()
        agents, tools = _registries(FakeAgent(), rag=rag)
        plan = Plan(
            plan_id="p", goal="answer",
            steps=[
                PlanStep(
                    step_id="1", tool_id="rag.query",
                    input={"query": "hello"},  # no corpus_id from planner
                )
            ],
        )
        result = run_plan_graph(
            plan, agents, tool_registry=tools, trace_id="t", corpus_id="nb-7"
        )
        assert result.step_results[0].status is StepStatus.SUCCESS
        assert rag.seen and rag.seen[0][0] == "nb-7"  # injected, not LLM-made
        assert "chunk-one" in (result.step_results[0].output or "")
        assert result.step_results[0].data["sources"] == [
            {"source": "f.pdf", "section": "H1"}
        ]

    def test_agent_steps_do_not_get_corpus_id(self):
        agent = FakeAgent()
        agents, tools = _registries(agent, rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[PlanStep(step_id="1", agent_id="fake", input={"message": "hi"})],
        )
        run_plan_graph(
            plan, agents, tool_registry=tools, trace_id="t", corpus_id="nb-1"
        )
        assert "corpus_id" not in agent.seen[0]

    def test_placeholders_resolve_across_steps(self):
        agents, tools = _registries(FakeAgent(output="21"), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", agent_id="fake", input={"message": "count"}),
                PlanStep(
                    step_id="2", tool_id="plot.chart",
                    input={
                        "chart_type": "bar", "labels": ["x"],
                        "values": ["{{1}}"], "title": "n={{1}}",
                    },
                    depends_on=["1"],
                ),
            ],
        )
        result = run_plan_graph(plan, agents, tool_registry=tools, trace_id="t")
        assert result.step_results[1].status is StepStatus.SUCCESS
        assert result.step_results[1].data["point_count"] == 1

    def test_cancel_short_circuits(self):
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[PlanStep(step_id="1", agent_id="fake", input={})],
        )
        event = threading.Event()
        event.set()
        result = run_plan_graph(
            plan, agents, tool_registry=tools, trace_id="t", cancel_event=event
        )
        assert result.step_results[0].error == "run cancelled"

    def test_context_scoped_to_terminal_answer(self):
        agent = FakeAgent(output="done")
        agents, tools = _registries(agent, rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", agent_id="fake", input={"message": "numbers"},
                         expected_output_type="numbers"),
                PlanStep(step_id="2", agent_id="fake", input={"message": "answer {{1}}"},
                         depends_on=["1"], expected_output_type="answer"),
            ],
        )
        result = run_plan_graph(
            plan, agents, tool_registry=tools, trace_id="t",
            context="CTX", fallback_message="FB",
        )
        assert result.step_results[0].status is StepStatus.SUCCESS
        assert "context" not in agent.seen[0]  # intermediate: task + upstream only
        assert agent.seen[1].get("context") == "CTX"  # terminal prose: full context

    def test_tool_steps_get_no_context_or_fallback(self):
        tool = FakeTool()
        tools = ToolRegistry()
        tools.register(tool)
        agents, _ = _registries(FakeAgent(), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", tool_id="fake.tool",
                         input={"query": "x", "context": "stale", "history": []}),
            ],
        )
        result = run_plan_graph(
            plan, agents, tool_registry=tools, trace_id="t", corpus_id="nb-1",
            context="CTX", fallback_message="FB",
        )
        assert result.step_results[0].status is StepStatus.SUCCESS
        seen = tool.seen[0]
        assert "context" not in seen and "history" not in seen
        assert "message" not in seen  # no fallback prose for tools
        assert seen.get("corpus_id") == "nb-1"  # run truth still injected

    def test_trivial_plan_empty(self):
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        result = run_plan_graph(
            Plan(plan_id="p", goal="g", steps=[]), agents,
            tool_registry=tools, trace_id="t",
        )
        assert result.step_results == []


# --- Planner (thin provider holder; L3 mega-prompt removed) ---


class TestPlanner:
    def test_exposes_provider(self):
        provider = FakeProvider()
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        planner = Planner(provider, agents, tools)
        assert planner.provider is provider

    def test_has_no_mega_prompt(self):
        assert not hasattr(Planner, "plan"), "L3 mega-prompt Planner.plan removed"


class ReasoningFakeAgent(FakeAgent):
    """FakeAgent registered as `reasoning` so L2 builder plans validate."""

    agent_id = "reasoning"


def _recall_orchestrator(plans: list[dict], agent=None, rag=None):
    provider = FakeProvider(queued=plans)
    agents = AgentRegistry()
    agents.register(agent or ReasoningFakeAgent(output="recovered"))
    tools = get_default_tool_registry(rag=rag or FakeRAG())
    orch = Orchestrator(
        Planner(provider, agents, tools),
        PlanValidator(agents, tools),
        Aggregator(), agents, tools,
    )
    return provider, orch


def _router_miss() -> dict:
    """Router reports unknown so the run exercises L3 ReAct.

    The engine routes once before delegating; every queued test consumes
    one router payload first. Confidence 0.0 also covers the
    low-confidence → unknown path.
    """
    return {"intent": "unknown", "queries": [], "confidence": 0.0}


def _react_final(answer: str) -> dict:
    return {
        "thought": "answer directly", "executor": "reasoning",
        "input": {}, "is_final": True, "answer": answer,
    }


class TestReactFallback:
    def test_builder_miss_runs_react(self):
        provider, orch = _recall_orchestrator([
            _router_miss(),
            {"thought": "need docs", "executor": "rag.query",
             "input": {"query": "x"}, "is_final": False},
            _react_final("react rescued"),
        ])
        result = orch.run("a vague request with no clear shape", "nb-1")
        assert result.status == "success" and result.summary == "react rescued"
        assert len(provider.models) == 3  # router + 2 react turns

    def test_react_failure_falls_back_to_last_resort_answer(self):
        # The loop still fails fast (2 idle turns, no recall) but ADR-035
        # adds one last-resort reasoning answer, so the user gets an answer
        # instead of the internal "no deterministic builder" text.
        provider, orch = _recall_orchestrator([
            _router_miss(),
            {"thought": "bad pick", "executor": "ghost",
             "input": {}, "is_final": False},
            {"thought": "bad pick again", "executor": "ghost",
             "input": {}, "is_final": False},
        ])
        result = orch.run("a vague request with no clear shape", "nb-1")
        assert result.status == "success"
        assert [r.step_id for r in result.step_results] == ["r-lastresort"]
        # router + 2 idle react turns, no recall; the last-resort answer
        # runs on the agent, not the provider.
        assert len(provider.models) == 3

    def test_builder_clarification_is_terminal(self):
        provider, orch = _recall_orchestrator(
            [{"intent": "compare_multi", "queries": ["x"], "confidence": 0.9}],
            agent=ReasoningFakeAgent(output="Please upload documents", clarify=True),
        )
        result = orch.run(
            "compare both reports in this empty corpus please",
            "nb-1", corpus_context="(no documents)",
        )
        assert result.summary == "Please upload documents"
        assert result.needs_clarification
        assert len(provider.models) == 1  # router only; builder plan, no ReAct

    def test_cancel_suppresses_react(self):
        provider, orch = _recall_orchestrator([_router_miss(), _react_final("x")])
        event = threading.Event()
        event.set()
        with pytest.raises(OrchestrationError, match="cancelled"):
            orch.run("plot it", "nb-1", cancel_event=event)
        assert len(provider.models) == 0


# --- Full Orchestrator e2e ---


class TestOrchestrator:
    def _orchestrator(self, queued: list[dict], rag=None, stream=False,
                      output="final answer", agent=None):
        provider = FakeProvider(queued=queued)
        agents = AgentRegistry()
        agents.register(agent or ReasoningFakeAgent(output=output, stream=stream))
        tools = get_default_tool_registry(rag=rag or FakeRAG())
        planner = Planner(provider, agents, tools)
        validator = PlanValidator(agents, tools)
        return provider, Orchestrator(planner, validator, Aggregator(), agents, tools)

    def test_rag_plan_end_to_end(self):
        provider, orch = self._orchestrator(
            [{"intent": "qa_single", "queries": ["hello"], "confidence": 0.9}],
        )
        events: list[dict] = []
        result = orch.run(
            "what do docs say?", "nb-e2e",
            on_event=events.append, context="prior chat",
        )
        assert result.status == "success"
        # Type-aware aggregation (ADR-023 as amended): intermediate chunks
        # are hidden, so the terminal answer surfaces verbatim.
        assert result.summary == "final answer"
        assert "Step 1 (rag.query)" not in (result.summary or "")
        assert result.goal == "what do docs say?"
        types = [e["type"] for e in events]
        assert "step_started" in types and "step_completed" in types
        assert len(provider.models) == 1  # router only; L2 builder, no ReAct

    def test_streaming_delta_events(self):
        _provider, orch = self._orchestrator(
            [{"intent": "chat", "queries": [], "confidence": 0.95}],
            stream=True,
        )
        events: list[dict] = []
        orch.run("hi there friend, how are you doing today?", "nb-1",
                 on_event=events.append)
        deltas = [e for e in events if e["type"] == "delta"]
        assert deltas and deltas[0]["step_id"] == "1"

    def test_chat_builder_single_step(self):
        _provider, orch = self._orchestrator(
            [{"intent": "chat", "queries": [], "confidence": 0.95}],
        )
        result = orch.run("hi there friend, how are you doing today?", "nb-1")
        assert result.status == "success" and not result.plan_incomplete
        assert len(result.step_results) == 1
        assert result.summary == "final answer"

    def test_builder_miss_with_failed_react_still_answers(self):
        _provider, orch = self._orchestrator([
            {"intent": "unknown", "queries": [], "confidence": 0.0},
            {"thought": "bad pick", "executor": "ghost",
             "input": {}, "is_final": False},
            {"thought": "bad pick again", "executor": "ghost",
             "input": {}, "is_final": False},
        ])
        result = orch.run("a vague request with no clear shape here", "nb-1")
        assert result.status == "success"
        assert [r.step_id for r in result.step_results] == ["r-lastresort"]

    def test_cancelled_run_raises(self):
        provider, orch = self._orchestrator(
            [{"intent": "chat", "queries": [], "confidence": 0.95}])
        event = threading.Event()
        event.set()
        with pytest.raises(OrchestrationError, match="cancelled"):
            orch.run("hi there friend, how are you doing?", "nb-1",
                     cancel_event=event)
        assert len(provider.models) == 0

    def test_signature_has_corpus_id_and_context(self):
        import inspect

        params = list(inspect.signature(Orchestrator.run).parameters)
        assert params[1:5] == [
            "request_text", "corpus_id", "on_event", "context",
        ]

    def test_failed_react_surfaces_honest_without_retry(self):
        # No planner recall: builder miss → L3 ReAct → 2 idle turns → the
        # single last-resort answer (ADR-035). Still exactly one pass.
        provider, orch = self._orchestrator([
            {"intent": "unknown", "queries": [], "confidence": 0.0},
            {"thought": "bad pick", "executor": "ghost",
             "input": {}, "is_final": False},
            {"thought": "bad pick again", "executor": "ghost",
             "input": {}, "is_final": False},
        ])
        result = orch.run("a vague request with no clear shape here", "nb-1")
        assert result.summary == "final answer"
        # router + 2 idle react turns; the last-resort answer uses the
        # agent (no provider call), and there is no recall.
        assert len(provider.models) == 3

    def test_last_resort_failure_still_raises(self):
        # The last-resort answer is a best effort, not a fabrication: when
        # the agent/provider is down the run fails honest with the original
        # routing error (no silent empty answer).
        _provider, orch = self._orchestrator([
            {"intent": "unknown", "queries": [], "confidence": 0.0},
            {"thought": "bad pick", "executor": "ghost",
             "input": {}, "is_final": False},
            {"thought": "bad pick again", "executor": "ghost",
             "input": {}, "is_final": False},
        ], agent=ReasoningFakeAgent(output="", fail=True))
        with pytest.raises(OrchestrationError, match="L3 ReAct required"):
            orch.run("a vague request with no clear shape here", "nb-1")


# --- Nested executor ids (live trace: model buries agent_id/tool_id in input) ---


class TestNestedExecutorHoist:
    def test_from_model_hoists_tool_id(self):
        plan = Plan.from_model(
            "p", "g",
            [{"step_id": "1", "input": {"tool_id": "rag.query", "query": "x"}}],
        )
        assert plan.steps[0].tool_id == "rag.query"
        assert "tool_id" not in plan.steps[0].input
        assert plan.steps[0].input["query"] == "x"

    def test_from_model_hoists_agent_id(self):
        plan = Plan.from_model(
            "p", "g",
            [{"step_id": "1", "input": {"agent_id": "fake", "message": "hi"}}],
        )
        assert plan.steps[0].agent_id == "fake"
        assert "agent_id" not in plan.steps[0].input
        assert plan.steps[0].input["message"] == "hi"

    def test_from_model_leaves_top_level_set(self):
        plan = Plan.from_model(
            "p", "g",
            [{"step_id": "1", "agent_id": "fake",
              "input": {"message": "hi", "tool_id": "rag.query"}}],
        )
        assert plan.steps[0].agent_id == "fake"
        assert plan.steps[0].tool_id is None
        assert plan.steps[0].input["tool_id"] == "rag.query"

    def test_from_model_leaves_both_nested_for_validator(self):
        plan = Plan.from_model(
            "p", "g",
            [{"step_id": "1",
              "input": {"agent_id": "fake", "tool_id": "rag.query"}}],
        )
        assert plan.steps[0].agent_id == "" and plan.steps[0].tool_id is None
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        with pytest.raises(PlanValidationError, match="TOP-LEVEL"):
            PlanValidator(agents, tools).validate(plan)

    def test_validator_hint_names_top_level(self):
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[PlanStep(step_id="1", input={"tool_id": "rag.query"})],
        )
        with pytest.raises(PlanValidationError, match="TOP-LEVEL"):
            PlanValidator(agents, tools).validate(plan)

    def test_validator_without_nesting_has_no_hint(self):
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g", steps=[PlanStep(step_id="1", input={})]
        )
        with pytest.raises(PlanValidationError) as exc:
            PlanValidator(agents, tools).validate(plan)
        assert "TOP-LEVEL" not in str(exc.value)

# --- Structural repair (live traces 77930808/26e4974f: stray elements,
# nested step objects, omitted eot) ---


class TestStructuralRepair:
    def test_non_dict_element_rejected(self):
        with pytest.raises(ValueError, match="not an object"):
            Plan.from_model(
                "p", "g",
                [
                    {"step_id": "1", "tool_id": "rag.query",
                     "input": {"query": "x"},
                     "expected_output_type": "chunks"},
                    {"step_id": "2", "agent_id": "fake",
                     "input": {"message": "answer {{1}}"},
                     "depends_on": ["1"], "expected_output_type": "answer"},
                    "step_id",
                ],
            )

    def test_nested_step_in_input_rejected(self):
        with pytest.raises(ValueError, match="buries step"):
            Plan.from_model(
                "p", "g",
                [
                    {"step_id": "1", "tool_id": "rag.query",
                     "input": {"query": "x"},
                     "expected_output_type": "chunks"},
                    {"step_id": "3",
                     "input": {"message": "write MCQs {{1}}", "step_id": "2",
                               "depends_on": ["1"],
                               "expected_output_type": "answer"},
                     "depends_on": ["1"], "expected_output_type": "text"},
                ],
            )

    def test_omitted_rag_eot_defaults_chunks(self):
        plan = Plan.from_model(
            "p", "Create 15 MCQs",
            [
                {"step_id": "1", "tool_id": "rag.query",
                 "input": {"query": "x", "top_k": 4}},
                {"step_id": "2", "agent_id": "fake",
                 "input": {"message": "write 15 MCQs from {{1}}"},
                 "expected_output_type": "answer"},
            ],
        )
        assert plan.steps[0].expected_output_type == "chunks"
        assert plan.steps[1].depends_on == ["1"]
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        assert PlanValidator(agents, tools).validate(plan) is plan

    def test_validator_names_buried_step(self):
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[PlanStep(step_id="3", input={"message": "x", "step_id": "2"})],
        )
        with pytest.raises(PlanValidationError, match="buries step"):
            PlanValidator(agents, tools).validate(plan)

# --- Wiring-key hoist (live trace 214e7509: ornith-1.5:9b nests
# depends_on/expected_output_type inside input, twice running) ---


class TestKeyHoist:
    def _rag_then_writer(self, writer_input):
        return [
            {"step_id": "1", "tool_id": "rag.query",
             "input": {"query": "x", "top_k": 4}},
            {"step_id": "2", "agent_id": "fake", "input": writer_input},
        ]

    def test_attempt1_shape_hoisted(self):
        # depends_on + expected_output_type buried in input move up.
        plan = Plan.from_model(
            "p", "Summarize the document",
            self._rag_then_writer({
                "message": "Summarize using ONLY these chunks: {{1}}",
                "depends_on": ["1"], "expected_output_type": "answer",
            }),
        )
        assert plan.steps[0].expected_output_type == "chunks"
        assert plan.steps[1].depends_on == ["1"]
        assert plan.steps[1].expected_output_type == "answer"
        assert "depends_on" not in plan.steps[1].input
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        assert PlanValidator(agents, tools).validate(plan) is plan

    def test_attempt2_shape_hoisted(self):
        # expected_output_type "summary" buried in input moves up and
        # validates (summary is a known answer type).
        plan = Plan.from_model(
            "p", "Summarize the document",
            self._rag_then_writer({
                "message": "Using ONLY {{1}}, write a summary.",
                "expected_output_type": "summary",
            }),
        )
        assert plan.steps[1].depends_on == ["1"]  # auto-wired from {{1}}
        assert plan.steps[1].expected_output_type == "summary"
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        assert PlanValidator(agents, tools).validate(plan) is plan

    def test_contradiction_rejected(self):
        with pytest.raises(ValueError, match="contradicts"):
            Plan.from_model(
                "p", "g",
                [
                    {"step_id": "1", "tool_id": "rag.query",
                     "input": {"query": "x"},
                     "expected_output_type": "chunks"},
                    {"step_id": "2", "agent_id": "fake",
                     "input": {"message": "x {{1}}",
                               "expected_output_type": "summary"},
                     "depends_on": ["1"], "expected_output_type": "answer"},
                ],
            )

    def test_depends_on_contradiction_rejected(self):
        with pytest.raises(ValueError, match="contradicts"):
            Plan.from_model(
                "p", "g",
                [
                    {"step_id": "1", "tool_id": "rag.query",
                     "input": {"query": "x"},
                     "expected_output_type": "chunks"},
                    {"step_id": "2", "tool_id": "rag.query",
                     "input": {"query": "y", "depends_on": ["9"]},
                     "depends_on": ["1"],
                     "expected_output_type": "chunks"},
                ],
            )

    def test_echo_dropped(self):
        # input step_id equal to the top-level id is a harmless echo.
        plan = Plan.from_model(
            "p", "g",
            [
                {"step_id": "1", "tool_id": "rag.query",
                 "input": {"query": "x", "step_id": "1"},
                 "expected_output_type": "chunks"},
            ],
        )
        assert "step_id" not in plan.steps[0].input

    def test_whole_step_nesting_still_rejected(self):
        # Differing input step_id with no top-level executor: the old
        # buried-step shape stays a retry-actionable ValueError.
        with pytest.raises(ValueError, match="buries step"):
            Plan.from_model(
                "p", "g",
                [{
                    "input": {"message": "x", "step_id": "2",
                              "depends_on": ["1"],
                              "expected_output_type": "answer"},
                    "depends_on": [], "expected_output_type": "text",
                }],
            )


# --- Placeholder edges (live trace be49925d: grounded message, missing edge) ---


class TestPlaceholderEdges:
    def test_from_model_autowires_missing_edge(self):
        # Planner emits {{1}} but omits depends_on — same super-step means
        # the placeholder never resolves (literal "{{1}}" reaches the LLM).
        plan = Plan.from_model(
            "p", "Create 15 MCQs",
            [
                {"step_id": "1", "tool_id": "rag.query",
                 "input": {"query": "x", "top_k": 4},
                 "expected_output_type": "chunks"},
                {"step_id": "2", "agent_id": "fake",
                 "input": {"message": "write 15 MCQs from {{1}}"},
                 "expected_output_type": "answer"},
            ],
        )
        assert plan.steps[1].depends_on == ["1"]
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        assert PlanValidator(agents, tools).validate(plan) is plan

    def test_dangling_placeholder_rejected(self):
        agents, tools = _registries(FakeAgent(), rag=FakeRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", tool_id="rag.query", input={"query": "x"},
                         expected_output_type="chunks"),
                PlanStep(step_id="2", agent_id="fake",
                         input={"message": "use {{1}} and {{99}}"},
                         depends_on=["1"], expected_output_type="answer"),
            ],
        )
        with pytest.raises(PlanValidationError, match="unknown step"):
            PlanValidator(agents, tools).validate(plan)

    def test_autowired_mcq_shape_resolves_chunks(self):
        # Plan-level: from_model wires the missing edge so the writer sees
        # resolved chunks instead of literal "{{1}}" (executed via
        # run_plan_graph to bypass L1 routing).
        plan = Plan.from_model(
            "p", "Create 15 MCQs",
            [
                {"step_id": "1", "tool_id": "rag.query",
                 "input": {"query": "hello"},
                 "depends_on": [], "expected_output_type": "chunks"},
                {"step_id": "2", "agent_id": "fake",
                 "input": {"message": "write MCQs from {{1}}"},
                 "expected_output_type": "answer"},
            ],
        )
        agent = FakeAgent(output="final")
        agents = AgentRegistry()
        agents.register(agent)
        tools = get_default_tool_registry(rag=FakeRAG())
        result = run_plan_graph(
            plan, agents, tool_registry=tools, trace_id="t",
            corpus_id="nb-1",
        )
        assert result.step_results[1].status is StepStatus.SUCCESS
        assert "{{1}}" not in agent.seen[0]["message"]
        assert "chunk-one" in agent.seen[0]["message"]


# --- Fail-closed placeholders (trace cfbaa9c3: summarize timeouts) ---


class TestFailClosedPlaceholders:
    def test_missing_placeholder_resolves_to_empty_marker(self):
        from rip_maf.orchestration.plan_graph import _resolve_value

        assert _resolve_value("{{1}}", {}) == "(no chunks retrieved)"
        out = _resolve_value("Using ONLY ({{1}} {{2}}), summarize", {"1": "ABC"})
        assert "{{" not in out and "(no chunks retrieved)" in out

    def test_dependent_runs_on_empty_marker_after_upstream_timeout(self):
        class _TimeoutRAG:
            def retrieve_context(self, *a, **k):
                import time as _t
                _t.sleep(5)
                return {"query": "x", "results": []}

        agent = FakeAgent(output="seen")
        agents = AgentRegistry()
        agents.register(agent)
        tools = get_default_tool_registry(rag=_TimeoutRAG())
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", tool_id="rag.query",
                         input={"query": "x", "top_k": 1},
                         expected_output_type="chunks"),
                PlanStep(step_id="2", agent_id="fake",
                         input={"message": "summarize {{1}}"},
                         depends_on=["1"], expected_output_type="answer"),
            ],
        )
        result = run_plan_graph(
            plan, agents, tool_registry=tools, trace_id="t", timeout_ms=50,
        )
        assert result.step_results[0].status is StepStatus.FAILURE
        assert result.step_results[1].status is StepStatus.SUCCESS
        assert "{{1}}" not in agent.seen[0]["message"]
        assert "(no chunks retrieved)" in agent.seen[0]["message"]

    def test_overview_shard_gets_double_tool_budget(self):
        from rip_maf.orchestration.plan_graph import _step_timeout_ms

        overview = PlanStep(step_id="1", tool_id="rag.query",
                            input={"query": "x", "mode": "overview"})
        specific = PlanStep(step_id="2", tool_id="rag.query",
                            input={"query": "x", "mode": "specific"})
        assert _step_timeout_ms(overview, 30000) == 60000
        assert _step_timeout_ms(specific, 30000) == 30000

    def test_aggregator_demotes_answer_with_unresolved_placeholders(self):
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", tool_id="rag.query", input={"query": "x"},
                         expected_output_type="chunks"),
                PlanStep(step_id="2", agent_id="reasoning",
                         input={"message": "m"}, depends_on=["1"],
                         expected_output_type="answer"),
            ],
        )
        res = ExecutionResult(trace_id="t", step_results=[
            _fail("1", "step timed out", "rag.query"),
            _ok("2", "sorry, no access to {{1}} chunks"),
        ])
        agg = Aggregator().aggregate(plan, res)
        assert "{{1}}" not in agg.summary
        assert agg.shown == [] or all(s != "2" for s in agg.shown)
        assert agg.status in ("partial", "failed")

    def test_aggregator_collapses_identical_chart_outputs(self):
        # Trace 07fb4f59 r2/r3: the same SVG surfaced twice, so the
        # frontend showed the same plot twice. Identical terminal outputs
        # keep the first step shown; the rest hide.
        svg = "<svg xmlns='x'><title>same chart</title></svg>"
        plan = Plan(
            plan_id="p", goal="g",
            steps=[
                PlanStep(step_id="1", tool_id="plot.chart",
                         input={"chart_type": "bar"}, expected_output_type="chart"),
                PlanStep(step_id="2", tool_id="plot.chart",
                         input={"chart_type": "bar"}, expected_output_type="chart"),
            ],
        )
        res = ExecutionResult(trace_id="t", step_results=[
            _ok("1", svg, "plot.chart"),
            _ok("2", svg, "plot.chart"),
        ])
        agg = Aggregator().aggregate(plan, res)
        assert agg.shown == ["1"]
        assert agg.hidden == ["2"]
        assert agg.summary.count("Chart generated") == 1
