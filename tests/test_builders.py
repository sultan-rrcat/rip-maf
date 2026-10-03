"""Phase 2 — deterministic builder tests (no Ollama, no DB)."""

from __future__ import annotations

import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from rip_maf.agents.registry import AgentRegistry
from rip_maf.orchestration.builders import build, build_compare_multi
from rip_maf.orchestration.intents import Intent
from rip_maf.orchestration.router import RouterResult
from rip_maf.orchestration.validator import PlanValidator
from rip_maf.providers.base import ModelProvider
from rip_maf.tools.registry import get_default_tool_registry


class _FakeProvider(ModelProvider):
    def generate(self, model, messages, *, temperature=0.2, max_tokens=None):
        return "ok"

    def generate_structured(self, model, messages, schema, *, temperature=0.0):
        return {"goal": "g", "steps": []}

    def embed(self, model: str, text: str) -> list[float]:
        raise NotImplementedError("test fake")

    def list_available_models(self) -> list[dict]:
        return [{"id": "fake"}]


def _registry_with(*extra) -> AgentRegistry:
    """Reasoning plus any extra agent (the coding builder needs `coding`)."""
    from rip_maf.agents.reasoning import ReasoningAgent

    registry = AgentRegistry()
    registry.register(ReasoningAgent(_FakeProvider()))
    for agent in extra:
        registry.register(agent)
    return registry


def _validator(*extra) -> PlanValidator:
    return PlanValidator(_registry_with(*extra), get_default_tool_registry())


def test_compare_multi_structure_and_placeholders() -> None:
    plan = build_compare_multi(
        ["fire report sections", "faultbook report sections"],
        "compare both report and rank them based on complexity",
    )
    assert len(plan.steps) == 3
    assert [s.tool_id for s in plan.steps[:2]] == ["rag.query", "rag.query"]
    assert all(
        (s.expected_output_type or "").lower() == "chunks" for s in plan.steps[:2]
    )
    final = plan.steps[2]
    assert final.agent_id == "reasoning"
    assert final.depends_on == ["1", "2"]
    assert "{{1}}" in str(final.input) and "{{2}}" in str(final.input)
    _validator().validate(plan)  # must pass grounding checks


def test_compare_multi_pads_short_queries() -> None:
    plan = build_compare_multi([], "compare A and B")
    assert len(plan.steps) == 3
    _validator().validate(plan)


def test_compare_multi_per_file_fanout() -> None:
    from rip_maf.orchestration.builders import build_compare_multi

    plan = build_compare_multi(
        ["x", "y"], "compare both reports", _SNAPSHOT,
    )
    assert len(plan.steps) == 3
    assert plan.steps[0].input["file_id"] == "aaa111"
    assert plan.steps[1].input["file_id"] == "bbb222"
    assert plan.steps[0].input["mode"] == "specific"
    assert plan.steps[0].input["top_k"] == 4
    assert plan.steps[0].input["query"] == "compare both reports"
    _validator().validate(plan)


def test_summarize_per_file_overview() -> None:
    from rip_maf.orchestration.builders import build_summarize

    plan = build_summarize("summarize both docs", _SNAPSHOT)
    assert len(plan.steps) == 3
    assert all(s.input.get("mode") == "overview" for s in plan.steps[:2])
    assert all(s.input.get("top_k") == 4 for s in plan.steps[:2])
    assert plan.steps[2].depends_on == ["1", "2"]
    _validator().validate(plan)


def test_quiz_multi_file_overview() -> None:
    from rip_maf.orchestration.builders import build_quiz

    plan = build_quiz("key concepts", "make quiz", _SNAPSHOT)
    assert len(plan.steps) == 3
    assert all(s.input.get("mode") == "overview" for s in plan.steps[:2])
    _validator().validate(plan)


def test_compare_too_many_files_falls_to_react() -> None:
    snapshot = "; ".join(f"f{i}.pdf [ready] id=id{i}" for i in range(6))
    route = RouterResult(
        intent=Intent.COMPARE_MULTI, confidence=0.9,
        routed_by="llm",
    )
    assert build("compare many", route, snapshot) is None


def test_summarize_dispatch() -> None:
    route = RouterResult(intent=Intent.SUMMARIZE, confidence=0.9, routed_by="llm")
    plan = build("summarize docs", route, _SNAPSHOT)
    assert plan is not None and len(plan.steps) == 3


def test_rag_query_mode_validation() -> None:
    import pytest

    from rip_maf.orchestration.plan import Plan, PlanStep

    bad = Plan(
        plan_id="p", goal="g",
        steps=[
            PlanStep(
                step_id="1", tool_id="rag.query",
                input={"query": "x", "mode": "bogus"},
                expected_output_type="chunks",
            )
        ],
    )
    with pytest.raises(Exception, match="mode"):
        _validator().validate(bad)
    placeholder = Plan(
        plan_id="p", goal="g",
        steps=[
            PlanStep(
                step_id="1", tool_id="rag.query",
                input={"query": "x", "file_id": "{{1}}"},
                expected_output_type="chunks",
            )
        ],
    )
    with pytest.raises(Exception, match="placeholder"):
        _validator().validate(placeholder)


def test_build_dispatch() -> None:
    route = RouterResult(
        intent=Intent.COMPARE_MULTI,
        confidence=0.9,
        routed_by="llm",
    )
    plan = build("compare a and b", route)
    assert plan is not None and len(plan.steps) == 3

    chat = build("hello", RouterResult(intent=Intent.CHAT, confidence=1.0))
    assert chat is not None and len(chat.steps) == 1

    unknown = build("???", RouterResult(intent=Intent.UNKNOWN, confidence=0.0))
    assert unknown is None


_SNAPSHOT = (
    "2 file(s): Fire Report.pdf [ready] id=aaa111; "
    "Faultbook.pdf [ready] id=bbb222"
)


def test_quiz_single_writer_shape() -> None:
    route = RouterResult(
        intent=Intent.QUIZ, confidence=0.85,
        routed_by="llm",
    )
    plan = build("make 10 MCQs", route)
    assert plan is not None and len(plan.steps) == 2
    assert plan.steps[0].tool_id == "rag.query"
    assert plan.steps[1].depends_on == ["1"]
    assert "{{1}}" in str(plan.steps[1].input)
    _validator().validate(plan)


def test_convert_all_star_shape() -> None:
    route = RouterResult(
        intent=Intent.CONVERT_ALL, confidence=0.9, routed_by="llm",
        file_hint="*", target_format="pdf",
    )
    plan = build("convert all documents to pdf", route)
    assert plan is not None and len(plan.steps) == 1
    assert plan.steps[0].input == {"file_id": "*", "target_format": "pdf"}
    _validator().validate(plan)


def _assert_asks_counter_question(plan) -> None:
    """An unresolvable convert must ask, never guess (one reasoning call)."""
    assert plan is not None and len(plan.steps) == 1
    assert plan.steps[0].tool_id is None
    assert (plan.steps[0].expected_output_type or "").lower() == "clarification"
    _validator().validate(plan)


def test_convert_all_without_format_asks_for_format() -> None:
    route = RouterResult(
        intent=Intent.CONVERT_ALL, confidence=0.9, routed_by="llm",
        file_hint="*",
    )
    # ADR-035: no format stated -> counter-question, not a ReAct round trip.
    _assert_asks_counter_question(build("convert everything", route))


def test_convert_all_empty_corpus_clarifies_instead_of_failing() -> None:
    # "*" with zero ready files would build a doomed doc.convert (fails
    # after retries) — clarify instead. Live UAT: "Convert it to docx" on
    # an empty corpus recovered format=docx and failed the run.
    route = RouterResult(
        intent=Intent.CONVERT_ALL, confidence=0.9, routed_by="llm",
        file_hint="*", target_format="docx",
    )
    plan = build("convert it to docx", route, "(no documents)")
    assert plan is not None and len(plan.steps) == 1
    assert plan.steps[0].tool_id is None
    assert (plan.steps[0].expected_output_type or "").lower() == "clarification"
    _validator().validate(plan)


def test_convert_all_processing_corpus_asks_to_wait() -> None:
    route = RouterResult(
        intent=Intent.CONVERT_ALL, confidence=0.9, routed_by="llm",
        file_hint="*", target_format="pdf",
    )
    plan = build(
        "convert all to pdf", route, "1 file(s): big.pdf [processing] id=zzz"
    )
    assert plan is not None and len(plan.steps) == 1
    assert "wait" in plan.steps[0].input["message"]
    _validator().validate(plan)


def test_convert_one_resolves_literal_id() -> None:
    route = RouterResult(
        intent=Intent.CONVERT_ONE, confidence=0.9, routed_by="llm",
        file_hint="faultbook", target_format="md",
    )
    plan = build("convert faultbook to md", route, _SNAPSHOT)
    assert plan is not None and len(plan.steps) == 1
    assert plan.steps[0].input == {"file_id": "bbb222", "target_format": "md"}
    _validator().validate(plan)


def test_convert_one_unresolvable_asks_counter_question() -> None:
    # Unresolvable file hint / no snapshot / no format: ask instead of
    # converting the wrong artifact (was: None -> L3 ReAct).
    hint = RouterResult(
        intent=Intent.CONVERT_ONE, confidence=0.9, routed_by="llm",
        file_hint="missing", target_format="md",
    )
    _assert_asks_counter_question(build("convert missing to md", hint, _SNAPSHOT))
    no_snapshot = RouterResult(
        intent=Intent.CONVERT_ONE, confidence=0.9, routed_by="llm",
        file_hint="faultbook", target_format="md",
    )
    _assert_asks_counter_question(
        build("convert faultbook to md", no_snapshot, None)
    )
    no_format = RouterResult(
        intent=Intent.CONVERT_ONE, confidence=0.9, routed_by="llm",
        file_hint="faultbook",
    )
    _assert_asks_counter_question(build("convert faultbook", no_format, _SNAPSHOT))


def test_convert_one_ambiguous_match_asks_counter_question() -> None:
    snapshot = "2 file(s): report-a.pdf [ready] id=1; report-b.pdf [ready] id=2"
    route = RouterResult(
        intent=Intent.CONVERT_ONE, confidence=0.9, routed_by="llm",
        file_hint="report", target_format="md",
    )
    plan = build("convert report to md", route, snapshot)
    _assert_asks_counter_question(plan)
    # The question names both candidates so the user can pick one.
    assert "report-a.pdf" in plan.steps[0].input["message"]


def test_convert_one_skips_unready_files() -> None:
    snapshot = "1 file(s): big.pdf [processing] id=zzz"
    route = RouterResult(
        intent=Intent.CONVERT_ONE, confidence=0.9, routed_by="llm",
        file_hint="big", target_format="pdf",
    )
    # Not ready -> nothing to convert; ask instead of a guaranteed failure.
    plan = build("convert big to pdf", route, snapshot)
    _assert_asks_counter_question(plan)
    assert "none is ready" in plan.steps[0].input["message"]


def test_summarize_plot_goes_to_react() -> None:
    # Plot labels are content-derived: no fixed shape may invent them.
    route = RouterResult(
        intent=Intent.SUMMARIZE_PLOT, confidence=0.9,
        routed_by="llm",
    )
    assert build("summarize and plot", route) is None


def test_summarize_plot_empty_corpus_goes_to_react() -> None:
    # Trace 27dcf635: labels/series are model-derived even with no docs
    # (placeholders carry whole text, so no deterministic shape can wire
    # them) — L3 ReAct owns this path, not a builder.
    for intent in (Intent.SUMMARIZE_PLOT, Intent.PLOT_STANDALONE):
        route = RouterResult(
            intent=intent, confidence=0.9, routed_by="llm",
        )
        assert build("plot gdp", route, "(no documents)") is None


def test_deterministic_intents_all_dispatched() -> None:
    from rip_maf.orchestration.intents import DETERMINISTIC_INTENTS

    assert Intent.SUMMARIZE_PLOT not in DETERMINISTIC_INTENTS
    assert {
        Intent.CHAT, Intent.KNOWLEDGE_QA, Intent.CODE, Intent.QA_SINGLE,
        Intent.COMPARE_MULTI, Intent.SUMMARIZE, Intent.QUIZ, Intent.REPORT,
        Intent.CONVERT_ONE, Intent.CONVERT_ALL, Intent.CONVERT_AMBIGUOUS,
    } <= DETERMINISTIC_INTENTS


# --- ADR-035: the non-doc and report builders ---------------------------


def test_knowledge_qa_is_one_reasoning_step() -> None:
    # The parametric home: never rag.query, never a ReAct hop.
    route = RouterResult(
        intent=Intent.KNOWLEDGE_QA, confidence=0.9, routed_by="llm",
    )
    plan = build("What is QLoRA?", route, _SNAPSHOT)
    assert plan is not None and len(plan.steps) == 1
    assert plan.steps[0].agent_id == "reasoning"
    assert plan.steps[0].input == {"message": "What is QLoRA?"}
    _validator().validate(plan)


def test_knowledge_qa_ignores_the_corpus() -> None:
    # Same plan with and without documents — a non-doc intent cannot be
    # pulled into retrieval by a populated corpus.
    route = RouterResult(
        intent=Intent.KNOWLEDGE_QA, confidence=0.9, routed_by="llm",
    )
    with_docs = build("What is QLoRA?", route, _SNAPSHOT)
    without = build("What is QLoRA?", route, "(no documents)")
    assert [s.tool_id for s in with_docs.steps] == [None]
    assert [s.tool_id for s in without.steps] == [None]


def test_code_is_one_coding_step() -> None:
    from rip_maf.agents.coding import CodingAgent

    route = RouterResult(intent=Intent.CODE, confidence=0.9, routed_by="llm")
    plan = build("write a python quicksort", route, "(no documents)")
    assert plan is not None and len(plan.steps) == 1
    assert plan.steps[0].agent_id == "coding"
    assert plan.steps[0].tool_id is None
    assert plan.steps[0].input == {"message": "write a python quicksort"}
    _validator(CodingAgent(_FakeProvider())).validate(plan)


def test_report_routes_through_an_answer_step() -> None:
    from rip_maf.orchestration.builders import build_report

    plan = build_report("write a report on both docs", _SNAPSHOT)
    # 2 shards + writer + doc.generate
    assert len(plan.steps) == 4
    assert [s.tool_id for s in plan.steps[:2]] == ["rag.query", "rag.query"]
    assert all(
        (s.expected_output_type or "").lower() == "chunks" for s in plan.steps[:2]
    )
    writer = plan.steps[2]
    assert writer.agent_id == "reasoning"
    assert (writer.expected_output_type or "").lower() == "answer"
    assert writer.depends_on == ["1", "2"]
    assert "{{1}}" in writer.input["message"] and "{{2}}" in writer.input["message"]
    report = plan.steps[3]
    assert report.tool_id == "doc.generate"
    assert report.depends_on == ["3"]
    assert report.input["title"] == "write a report on both docs"
    # The report body carries the writer's output — ADR-027 grounding is
    # structural (the validator requires an upstream answer step).
    assert report.input["sections"] == [
        {"heading": "Report", "body": "{{3}}"}
    ]
    _validator().validate(plan)


def test_report_title_is_derived_and_bounded() -> None:
    from rip_maf.orchestration.builders import _report_title, build_report

    assert _report_title("  multi\nline   request  ") == "multi line request"
    long_request = "x" * 400
    assert len(_report_title(long_request)) == 90
    assert _report_title("") == "Report"
    plan = build_report("report please", _SNAPSHOT)
    _validator().validate(plan)


def test_report_without_snapshot_uses_one_global_shard() -> None:
    from rip_maf.orchestration.builders import build_report

    plan = build_report("report please", None)
    assert len(plan.steps) == 3  # 1 shard + writer + doc.generate
    assert plan.steps[0].input.get("file_id") is None
    _validator().validate(plan)


def test_report_empty_corpus_asks_to_upload() -> None:
    route = RouterResult(intent=Intent.REPORT, confidence=0.9, routed_by="llm")
    plan = build("write a report", route, "(no documents)")
    assert plan is not None and len(plan.steps) == 1
    assert (plan.steps[0].expected_output_type or "").lower() == "clarification"
    _validator().validate(plan)


def test_report_too_many_files_falls_to_react() -> None:
    snapshot = "; ".join(f"f{i}.pdf [ready] id=id{i}" for i in range(6))
    route = RouterResult(intent=Intent.REPORT, confidence=0.9, routed_by="llm")
    assert build("write a report", route, snapshot) is None


def test_convert_ambiguous_names_the_candidates() -> None:
    route = RouterResult(
        intent=Intent.CONVERT_AMBIGUOUS, confidence=0.9, routed_by="llm",
    )
    plan = build("convert it", route, _SNAPSHOT)
    assert plan is not None and len(plan.steps) == 1
    message = plan.steps[0].input["message"]
    assert "aaa111" not in message  # ids are never asked for
    assert "Faultbook.pdf" in message and "Fire Report.pdf" in message
    assert "file(s)" not in message  # snapshot prefix stripped for display
    assert "md, docx or pdf" in message
    _validator().validate(plan)


def test_convert_ambiguous_with_single_ready_file() -> None:
    snapshot = "1 file(s): only.pdf [ready] id=one"
    route = RouterResult(
        intent=Intent.CONVERT_AMBIGUOUS, confidence=0.9, routed_by="llm",
    )
    plan = build("convert this", route, snapshot)
    assert "one ready file (only.pdf)" in plan.steps[0].input["message"]
    _validator().validate(plan)


def _qa_route(queries=None) -> RouterResult:
    return RouterResult(
        intent=Intent.QA_SINGLE,
        confidence=0.95, routed_by="llm",
    )


def test_qa_single_empty_corpus_answers_generally() -> None:
    # Trace ea48cb30: "What is QLoRA?" on "(no documents)" must not emit
    # rag.query — a single general-answer reasoning step carrying the
    # request verbatim (no document-grounding wrapper).
    plan = build("What is QLoRA?", _qa_route([]), "(no documents)")
    assert plan is not None and len(plan.steps) == 1
    assert plan.steps[0].agent_id == "reasoning"
    assert plan.steps[0].tool_id is None
    assert (plan.steps[0].expected_output_type or "").lower() == "answer"
    assert plan.steps[0].input == {"message": "What is QLoRA?"}
    _validator().validate(plan)


def test_qa_single_processing_corpus_asks_to_wait() -> None:
    plan = build(
        "What is QLoRA?", _qa_route([]),
        "1 file(s): big.pdf [processing] id=zzz",
    )
    assert plan is not None and len(plan.steps) == 1
    assert (plan.steps[0].expected_output_type or "").lower() == "clarification"
    _validator().validate(plan)


def test_qa_single_unknown_snapshot_still_retrieves() -> None:
    # Snapshot None (DB failure) keeps the retrieval path: the database,
    # not the snapshot, is ground truth.
    plan = build("What is QLoRA?", _qa_route([]), None)
    assert plan is not None and len(plan.steps) == 2
    assert plan.steps[0].tool_id == "rag.query"
    assert plan.steps[0].input["top_k"] == 4
    assert plan.steps[0].input["mode"] == "specific"
    _validator().validate(plan)


def test_doc_intents_empty_corpus_yield_clarification() -> None:
    for intent in (Intent.SUMMARIZE, Intent.COMPARE_MULTI, Intent.QUIZ):
        route = RouterResult(
            intent=intent, confidence=0.9, routed_by="llm",
        )
        plan = build("summarize/compare/quiz with no docs", route, "(no documents)")
        assert plan is not None and len(plan.steps) == 1, intent
        assert (plan.steps[0].expected_output_type or "").lower() == "clarification"
        _validator().validate(plan)


def test_corpus_state_tristate() -> None:
    from rip_maf.orchestration.builders import _corpus_state

    assert _corpus_state(None) == "unknown"
    assert _corpus_state("(no documents)") == "empty"
    assert _corpus_state("1 file(s): big.pdf [processing] id=zzz") == "processing"
    assert _corpus_state(_SNAPSHOT) == "ready"


def test_react_prompt_carries_zero_file_plot_rule() -> None:
    # Trace 27dcf635: a plot with no docs and no user numbers must recall
    # figures parametrically via reasoning before plot.chart — L3 ReAct
    # owns this path (mega-prompt removed).
    from rip_maf.orchestration.react import run_react

    seen: list = []

    class _CaptureProvider(_FakeProvider):
        def generate_structured(self, model, messages, schema, *, temperature=0.0):
            seen.append(messages)
            return {
                "thought": "done",
                "executor": "reasoning",
                "input": {},
                "is_final": True,
                "answer": "ok",
            }

    from rip_maf.agents.registry import get_default_agent_registry

    agents = get_default_agent_registry(_FakeProvider())
    tools = get_default_tool_registry()
    run_react(
        "Plot GDP comparison", _CaptureProvider(), agents, tools,
        trace_id="t", corpus_id="nb-1",
        corpus_context="(no documents)",
    )
    system = seen[0][0]["content"]
    assert "no documents" in system.lower() or "(no documents)" in system
    assert "approximate" in system
    assert "plot.chart" in system
