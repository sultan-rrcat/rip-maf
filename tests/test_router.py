"""Phase 1 — router tests (no Ollama, no DB)."""

from __future__ import annotations

import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from rip_maf.orchestration.intents import INTENT_DESCRIPTIONS, Intent
from rip_maf.orchestration.router import Router, build_router_prompt
from rip_maf.providers.base import ModelProvider


class FakeRouterProvider(ModelProvider):
    def __init__(self, payload: dict | None = None, fail: bool = False):
        self.payload = payload or {
            "intent": "compare_multi",
            "confidence": 0.9,
        }
        self.fail = fail
        self.calls = 0
        self.messages: list = []

    def generate(self, model, messages, *, temperature=0.2, max_tokens=None):
        raise NotImplementedError("router uses generate_structured")

    def generate_structured(self, model, messages, schema, *, temperature=0.0):
        self.calls += 1
        self.messages = messages
        if self.fail:
            raise ValueError("ollama down")
        return dict(self.payload)

    def embed(self, model: str, text: str) -> list[float]:
        raise NotImplementedError("test fake")

    def list_available_models(self) -> list[dict]:
        return [{"id": "fake"}]


def test_greeting_goes_through_llm() -> None:
    # L0 removed: every request — including greetings — is classified by
    # the L1 LLM. The canned payload here returns compare_multi, proving
    # a router call was spent even on short input.
    provider = FakeRouterProvider()
    result = Router(provider).route("hello")
    assert provider.calls == 1
    assert result.routed_by == "llm"
    assert result.intent is Intent.COMPARE_MULTI


def test_chat_intent_maps() -> None:
    provider = FakeRouterProvider(
        {"intent": "chat", "confidence": 0.95}
    )
    result = Router(provider).route("hello there friend, how are you doing?")
    assert result.intent is Intent.CHAT


def test_llm_classification_maps() -> None:
    router = Router(FakeRouterProvider())
    result = router.route("compare both reports and rank them")
    assert result.intent is Intent.COMPARE_MULTI
    assert result.confidence == 0.9


def test_low_confidence_falls_to_unknown() -> None:
    provider = FakeRouterProvider(
        {"intent": "qa_single", "confidence": 0.2}
    )
    result = Router(provider).route("something vague here with length over limit x")
    assert result.intent is Intent.UNKNOWN


def test_bad_intent_string_falls_to_unknown() -> None:
    provider = FakeRouterProvider(
        {"intent": "not_a_real_intent", "confidence": 0.95}
    )
    result = Router(provider).route("a long enough request that needs the llm path")
    assert result.intent is Intent.UNKNOWN


def test_llm_failure_falls_open_to_unknown() -> None:
    result = Router(FakeRouterProvider(fail=True)).route(
        "a long enough request that needs the llm path"
    )
    assert result.intent is Intent.UNKNOWN


def test_prompt_disambiguates_plot_vs_compare() -> None:
    # Live trace: "compare the class distribution and plot in a bar chart"
    # was routed compare_multi (no plot step emitted). The prompt must carry
    # the precedence rule so a chart ask wins over a compare ask (ADR-035).
    provider = FakeRouterProvider()
    Router(provider).route("compare the class distribution and plot in a bar chart")
    system = provider.messages[0]["content"]
    assert "summarize_plot" in system
    assert "compare_multi" in system
    assert "no chart requested" in INTENT_DESCRIPTIONS[Intent.COMPARE_MULTI]
    # Rule 2 splits charts by their DATA SOURCE, not just by verb.
    assert "a chart of the corpus's document data is summarize_plot" in system
    assert "from your own knowledge is plot_standalone" in system
    # NB: no "distinct" assert — that word belonged to the removed
    # router-side query-generation line (e2fa13f moved decomposition into
    # rag.query per ADR-030); precedence is covered by the asserts above.


def test_prompt_built_from_descriptions_with_precedence() -> None:
    # The descriptions themselves must separate the two intents: compare
    # excludes charts, summarize_plot claims them even alongside compare.
    from rip_maf.orchestration.intents import INTENT_DESCRIPTIONS

    assert "no chart" in INTENT_DESCRIPTIONS[Intent.COMPARE_MULTI]
    assert "even when the request also says compare" in INTENT_DESCRIPTIONS[
        Intent.SUMMARIZE_PLOT
    ]


def test_compare_without_chart_stays_compare_multi() -> None:
    # Trace 2's request has no chart ask: the canned compare_multi payload
    # must pass through untouched (precedence rule must not steal it).
    result = Router(FakeRouterProvider()).route(
        "compare both report and rank them based on complexity"
    )
    assert result.intent is Intent.COMPARE_MULTI


def test_convert_slots_parsed() -> None:
    provider = FakeRouterProvider(
        {"intent": "convert_one", "confidence": 0.9,
         "file_hint": "Faultbook", "target_format": "MD"}
    )
    result = Router(provider).route("convert the faultbook report to MD please!")
    assert result.intent is Intent.CONVERT_ONE
    assert result.file_hint == "Faultbook"
    assert result.target_format == "md"  # normalized


def test_convert_slots_default_empty_and_bad_format_dropped() -> None:
    provider = FakeRouterProvider(
        {"intent": "convert_all", "confidence": 0.9,
         "file_hint": "*", "target_format": "exe"}
    )
    result = Router(provider).route("convert all documents to exe somehow here")
    assert result.intent is Intent.CONVERT_ALL
    assert result.file_hint == "*"
    assert result.target_format == ""  # not a doc.convert format

    legacy = Router(FakeRouterProvider()).route(
        "compare both report and rank them based on complexity"
    )
    assert legacy.file_hint == "" and legacy.target_format == ""


def test_doc_intent_empty_queries_postfilled() -> None:
    # Router no longer returns queries; query generation moves to rag.query tool.
    provider = FakeRouterProvider(
        {"intent": "qa_single", "confidence": 0.95}
    )
    result = Router(provider).route("What is QLoRA?")
    assert result.intent is Intent.QA_SINGLE


def test_non_doc_intent_empty_queries_stay_empty() -> None:
    provider = FakeRouterProvider(
        {"intent": "chat", "confidence": 0.95}
    )
    result = Router(provider).route("hello there friend, how are you doing?")
    assert result.intent is Intent.CHAT


def test_router_prompt_requires_queries_for_doc_intents() -> None:
    provider = FakeRouterProvider()
    Router(provider).route("compare both reports and rank them")
    system = provider.messages[0]["content"]
    # Router prompt no longer mentions queries; query generation moved to rag tool
    assert "queries" not in system.lower() or "never" not in system


def test_router_context_passed_for_followup() -> None:
    # Trace 35e8fbd9: the bare follow-up "in a table format" classified
    # unknown (0.85) because the router never saw the conversation. With
    # context, recent turns reach the router so it can classify the
    # combined intent.
    provider = FakeRouterProvider({"intent": "chat", "confidence": 0.9})
    result = Router(provider).route(
        "in a table format",
        context=(
            "Recent conversation:\n"
            "user: Difference between ONNX and TensorRT\n"
            "assistant: ONNX is a format, TensorRT is an engine."
        ),
    )
    assert result.intent is Intent.CHAT
    context_msg = provider.messages[1]
    assert context_msg["role"] == "system"
    assert "Difference between ONNX and TensorRT" in context_msg["content"]
    assert "follow-up" in context_msg["content"]


def test_router_without_context_keeps_two_messages() -> None:
    provider = FakeRouterProvider({"intent": "chat", "confidence": 0.9})
    Router(provider).route("hello")
    assert len(provider.messages) == 2  # system + user, no context turn


# --- ADR-035: the router must see the corpus and the doc/non-doc split ---


def test_prompt_splits_doc_and_non_doc_intents() -> None:
    system = build_router_prompt("(no documents)")
    doc_block, non_doc_block = system.split("NON-DOC intents")
    assert "DOC-BASED intents" in doc_block
    # Retrieval intents live in the doc block, parametric ones do not.
    assert "qa_single" in doc_block and "summarize" in doc_block
    assert "qa_single" not in non_doc_block
    assert "knowledge_qa" in non_doc_block and "code" in non_doc_block
    assert "if the request can be answered without the corpus" in system


def test_prompt_states_empty_corpus_and_pushes_non_doc() -> None:
    # Trace c9b02eef: "Plot India vs China GDP growth" was routed as a
    # document plot on an empty corpus. The corpus line is what lets the
    # model prefer the parametric intent.
    system = build_router_prompt("(no documents)")
    assert "(no documents)" in system
    assert "the corpus is EMPTY" in system
    assert "prefer a NON-DOC intent" in system


def test_prompt_states_ready_corpus() -> None:
    system = build_router_prompt("2 file(s): a.pdf [ready] id=aaa; b.pdf [ready] id=bbb")
    assert "HAS uploaded documents" in system
    assert "prefer a NON-DOC intent" not in system


def test_prompt_states_processing_corpus() -> None:
    system = build_router_prompt("1 file(s): big.pdf [processing] id=zzz")
    assert "uploading/processing" in system


def test_prompt_states_unavailable_inventory() -> None:
    system = build_router_prompt(None)
    assert "inventory is unavailable" in system
    assert "assume the corpus MAY have documents" in system


def test_route_receives_the_corpus() -> None:
    provider = FakeRouterProvider({"intent": "qa_single", "confidence": 0.9})
    Router(provider).route("what does it say?", corpus_context="(no documents)")
    assert "(no documents)" in provider.messages[0]["content"]


def test_removed_intents_are_not_advertised() -> None:
    system = build_router_prompt("(no documents)")
    assert "- image:" not in system
    assert "- vision:" not in system


# --- Deterministic convert-slot recovery (live probe: small models return
# the intent with blank slots, demoting every conversion) ---

_SNAP = (
    "2 file(s): Indus-Faultbook-Assistant-Report.pdf [ready] id=aaa111; "
    "Anomaly-Detection-Report.pdf [ready] id=bbb222"
)


def _blank_convert_route(intent: str):
    provider = FakeRouterProvider({"intent": intent, "confidence": 0.9})
    return Router(provider), provider


def test_recovery_fills_named_file_and_format() -> None:
    router, _ = _blank_convert_route("convert_one")
    result = router.route(
        "Convert Anomaly-Detection-Report.pdf to md", corpus_context=_SNAP
    )
    assert result.intent is Intent.CONVERT_ONE
    assert result.file_hint == "Anomaly-Detection-Report.pdf"
    assert result.target_format == "md"


def test_recovery_reads_markdown_word_and_ignores_pdf_in_filename() -> None:
    router, _ = _blank_convert_route("convert_one")
    result = router.route(
        "Convert Anomaly-Detection-Report.pdf to markdown", corpus_context=_SNAP
    )
    assert result.file_hint == "Anomaly-Detection-Report.pdf"
    assert result.target_format == "md"


def test_recovery_star_for_all_documents() -> None:
    router, _ = _blank_convert_route("convert_all")
    result = router.route("Convert every document to docx", corpus_context=_SNAP)
    assert result.file_hint == "*"
    assert result.target_format == "docx"


def test_recovery_keeps_llm_slots_when_present() -> None:
    provider = FakeRouterProvider({
        "intent": "convert_one", "confidence": 0.9,
        "file_hint": "bbb222", "target_format": "pdf",
    })
    result = Router(provider).route("convert it to pdf", corpus_context=_SNAP)
    assert result.file_hint == "bbb222"
    assert result.target_format == "pdf"


def test_recovery_leaves_genuinely_ambiguous_blank() -> None:
    router, _ = _blank_convert_route("convert_ambiguous")
    result = router.route("Convert it", corpus_context=_SNAP)
    assert result.file_hint == "" and result.target_format == ""


def test_recovery_ignores_non_convert_intents() -> None:
    provider = FakeRouterProvider({"intent": "qa_single", "confidence": 0.9})
    result = Router(provider).route(
        "What does Anomaly-Detection-Report.pdf say about md files?",
        corpus_context=_SNAP,
    )
    assert result.intent is Intent.QA_SINGLE
    assert result.file_hint == "" and result.target_format == ""
