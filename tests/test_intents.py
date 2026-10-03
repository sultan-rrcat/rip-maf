"""Phase 0 — intent taxonomy tests (no Ollama, no DB)."""

from __future__ import annotations

import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from rip_maf.orchestration.intents import (
    DETERMINISTIC_INTENTS,
    DOC_INTENTS,
    INTENT_DESCRIPTIONS,
    NON_DOC_INTENTS,
    REACT_ONLY_INTENTS,
    ROUTER_CONFIDENCE_THRESHOLD,
    Intent,
)


def test_all_intents_have_descriptions() -> None:
    for intent in Intent:
        assert INTENT_DESCRIPTIONS[intent].strip(), intent


def test_threshold_in_range() -> None:
    assert 0.0 < ROUTER_CONFIDENCE_THRESHOLD < 1.0


def test_compare_multi_is_deterministic() -> None:
    assert Intent.COMPARE_MULTI in DETERMINISTIC_INTENTS
    assert Intent.UNKNOWN not in DETERMINISTIC_INTENTS


# --- ADR-035: doc vs non-doc segregation -------------------------------


def test_buckets_partition_every_intent() -> None:
    """Every intent belongs to exactly one bucket, UNKNOWN excepted."""
    assert DOC_INTENTS | NON_DOC_INTENTS == set(Intent) - {Intent.UNKNOWN}
    assert not (DOC_INTENTS & NON_DOC_INTENTS)


def test_unknown_is_in_neither_bucket() -> None:
    assert Intent.UNKNOWN not in DOC_INTENTS
    assert Intent.UNKNOWN not in NON_DOC_INTENTS


def test_doc_intents_are_the_retrieval_and_file_ops() -> None:
    assert {Intent.QA_SINGLE, Intent.SUMMARIZE, Intent.QUIZ,
            Intent.REPORT, Intent.CONVERT_ONE} <= DOC_INTENTS
    # Non-doc intents must never be reachable through retrieval.
    assert not (NON_DOC_INTENTS & {Intent.QA_SINGLE, Intent.SUMMARIZE})


def test_knowledge_qa_is_a_non_doc_intent() -> None:
    # The parametric home that keeps "What is QLoRA?" out of rag.query.
    assert Intent.KNOWLEDGE_QA in NON_DOC_INTENTS
    assert Intent.KNOWLEDGE_QA in DETERMINISTIC_INTENTS


def test_plot_intents_are_react_only() -> None:
    assert REACT_ONLY_INTENTS == {Intent.SUMMARIZE_PLOT, Intent.PLOT_STANDALONE}
    assert not (REACT_ONLY_INTENTS & DETERMINISTIC_INTENTS)


def test_deterministic_is_derived_from_the_buckets() -> None:
    # Derived, not hand-listed: a new intent is deterministic by default
    # and must be declared REACT_ONLY to earn a ReAct hop.
    assert DETERMINISTIC_INTENTS == (DOC_INTENTS | NON_DOC_INTENTS) - REACT_ONLY_INTENTS


def test_dead_intents_removed() -> None:
    # image.generate needs an unconfigured model (ollama_image_model="")
    # and the vision agent is text-only with no image upload path — both
    # were guaranteed failures advertised to the router.
    assert "image" not in {i.value for i in Intent}
    assert "vision" not in {i.value for i in Intent}
