"""Retrieval diversity: per-file round-robin interleave (pure unit tests).

No DB, no models — covers the helper only. Live replay of the trace-2
queries (2 ready files, one-doc collapse) was verified manually against
the office backend before wiring it in.
"""

from __future__ import annotations

import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from rip_maf.rag.vector_rag import (
    _interleave_by_source,
    _matches_overview_section,
    _rerank_text,
    _section_prefix,
    _select_top,
    _stratify_overview,
)


def _chunk(source: str, text: str) -> dict:
    return {"text": text, "metadata": {"source": source}}


def test_single_file_unchanged() -> None:
    items = [_chunk("a.pdf", f"t{i}") for i in range(3)]
    assert _interleave_by_source(items) == items


def test_empty_unchanged() -> None:
    assert _interleave_by_source([]) == []


def test_round_robin_across_files() -> None:
    items = [
        _chunk("a.pdf", "a1"),
        _chunk("a.pdf", "a2"),
        _chunk("a.pdf", "a3"),
        _chunk("b.pdf", "b1"),
    ]
    merged = _interleave_by_source(items)
    assert [c["text"] for c in merged] == ["a1", "b1", "a2", "a3"]


def test_within_file_order_preserved() -> None:
    items = [
        _chunk("b.pdf", "b1"),
        _chunk("a.pdf", "a1"),
        _chunk("b.pdf", "b2"),
        _chunk("a.pdf", "a2"),
    ]
    merged = _interleave_by_source(items)
    assert [c["text"] for c in merged] == ["b1", "a1", "b2", "a2"]


def test_missing_metadata_groups_as_unknown() -> None:
    items = [{"text": "x"}, _chunk("a.pdf", "a1")]
    merged = _interleave_by_source(items)
    assert [c["text"] for c in merged] == ["x", "a1"]


def _hchunk(source: str, h1: str, text: str, idx: int, score: float = 0.5) -> dict:
    return {
        "text": text,
        "metadata": {"source": source, "H1": h1},
        "chunk_index": idx,
        "rerank_score": score,
    }


def test_overview_section_match() -> None:
    assert _matches_overview_section({"H1": "Introduction"}) is True
    assert _matches_overview_section({"H2": "Summary of results"}) is True
    assert _matches_overview_section({"H1": "Random Methods"}) is False
    assert _matches_overview_section({}) is False


def test_stratify_one_per_h1_in_doc_order() -> None:
    items = [
        _hchunk("a.pdf", "Intro", "i1", 0, 0.9),
        _hchunk("a.pdf", "Intro", "i2", 1, 0.8),
        _hchunk("a.pdf", "Methods", "m1", 5, 0.7),
        _hchunk("a.pdf", "Conclusion", "c1", 9, 0.6),
    ]
    out = _stratify_overview(items, 3)
    assert [c["text"] for c in out] == ["i1", "m1", "c1"]


def test_stratify_single_section_truncates() -> None:
    items = [_hchunk("a.pdf", "Intro", f"t{i}", i) for i in range(5)]
    assert len(_stratify_overview(items, 4)) == 4


def test_stratify_empty() -> None:
    assert _stratify_overview([], 4) == []


def test_section_prefix_joins_headers() -> None:
    assert (
        _section_prefix({"H1": "Sys Req", "H2": "2. System", "H3": "2.1 Functional"})
        == "Section: Sys Req > 2. System > 2.1 Functional"
    )
    assert _section_prefix({"H2": "Abstract"}) == "Section: Abstract"
    assert _section_prefix({}) == ""
    assert _section_prefix(None) == ""


def test_rerank_text_prefixed_and_bare() -> None:
    item = {"text": "body", "metadata": {"H3": "2.1 Functional Requirements"}}
    assert _rerank_text(item) == "Section: 2.1 Functional Requirements\nbody"
    assert _rerank_text({"text": "body", "metadata": {}}) == "body"
    assert _rerank_text({"text": "body"}) == "body"


def _scored(text: str, score: float) -> dict:
    return {"text": text, "metadata": {}, "rerank_score": score}


def test_select_top_prefers_above_threshold_then_tops_up() -> None:
    # Trace shape: 4 candidates, 3 above 0.05 — the answer step must still
    # receive top_k=4, with the below-threshold best filling the last slot.
    items = [_scored("a", 0.58), _scored("b", 0.45), _scored("c", 0.058), _scored("d", 0.009)]
    out = _select_top(items, 4, 0.05)
    assert [c["text"] for c in out] == ["a", "b", "c", "d"]


def test_select_top_none_passing_falls_back_in_order() -> None:
    items = [_scored("a", 0.01), _scored("b", 0.02)]
    out = _select_top(items, 4, 0.05)
    assert [c["text"] for c in out] == ["a", "b"]


def test_select_top_truncates_to_top_k() -> None:
    items = [_scored(f"t{i}", 0.9 - i * 0.1) for i in range(6)]
    out = _select_top(items, 4, 0.05)
    assert [c["text"] for c in out] == ["t0", "t1", "t2", "t3"]
