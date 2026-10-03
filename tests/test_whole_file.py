"""Whole-file RAG shortcut: budget math, row mapping, and fall-back gates.

Pure unit tests — no DB, no models. `retrieve_whole_file` is exercised through
a fake cursor/connection so the SQL gating is covered without Postgres or the
BGE weights (the real DB path is covered by test_app.py::TestRetrieval).
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from rip_maf.core.config import settings
from rip_maf.rag import vector_rag as vr
from rip_maf.rag.vector_rag import (
    _bind_scope,
    _scope_sql,
    _whole_file_budget_chars,
    _whole_file_rows_to_results,
)

# --- fakes ---------------------------------------------------------------


class FakeCursor:
    """Answers the SUM probe, then the row fetch; records both statements."""

    def __init__(self, total_chars, chunk_count, rows=None, raise_on=None):
        self._answers = [(total_chars, chunk_count), rows or []]
        self._raise_on = raise_on
        self.statements: list[tuple[str, tuple]] = []

    def execute(self, sql, params=()):
        self.statements.append((" ".join(sql.split()), tuple(params)))
        if self._raise_on is not None and self._raise_on in self.statements[-1][0]:
            raise RuntimeError("db down")

    def fetchall(self):
        return self._answers[1]

    def fetchone(self):
        return self._answers[0]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def commit(self):
        pass


@contextmanager
def fake_pg(cursor):
    yield FakeConn(cursor)


def rag_without_weights():
    """A VectorRAG instance without __init__ (skips loading BGE weights)."""
    return object.__new__(vr.VectorRAG)


# --- budget -------------------------------------------------------------


def test_budget_scales_with_window_and_pct(monkeypatch):
    monkeypatch.setattr(settings, "ollama_context_window", 32768)
    monkeypatch.setattr(settings, "rag_whole_file_pct", 0.15)
    # int(32768 * 0.15) = 4915 tokens * 4 chars/token
    assert _whole_file_budget_chars() == 4915 * 4


def test_budget_read_live_so_tests_can_override(monkeypatch):
    monkeypatch.setattr(settings, "ollama_context_window", 1000)
    monkeypatch.setattr(settings, "rag_whole_file_pct", 0.5)
    assert _whole_file_budget_chars() == 500 * 4


def test_zero_pct_disables_shortcut(monkeypatch):
    monkeypatch.setattr(settings, "rag_whole_file_pct", 0.0)
    assert _whole_file_budget_chars() == 0


# --- row mapping --------------------------------------------------------


def test_rows_map_to_canonical_result_shape():
    rows = [
        ("body one", {"source": "a.pdf", "H1": "Intro", "H2": "Scope"}, 0),
        ("body two", {"source": "a.pdf", "H1": "Intro", "H3": "Detail"}, 1),
    ]
    results = _whole_file_rows_to_results(rows)
    assert [r["source"] for r in results] == ["a.pdf", "a.pdf"]
    assert results[0]["section"] == "Intro > Scope"
    assert results[1]["section"] == "Intro > Detail"
    # Section-prefixed content, same as the ranked path feeds the LLM.
    assert results[0]["content"].startswith("Section: Intro > Scope\n")
    assert results[0]["content"].endswith("body one")


def test_rerank_score_is_none_because_nothing_was_reranked():
    results = _whole_file_rows_to_results([("t", {"source": "a.pdf"}, 0)])
    assert results[0]["rerank_score"] is None


def test_headerless_metadata_yields_empty_section():
    results = _whole_file_rows_to_results([("t", {}, 0)])
    assert results[0]["section"] == ""
    assert results[0]["content"] == "t"
    assert results[0]["source"] == "unknown"


def test_non_dict_metadata_is_tolerated():
    results = _whole_file_rows_to_results([("t", None, 0)])
    assert results[0]["source"] == "unknown"
    assert results[0]["section"] == ""


# --- scope SQL ----------------------------------------------------------


def test_scope_variants_match_retrieve_context_branches():
    assert _scope_sql(None, None) == (
        "file_id IN (SELECT file_id FROM files WHERE corpus_id=%s)",
        ["nb"],
    )
    assert _scope_sql("f1", None) == (
        "file_id IN (SELECT file_id FROM files WHERE corpus_id=%s) AND file_id = %s",
        ["nb", "fid"],
    )
    assert _scope_sql(None, "a.pdf") == (
        (
            "file_id IN (SELECT file_id FROM files"
            " WHERE corpus_id=%s AND file_name ILIKE %s)"
        ),
        ["nb", "fname"],
    )
    where, keys = _scope_sql("f1", "a.pdf")
    assert keys == ["nb", "fid", "nb", "fname"]
    assert where.count("%s") == 4


def test_bind_scope_expands_placeholders_in_order():
    where, keys = _scope_sql("f1", "a.pdf")
    bound = _bind_scope(where, keys, "nb-1", "f1", "a.pdf")
    assert bound == ("nb-1", "f1", "nb-1", "a.pdf")


def test_bind_scope_corpus_only():
    where, keys = _scope_sql(None, None)
    assert _bind_scope(where, keys, "nb-1", None, None) == ("nb-1",)


# --- gating (retrieve_whole_file) ---------------------------------------


def _patch_pg(monkeypatch, cursor):
    monkeypatch.setattr(vr, "pg_connection", lambda: fake_pg(cursor))


def test_fits_returns_every_row_verbatim(monkeypatch):
    monkeypatch.setattr(settings, "ollama_context_window", 32768)
    monkeypatch.setattr(settings, "rag_whole_file_pct", 0.15)
    rows = [("b", {"source": "a.pdf"}, 1), ("a", {"source": "a.pdf"}, 0)]
    cursor = FakeCursor(total_chars=100, chunk_count=2, rows=rows)
    _patch_pg(monkeypatch, cursor)

    out = vr.VectorRAG.retrieve_whole_file(rag_without_weights(), "nb-1")
    # No ranking, no truncation: the DB's row order is the result order, and
    # document order is requested via ORDER BY in the fetch (below).
    assert [r["content"] for r in out] == ["b", "a"]
    assert len(cursor.statements) == 2


def test_fetch_orders_by_chunk_index_for_document_order(monkeypatch):
    """Whole-file reads must arrive in document order, not rank order."""
    monkeypatch.setattr(settings, "ollama_context_window", 32768)
    monkeypatch.setattr(settings, "rag_whole_file_pct", 0.15)
    cursor = FakeCursor(total_chars=10, chunk_count=1, rows=[("t", {}, 0)])
    _patch_pg(monkeypatch, cursor)

    vr.VectorRAG.retrieve_whole_file(rag_without_weights(), "nb-1")
    fetch_sql, _ = cursor.statements[1]
    assert "ORDER BY chunk_index" in fetch_sql


def test_over_budget_returns_none_without_row_fetch(monkeypatch):
    monkeypatch.setattr(settings, "ollama_context_window", 32768)
    monkeypatch.setattr(settings, "rag_whole_file_pct", 0.15)
    cursor = FakeCursor(total_chars=10_000_000, chunk_count=3, rows=[("x", {}, 0)])
    _patch_pg(monkeypatch, cursor)

    out = vr.VectorRAG.retrieve_whole_file(rag_without_weights(), "nb-1")
    assert out is None
    # Probe only — the expensive fetch is never issued.
    assert len(cursor.statements) == 1


def test_over_chunk_cap_returns_none(monkeypatch):
    monkeypatch.setattr(settings, "ollama_context_window", 32768)
    monkeypatch.setattr(settings, "rag_whole_file_pct", 0.15)
    cursor = FakeCursor(
        total_chars=100,
        chunk_count=vr._WHOLE_FILE_MAX_CHUNKS + 1,
        rows=[("x", {}, 0)],
    )
    _patch_pg(monkeypatch, cursor)
    assert vr.VectorRAG.retrieve_whole_file(rag_without_weights(), "nb-1") is None
    assert len(cursor.statements) == 1


def test_exactly_at_budget_is_included(monkeypatch):
    monkeypatch.setattr(settings, "ollama_context_window", 1000)
    monkeypatch.setattr(settings, "rag_whole_file_pct", 0.5)
    # budget = 500 tokens * 4 = 2000 chars; exactly at the limit fits.
    cursor = FakeCursor(total_chars=2000, chunk_count=1, rows=[("t", {}, 0)])
    _patch_pg(monkeypatch, cursor)
    assert vr.VectorRAG.retrieve_whole_file(rag_without_weights(), "nb-1") is not None


def test_empty_corpus_returns_none(monkeypatch):
    monkeypatch.setattr(settings, "ollama_context_window", 32768)
    monkeypatch.setattr(settings, "rag_whole_file_pct", 0.15)
    # Empty scope: SUM -> 0, COUNT -> 0 (NOT NULL — COALESCE guards it).
    cursor = FakeCursor(total_chars=0, chunk_count=0, rows=[])
    _patch_pg(monkeypatch, cursor)
    assert vr.VectorRAG.retrieve_whole_file(rag_without_weights(), "nb-1") is None
    assert len(cursor.statements) == 1


def test_null_sum_is_treated_as_zero(monkeypatch):
    monkeypatch.setattr(settings, "ollama_context_window", 32768)
    monkeypatch.setattr(settings, "rag_whole_file_pct", 0.15)
    cursor = FakeCursor(total_chars=None, chunk_count=None, rows=[])
    _patch_pg(monkeypatch, cursor)
    assert vr.VectorRAG.retrieve_whole_file(rag_without_weights(), "nb-1") is None


def test_db_error_falls_back_instead_of_raising(monkeypatch):
    """An optimization must never fail retrieval."""
    monkeypatch.setattr(settings, "ollama_context_window", 32768)
    monkeypatch.setattr(settings, "rag_whole_file_pct", 0.15)
    cursor = FakeCursor(0, 0, raise_on="SELECT")
    _patch_pg(monkeypatch, cursor)
    assert vr.VectorRAG.retrieve_whole_file(rag_without_weights(), "nb-1") is None


def test_zero_pct_skips_db_entirely(monkeypatch):
    monkeypatch.setattr(settings, "rag_whole_file_pct", 0.0)
    cursor = FakeCursor(1, 1, rows=[("t", {}, 0)])
    _patch_pg(monkeypatch, cursor)
    assert vr.VectorRAG.retrieve_whole_file(rag_without_weights(), "nb-1") is None
    assert cursor.statements == []


def test_file_scope_is_forwarded_to_both_statements(monkeypatch):
    monkeypatch.setattr(settings, "ollama_context_window", 32768)
    monkeypatch.setattr(settings, "rag_whole_file_pct", 0.15)
    cursor = FakeCursor(total_chars=10, chunk_count=1, rows=[("t", {}, 0)])
    _patch_pg(monkeypatch, cursor)

    rag_without_weights().retrieve_whole_file("nb-1", file_id="  f1  ")
    for sql, params in cursor.statements:
        assert "file_id = %s" in sql
        assert params == ("nb-1", "f1")


def test_blank_file_id_is_treated_as_corpus_scope(monkeypatch):
    monkeypatch.setattr(settings, "ollama_context_window", 32768)
    monkeypatch.setattr(settings, "rag_whole_file_pct", 0.15)
    cursor = FakeCursor(total_chars=10, chunk_count=1, rows=[("t", {}, 0)])
    _patch_pg(monkeypatch, cursor)

    rag_without_weights().retrieve_whole_file("nb-1", file_id="   ", file_name="")
    for sql, params in cursor.statements:
        assert "file_id = %s" not in sql
        assert params == ("nb-1",)


def test_probe_selects_sum_of_char_length(monkeypatch):
    monkeypatch.setattr(settings, "ollama_context_window", 32768)
    monkeypatch.setattr(settings, "rag_whole_file_pct", 0.15)
    cursor = FakeCursor(total_chars=10, chunk_count=1, rows=[("t", {}, 0)])
    _patch_pg(monkeypatch, cursor)

    vr.VectorRAG.retrieve_whole_file(rag_without_weights(), "nb-1")
    probe = cursor.statements[0][0]
    assert "SUM(char_length(chunk_text))" in probe
    assert "COUNT(*)" in probe