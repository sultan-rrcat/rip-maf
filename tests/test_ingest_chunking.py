"""Ingest chunking: headings must survive page boundaries (pure unit tests).

No DB, no models, no loaders — feeds synthetic page Documents straight into
`RagPipeline.chunk_documents` via `__new__` (skips loading BGE weights).

Regression guard for ADR-034: loaders return one Document per page, so a
Markdown heading that lands at the end of a page has no body until the next
page. Splitting each page independently made MarkdownHeaderTextSplitter emit
no chunk for such a trailing heading — the heading was silently discarded and
its body was stored with NO header metadata. Live case: an 18-page lab report
whose "Lab 4: Network and Information Lab" heading sat at the end of page 10,
leaving "Lab 4" present nowhere in `embeddings`.
"""

from __future__ import annotations

import os
import re
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from langchain_core.documents import Document

from rip_maf.rag.pipeline import RagPipeline


def chunk(pages: list[str], file_name: str = "f.pdf") -> list[Document]:
    """Run the real chunker over per-page Documents without loading weights."""
    pipe = RagPipeline.__new__(RagPipeline)
    return pipe.chunk_documents(
        [Document(page_content=p) for p in pages], file_name
    )


def titles(chunks: list[Document]) -> list[str]:
    return [c.metadata["H1"] for c in chunks if c.metadata.get("H1")]


# --- the live regression -------------------------------------------------


def test_trailing_heading_survives_page_boundary() -> None:
    """Heading alone at the end of a page must attach to the next page's body."""
    pages = [
        "# Lab 3: Scientific Computing Lab\n\nHPC clusters and storage.",
        "# Lab 4: Network and Information Lab\n",
        "Deals with networking infrastructure. 1 Gbps. 2FA based auth.",
    ]
    chunks = chunk(pages)
    assert titles(chunks) == [
        "Lab 3: Scientific Computing Lab",
        "Lab 4: Network and Information Lab",
    ]
    lab4 = [c for c in chunks if c.metadata.get("H1") == "Lab 4: Network and Information Lab"]
    assert len(lab4) == 1
    # The body the heading introduces is attributed to it.
    assert "networking infrastructure" in lab4[0].page_content


def test_lab4_label_is_recoverable_from_chunks() -> None:
    """Every heading in the source must survive into some chunk's metadata.

    This is the property that actually broke: the answer could not name the
    section because its label was gone from the database.
    """
    pages = [
        "# Report\n\nTitle page.",
        "# Lab 1: Alpha\n\nAlpha work.",
        "# Lab 2: Beta\n",
        "Beta work on page three.",
        "# Lab 3: Gamma\n",
        "Gamma work.",
        "# Lab 4: Delta\n",
        "Delta work.",
        "# Lab 5: Epsilon\n",
        "Epsilon work.",
    ]
    chunks = chunk(pages)
    joined = " ".join(
        [c.page_content for c in chunks] + [" ".join(c.metadata.values()) for c in chunks]
    )
    for n in (1, 2, 3, 4, 5):
        assert re.search(rf"lab\s*{n}\b", joined, re.IGNORECASE), f"Lab {n} lost"


def test_no_bodyless_chunk_is_emitted() -> None:
    """A heading with no body anywhere yields no empty chunk."""
    pages = ["# Only A Heading\n", "# With Body\n\nSome text here."]
    chunks = chunk(pages)
    assert all(c.page_content.strip() for c in chunks)


def test_multi_page_section_keeps_its_h1() -> None:
    """A section spanning pages without sub-headings becomes one labelled chunk.

    Joining is what makes this possible: per-page splitting produced three
    chunks and only the first carried the H1.
    """
    pages = [
        "# Lab 4: Network\n\nFirst part.",
        "Second part, same section.",
        "Third part, still the same section.",
        "# Lab 5: Security\n\nNew section.",
    ]
    chunks = chunk(pages)
    net = [c for c in chunks if c.metadata.get("H1") == "Lab 4: Network"]
    assert len(net) == 1
    body = net[0].page_content
    assert "First part." in body
    assert "Second part, same section." in body
    assert "Third part, still the same section." in body
    # No chunk left unlabelled.
    assert all(c.metadata.get("H1") for c in chunks)


def test_multi_page_section_with_subheadings_keeps_h1_on_each() -> None:
    """The Lab 4 shape: H2 subsections spread over pages, all under one H1."""
    pages = [
        "# Lab 4: Network\n\nFirst part.",
        "## Q1\n\nAnswer one.",
        "## Q2\n\nAnswer two.",
        "# Lab 5: Security\n\nNew section.",
    ]
    chunks = chunk(pages)
    net = [c for c in chunks if c.metadata.get("H1") == "Lab 4: Network"]
    assert len(net) == 3
    assert all(c.metadata["H1"] == "Lab 4: Network" for c in net)
    assert [c.metadata.get("H2") for c in net] == [None, "Q1", "Q2"]


# --- behaviour preserved from the old per-page path ----------------------


def test_single_page_document_is_unaffected() -> None:
    pages = ["# A\n\nalpha\n\n## A1\n\nalpha one\n\n# B\n\nbeta"]
    chunks = chunk(pages)
    assert titles(chunks) == ["A", "A", "B"]


def test_h2_and_h3_are_still_captured() -> None:
    pages = ["# Top\n\n## Mid\n\nmid body\n\n### Leaf\n\nleaf body"]
    chunks = chunk(pages)
    assert any(c.metadata.get("H2") == "Mid" for c in chunks)
    assert any(c.metadata.get("H3") == "Leaf" for c in chunks)


def test_source_metadata_is_the_file_name() -> None:
    chunks = chunk(["# A\n\nbody"], file_name="report.pdf")
    assert all(c.metadata["source"] == "report.pdf" for c in chunks)


def test_blank_page_between_sections_is_tolerated() -> None:
    pages = ["# A\n\nalpha", "", "# B\n\nbeta"]
    chunks = chunk(pages)
    assert titles(chunks) == ["A", "B"]


def test_document_without_headers_yields_unlabelled_chunk() -> None:
    """Headerless input still chunks — just with no H1 (unchanged behaviour)."""
    chunks = chunk(["just some prose, no headings at all"])
    assert len(chunks) == 1
    assert "H1" not in chunks[0].metadata
    assert "prose" in chunks[0].page_content


def test_empty_document_list_yields_no_chunks() -> None:
    assert chunk([]) == []


def test_joining_does_not_duplicate_content() -> None:
    """Joining pages must not repeat body text across the new seam."""
    pages = ["# A\n\nalpha body", "beta body continues", "# B\n\nbeta body"]
    chunks = chunk(pages)
    all_text = " ".join(c.page_content for c in chunks)
    assert all_text.count("beta body continues") == 1
    assert all_text.count("alpha body") == 1


# --- store is idempotent (re-processing must replace, not append) --------


class FakeCur:
    rowcount = 3  # pretend three stale rows were removed

    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple]] = []
        self._rows = [["11111111-1111-1111-1111-111111111111"]]

    def execute(self, sql, params=()):
        self.executed.append((" ".join(sql.split()), tuple(params)))

    def fetchone(self):
        return self._rows[0]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConn:
    def __init__(self, cur) -> None:
        self._cur = cur

    def cursor(self):
        return self._cur

    def commit(self):
        pass


def _store(chunks, embeddings, monkeypatch, file_id="f-1"):
    from rip_maf.rag import pipeline as pipeline_module

    cur = FakeCur()
    monkeypatch.setattr(
        pipeline_module, "pg_connection", lambda: _ctx(FakeConn(cur))
    )
    pipe = RagPipeline.__new__(RagPipeline)
    ids = pipe.store_chunks_and_embeddings(file_id, chunks, embeddings)
    return ids, cur


class _ctx:
    def __init__(self, conn) -> None:
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *exc):
        return False


def test_store_deletes_previous_rows_before_inserting(monkeypatch) -> None:
    """Re-processing must replace the file's rows, not append to them."""
    docs = [Document(page_content="a", metadata={"source": "f.pdf"})]
    vecs = [([0.0] * 1024)]
    _ids, cur = _store(docs, vecs, monkeypatch)

    stmts = [sql.upper() for sql, _ in cur.executed]
    delete_at = next(
        (i for i, s in enumerate(stmts) if s.startswith("DELETE FROM EMBEDDINGS")),
        None,
    )
    insert_at = next(
        (i for i, s in enumerate(stmts) if s.startswith("INSERT INTO EMBEDDINGS")),
        None,
    )
    assert delete_at is not None, "must delete the file's previous chunks"
    assert insert_at is not None
    assert delete_at < insert_at, "delete must happen before the inserts"


def test_store_delete_is_scoped_to_the_file(monkeypatch) -> None:
    docs = [Document(page_content="a", metadata={"source": "f.pdf"})]
    _ids, cur = _store(docs, [([0.0] * 1024)], monkeypatch, file_id="f-42")
    delete = next(
        (p for s, p in cur.executed if s.upper().startswith("DELETE FROM EMBEDDINGS")),
        None,
    )
    assert delete == ("f-42",)


def test_store_indexes_from_zero(monkeypatch) -> None:
    docs = [
        Document(page_content="a", metadata={"source": "f.pdf"}),
        Document(page_content="b", metadata={"source": "f.pdf"}),
    ]
    _ids, cur = _store(docs, [([0.0] * 1024), ([0.0] * 1024)], monkeypatch)
    inserts = [p for s, p in cur.executed if s.upper().startswith("INSERT INTO")]
    assert [p[1] for p in inserts] == [0, 1]