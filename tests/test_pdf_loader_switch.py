"""Unit tests for RAG_PDF_LOADER switch (no model load, no Java)."""

import os
import sys

import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


def test_default_is_docling():
    from rip_maf.core.config import Settings

    assert Settings.model_fields["rag_pdf_loader"].default == "docling"


def test_normalizes_case_and_whitespace():
    from rip_maf.core.config import Settings

    s = Settings.model_validate({"rag_pdf_loader": "  Docling "})
    assert s.rag_pdf_loader == "docling"


def test_rejects_unknown_loader():
    from pydantic import ValidationError

    from rip_maf.core.config import Settings

    with pytest.raises(ValidationError):
        Settings.model_validate({"rag_pdf_loader": "pdfminer"})


def _pipeline_without_init():
    from rip_maf.rag.pipeline import RagPipeline

    return RagPipeline.__new__(RagPipeline)


def test_docling_primary_order(monkeypatch):
    from rip_maf.rag import pipeline as pipeline_module

    pipe = _pipeline_without_init()
    calls: list[str] = []
    monkeypatch.setattr(pipeline_module.settings, "rag_pdf_loader", "docling")
    monkeypatch.setattr(
        pipe, "_load_with_docling", lambda f: calls.append("docling") or "DOC"
    )
    monkeypatch.setattr(
        pipe, "_load_with_opendataloader", lambda f: calls.append("odl") or "ODL"
    )
    assert pipe.document_loader("f.pdf") == "DOC"
    assert calls == ["docling"]


def test_opendataloader_primary_order(monkeypatch):
    from rip_maf.rag import pipeline as pipeline_module

    pipe = _pipeline_without_init()
    calls: list[str] = []
    monkeypatch.setattr(pipeline_module.settings, "rag_pdf_loader", "opendataloader")
    monkeypatch.setattr(
        pipe, "_load_with_docling", lambda f: calls.append("docling") or "DOC"
    )
    monkeypatch.setattr(
        pipe, "_load_with_opendataloader", lambda f: calls.append("odl") or "ODL"
    )
    assert pipe.document_loader("f.pdf") == "ODL"
    assert calls == ["odl"]


def test_fallback_on_primary_failure(monkeypatch):
    from rip_maf.rag import pipeline as pipeline_module

    pipe = _pipeline_without_init()
    calls: list[str] = []
    monkeypatch.setattr(pipeline_module.settings, "rag_pdf_loader", "opendataloader")

    def boom(_f):
        calls.append("odl")
        raise RuntimeError("no java")

    monkeypatch.setattr(pipe, "_load_with_opendataloader", boom)
    monkeypatch.setattr(
        pipe, "_load_with_docling", lambda f: calls.append("docling") or "DOC"
    )
    assert pipe.document_loader("f.pdf") == "DOC"
    assert calls == ["odl", "docling"]


def test_both_fail_raises(monkeypatch):
    from rip_maf.rag import pipeline as pipeline_module

    pipe = _pipeline_without_init()
    monkeypatch.setattr(pipeline_module.settings, "rag_pdf_loader", "docling")
    monkeypatch.setattr(
        pipe, "_load_with_docling", lambda f: (_ for _ in ()).throw(RuntimeError("a"))
    )
    monkeypatch.setattr(
        pipe,
        "_load_with_opendataloader",
        lambda f: (_ for _ in ()).throw(RuntimeError("b")),
    )
    with pytest.raises(RuntimeError):
        pipe.document_loader("f.pdf")
