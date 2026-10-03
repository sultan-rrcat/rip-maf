"""doc.convert tool — exact file-to-file conversion (lossless, deterministic).

Unlike doc.generate (which renders LLM-synthesized report content),
doc.convert transforms the bytes of already-uploaded Files into
downloadable Artifacts without any LLM or retrieval step:

    PDF  (Docling Markdown export) → md | docx | pdf
    DOCX (python-docx extraction)  → md | docx | pdf

No RAG (`retrieve_context` is top-k/threshold lossy and must never sit
on the convert path), no reasoning/paraphrase — the full source text is
preserved verbatim (DOCX formatting is normalized on re-render).

`corpus_id` is injected by the orchestrator (never LLM-generated).
Single-file conversion takes `file_id` (preferred — a literal id from the
`corpus_context` snapshot, never invented, never a placeholder) or
`file_name` (alias, resolved within the corpus). `file_id: "*"`
converts every ready file (convert-all requests).

Effect class: sandboxed (bounded compute producing artifacts).
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, ClassVar

from rip_maf.tools.base import Tool, ToolRequest, ToolResponse

logger = logging.getLogger("tools.doc_convert")

_TARGET_FORMATS = ("md", "docx", "pdf")
_SOURCE_EXTS = (".pdf", ".docx")


def _load_pdf_markdown(file_path: str) -> str:
    """Full PDF→Markdown via the same loaders as ingest (no BGE needed).

    Mirrors `RagPipeline.document_loader` order (`RAG_PDF_LOADER` setting)
    but stays lightweight: no embedding/reranker models are constructed.
    Returns the complete markdown text (never chunked/filtered).
    """
    from rip_maf.core.config import settings as _settings

    primary = (_settings.rag_pdf_loader or "docling").strip().lower()

    def _via_docling(path: str) -> str:
        from docling.document_converter import DocumentConverter

        result = DocumentConverter().convert(path)
        return str(result.document.export_to_markdown())

    def _via_opendataloader(path: str) -> str:
        from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader

        docs = OpenDataLoaderPDFLoader(path, format="markdown").load_and_split()
        return "\n\n".join(getattr(d, "page_content", "") for d in docs)

    first, second = (
        (_via_opendataloader, _via_docling)
        if primary == "opendataloader"
        else (_via_docling, _via_opendataloader)
    )
    try:
        return first(file_path)
    except Exception as e:  # noqa: BLE001 - fallback loader gets its chance
        logger.warning("doc.convert primary loader failed, trying fallback: %s", e)
        return second(file_path)


def _load_docx_markdown(file_path: str) -> str:
    """Full DOCX→Markdown via python-docx (no LLM, no chunking).

    Heading styles map to Markdown headings; tables become Markdown
    tables; everything else is preserved paragraph-verbatim.
    """
    try:
        from docx import Document as _DocxDocument
    except ImportError as e:
        raise RuntimeError(f"python-docx is not installed: {e}") from e
    doc = _DocxDocument(file_path)
    lines: list[str] = []
    for para in doc.paragraphs:
        text = (para.text or "").strip()
        if not text:
            continue
        style = (getattr(para.style, "name", "") or "").lower()
        if style.startswith("heading 1"):
            lines.append(f"# {text}")
        elif style.startswith("heading 2"):
            lines.append(f"## {text}")
        elif style.startswith("heading 3"):
            lines.append(f"### {text}")
        else:
            lines.append(text)
    for table in doc.tables:
        rows = [[(c.text or "").strip() for c in row.cells] for row in table.rows]
        if not rows:
            continue
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        lines.append(" | ".join(rows[0]))
        lines.append(" | ".join("---" for _ in range(width)))
        for row in rows[1:]:
            lines.append(" | ".join(row))
    return "\n\n".join(lines)


def _load_markdown(file_path: str, ext: str) -> str:
    if ext == ".docx":
        return _load_docx_markdown(file_path)
    return _load_pdf_markdown(file_path)


def _disk_path(corpus_id: str, file_id: str, ext: str) -> str:
    from rip_maf.core.config import settings as _settings

    return os.path.join(str(_settings.upload_dir), str(corpus_id), f"{file_id}{ext}")


def _resolve_source(corpus_id: str, file_id: str) -> tuple[str, str, str]:
    """Validate ownership + ready status; return (file_path, file_name, ext).

    Raises ValueError with a user-safe message on any failure.
    """
    from rip_maf.core.db import pg_connection

    with pg_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT corpus_id, file_name, file_status FROM files WHERE file_id = %s",
            (str(file_id),),
        )
        row = cur.fetchone()
    if not row:
        raise ValueError(f"unknown file_id: {file_id}")
    owner_nb, file_name, status = str(row[0]), row[1] or "document", row[2]
    if owner_nb != str(corpus_id):
        raise ValueError(f"file {file_id} does not belong to this corpus")
    if status != "ready":
        raise ValueError(f"file '{file_name}' is not ready (status={status}) — retry when ready")
    ext = (os.path.splitext(file_name or "")[1].lower() or ".pdf")
    if ext not in _SOURCE_EXTS:
        raise ValueError(
            f"unsupported source '{file_name}' — converts PDF and DOCX only"
        )
    file_path = _disk_path(str(corpus_id), str(file_id), ext)
    if not os.path.isfile(file_path):
        raise ValueError(f"source file missing on disk for '{file_name}'")
    return file_path, str(file_name), ext


def _resolve_by_name(corpus_id: str, file_name: str) -> tuple[str, str, str, str]:
    """Resolve `file_name` within the corpus; return (file_id, path, name, ext)."""
    from rip_maf.core.db import pg_connection

    with pg_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT file_id, file_name, file_status FROM files "
            "WHERE corpus_id = %s AND file_name = %s ORDER BY created_at ASC",
            (str(corpus_id), str(file_name)),
        )
        rows = cur.fetchall()
    if not rows:
        raise ValueError(f"no file named '{file_name}' in this corpus")
    if len(rows) > 1:
        raise ValueError(
            f"multiple files named '{file_name}' — ask for the file_id from corpus.inspect"
        )
    file_id, name, status = str(rows[0][0]), rows[0][1] or file_name, rows[0][2]
    if status != "ready":
        raise ValueError(f"file '{name}' is not ready (status={status}) — retry when ready")
    ext = (os.path.splitext(name or "")[1].lower() or ".pdf")
    if ext not in _SOURCE_EXTS:
        raise ValueError(
            f"unsupported source '{name}' — converts PDF and DOCX only"
        )
    file_path = _disk_path(str(corpus_id), file_id, ext)
    if not os.path.isfile(file_path):
        raise ValueError(f"source file missing on disk for '{name}'")
    return file_id, file_path, str(name), ext


def _list_ready(corpus_id: str) -> list[tuple[str, str, str]]:
    """All ready, convertible files: [(file_id, file_name, ext)]."""
    from rip_maf.core.db import pg_connection

    with pg_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT file_id, file_name FROM files "
            "WHERE corpus_id = %s AND file_status = 'ready' ORDER BY created_at ASC",
            (str(corpus_id),),
        )
        rows = cur.fetchall()
    ready: list[tuple[str, str, str]] = []
    for r in rows:
        name = r[1] or "document"
        ext = (os.path.splitext(name or "")[1].lower() or ".pdf")
        if ext in _SOURCE_EXTS:
            ready.append((str(r[0]), str(name), ext))
    return ready


def _render_target(stem: str, markdown: str, target: str) -> dict[str, str]:
    """Render docx/pdf binaries for one conversion; {} for md targets."""
    if target == "md":
        return {}
    from rip_maf.tools.doc_generate import render_docx, render_pdf

    sections = [{"heading": stem, "body": markdown}]
    if target == "docx":
        return {
            "docx_b64": base64.b64encode(render_docx(stem, sections, [])).decode("ascii")
        }
    return {
        "pdf_b64": base64.b64encode(render_pdf(stem, sections, [])).decode("ascii")
    }


def _convert_one(
    corpus_id: str, file_id: str, file_path: str, file_name: str, ext: str, target: str
) -> tuple[str, dict[str, Any]]:
    """Convert one resolved file; return (markdown, data-entry)."""
    del corpus_id  # ownership already validated by the resolver
    markdown = _load_markdown(file_path, ext)
    if not markdown.strip():
        raise ValueError(f"source '{file_name}' produced no text")
    stem = os.path.splitext(file_name)[0] or "document"
    entry: dict[str, Any] = {
        "markdown": markdown,
        "source_file_id": str(file_id),
        "source_file_name": file_name,
        "target_format": target,
    }
    entry.update(_render_target(stem, markdown, target))
    return markdown, entry


class DocConvertTool(Tool):
    tool_id = "doc.convert"
    name = "Doc Convert"
    description = (
        "Convert uploaded PDF/DOCX files to md, docx or pdf from the source "
        "(lossless, no LLM, no search). Input: file_id (preferred — a literal "
        "id from the corpus snapshot, never invented; \"*\" converts every "
        "ready file) or file_name (alias within the corpus), plus "
        "target_format (md|docx|pdf). NEVER use {{{{...}}}} placeholders for "
        "file_id. Use ONLY for convert/export/save-as requests; use "
        "doc.generate for reports synthesized from answers."
    )
    input_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "corpus_id": {"type": "string"},
            "file_id": {"type": "string"},
            "file_name": {"type": "string"},
            "target_format": {
                "type": "string",
                "enum": sorted(_TARGET_FORMATS),
            },
        },
        "required": ["corpus_id", "target_format"],
        "additionalProperties": False,
    }
    input_example: ClassVar[str] = (
        'doc.convert {"file_id": "<literal snapshot id>", "target_format": '
        '"md"}. Use file_id "*" for every ready file, or file_name for a '
        "name. Never a {{id}} placeholder for file_id."
    )
    output_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "markdown": {"type": "string"},
            "docx_b64": {"type": "string"},
            "pdf_b64": {"type": "string"},
            "conversions": {"type": "array"},
        },
    }
    effect_class = "sandboxed"  # type: ignore[assignment]
    cost_class = "low"

    def validate_input(self, tool_input: dict) -> str | None:
        err = super().validate_input(tool_input)
        if err is not None:
            return err
        file_id = tool_input.get("file_id")
        file_name = tool_input.get("file_name")
        # isinstance, not truthiness: str(None) is "None" and would read as
        # a target.
        has_target = (isinstance(file_id, str) and file_id.strip()) or (
            isinstance(file_name, str) and file_name.strip()
        )
        if not has_target:
            return (
                "'file_id' (or \"*\" for all files, or 'file_name') is "
                "required in input"
            )
        return None

    def execute(self, request: ToolRequest) -> ToolResponse:
        corpus_id = request.input.get("corpus_id")
        if not corpus_id:
            return ToolResponse(
                tool_id=self.tool_id,
                ok=False,
                output=None,
                error="'corpus_id' is required in input (injected by the orchestrator, never the LLM)",
            )
        invalid = self.invalid_response(request.input)
        if invalid.error is not None:
            return invalid
        target = str(request.input.get("target_format") or "").strip().lower()
        file_id = request.input.get("file_id")
        file_name = request.input.get("file_name")
        if file_id == "*":
            return self._execute_all(str(corpus_id), target)
        if isinstance(file_id, str) and file_id:
            return self._execute_one_by_id(str(corpus_id), file_id, target)
        return self._execute_one_by_name(str(corpus_id), str(file_name), target)

    def _execute_one_by_id(
        self, corpus_id: str, file_id: str, target: str
    ) -> ToolResponse:
        try:
            file_path, file_name, ext = _resolve_source(corpus_id, file_id)
        except ValueError as e:
            return ToolResponse(tool_id=self.tool_id, ok=False, output=None, error=str(e))
        except Exception as e:
            logger.exception("doc.convert source lookup failed")
            return ToolResponse(
                tool_id=self.tool_id, ok=False, output=None, error=f"lookup failed: {e}"
            )
        return self._finish_one(corpus_id, file_id, file_path, file_name, ext, target)

    def _execute_one_by_name(
        self, corpus_id: str, file_name: str, target: str
    ) -> ToolResponse:
        try:
            file_id, file_path, name, ext = _resolve_by_name(corpus_id, file_name)
        except ValueError as e:
            return ToolResponse(tool_id=self.tool_id, ok=False, output=None, error=str(e))
        except Exception as e:
            logger.exception("doc.convert source lookup failed")
            return ToolResponse(
                tool_id=self.tool_id, ok=False, output=None, error=f"lookup failed: {e}"
            )
        return self._finish_one(corpus_id, file_id, file_path, name, ext, target)

    def _finish_one(
        self,
        corpus_id: str,
        file_id: str,
        file_path: str,
        file_name: str,
        ext: str,
        target: str,
    ) -> ToolResponse:
        try:
            markdown, entry = _convert_one(
                corpus_id, file_id, file_path, file_name, ext, target
            )
        except ValueError as e:
            return ToolResponse(tool_id=self.tool_id, ok=False, output=None, error=str(e))
        except RuntimeError as e:
            return ToolResponse(tool_id=self.tool_id, ok=False, output=None, error=str(e))
        except Exception as e:
            logger.exception("doc.convert conversion failed")
            return ToolResponse(
                tool_id=self.tool_id, ok=False, output=None, error=f"conversion failed: {e}"
            )
        output = markdown if target == "md" else (
            f"Converted '{file_name}' to {target} ({len(markdown)} chars source)."
        )
        return ToolResponse(tool_id=self.tool_id, ok=True, output=output, data=entry)

    def _execute_all(self, corpus_id: str, target: str) -> ToolResponse:
        try:
            ready = _list_ready(corpus_id)
        except Exception as e:
            logger.exception("doc.convert listing failed")
            return ToolResponse(
                tool_id=self.tool_id, ok=False, output=None, error=f"lookup failed: {e}"
            )
        if not ready:
            return ToolResponse(
                tool_id=self.tool_id,
                ok=False,
                output=None,
                error="no ready PDF/DOCX files in this corpus",
            )
        conversions: list[dict[str, Any]] = []
        failures: list[str] = []
        for file_id, file_name, ext in ready:
            file_path = _disk_path(corpus_id, file_id, ext)
            if not os.path.isfile(file_path):
                failures.append(f"'{file_name}': source file missing on disk")
                continue
            try:
                _markdown, entry = _convert_one(
                    corpus_id, file_id, file_path, file_name, ext, target
                )
            except (ValueError, RuntimeError) as e:
                failures.append(f"'{file_name}': {e}")
            except Exception as e:
                logger.exception("doc.convert per-file conversion failed")
                failures.append(f"'{file_name}': conversion failed: {e}")
            else:
                conversions.append(entry)
        if not conversions:
            return ToolResponse(
                tool_id=self.tool_id,
                ok=False,
                output=None,
                error="all conversions failed: " + "; ".join(failures),
            )
        names = ", ".join(c["source_file_name"] for c in conversions)
        output = f"Converted {len(conversions)} file(s) to {target}: {names}."
        if failures:
            output += " Failed: " + "; ".join(failures) + "."
        return ToolResponse(
            tool_id=self.tool_id,
            ok=True,
            output=output,
            data={"conversions": conversions, "target_format": target},
        )
