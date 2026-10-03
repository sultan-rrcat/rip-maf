"""File-based artifacts (Q34 REWRITE — not Athena's inline base64).

Tool steps that produce files/charts/images carry their raw
`ToolResponse.data` dict forward on `StepResult.data`. This module converts
those payloads into FILES under:

    {upload_dir}/{corpus_id}/artifacts/{run_id}/{step_id}/{filename}

plus an `index.json` per run mapping `artifact_id -> {step_id, filename,
kind, mime}`. The SSE `artifacts` event carries only
`{artifact_id, kind, filename, url}` download links (never inline base64);
`GET /v1/runs/{run_id}/artifacts/{artifact_id}` serves the bytes.

Collection keys on DATA SHAPES, never on tool ids (same shapes as Athena):
- {"svg": "<svg...>"} → chart (image/svg+xml)
- {"image_b64": ..., "mime": ...} → image
- {"docx_b64": ...} / {"pdf_b64": ...} → document
- {"markdown": ...} (only when it rides with binaries) → document (.md)
- {"file_b64": ..., "filename": ..., "mime": ...} → document (verbatim
  text file from doc.generate: txt/csv/md/json/code, byte-for-byte)
- {"rows": [...], "row_count": ...} → data (.json)

Anything else (plain text answers, stdout dumps, RAG passages) is NOT an
artifact — it already travels via the step output / final answer.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import uuid
from pathlib import Path

logger = logging.getLogger("artifacts")

MIME_SVG = "image/svg+xml"
MIME_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
MIME_PDF = "application/pdf"
MIME_MARKDOWN = "text/markdown"
MIME_JSON = "application/json"
MIME_PNG_FALLBACK = "image/png"

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe(name: str) -> str:
    """Filesystem-safe segment (planner-generated ids are untrusted input)."""
    return _SAFE.sub("_", name).strip("._") or "file"


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def collect_artifacts(
    step_results: list,
    *,
    upload_dir: str,
    corpus_id: str,
    run_id: str,
) -> list[dict]:
    """Persist tool outputs to disk; return SSE-ready artifact descriptors.

    Each descriptor is `{artifact_id, kind, filename, url}` with
    `url = /v1/runs/{run_id}/artifacts/{artifact_id}`. Returns [] when no
    step produced an artifact shape. Fail-soft per artifact: an undecodable
    payload is logged and skipped, never aborting the run.
    """
    run_dir = (
        Path(upload_dir) / str(corpus_id) / "artifacts" / str(run_id)
    )
    found: list[dict] = []
    # Content hashes of charts already collected: identical SVG bytes
    # (trace 07fb4f59 r2/r3 plotted the same data twice) would otherwise
    # surface the same plot twice in the frontend Artifacts panel.
    seen_charts: set[str] = set()
    for result in step_results:
        if getattr(result.status, "value", result.status) != "success":
            continue
        data = getattr(result, "data", None) or {}
        if not isinstance(data, dict):
            continue
        step_id = _safe(str(getattr(result, "step_id", "step")))
        found.extend(
            _collect_from_data(
                data, run_dir=run_dir, run_id=str(run_id), step_id=step_id,
                seen_charts=seen_charts,
            )
        )
    if found:
        _write_index(run_dir, found)
    return found


def _collect_from_data(
    data: dict, *, run_dir: Path, run_id: str, step_id: str,
    seen_charts: set[str] | None = None,
) -> list[dict]:
    out: list[dict] = []

    # doc.convert convert-all: one entry per source file.
    conversions = data.get("conversions")
    if isinstance(conversions, list) and conversions:
        for entry in conversions:
            if isinstance(entry, dict):
                out.extend(
                    _collect_from_data(
                        entry, run_dir=run_dir, run_id=run_id, step_id=step_id,
                        seen_charts=seen_charts,
                    )
                )
        return out

    def _stem() -> str:
        """Filename stem: source file name when present, else the step id."""
        raw = data.get("source_file_name")
        if isinstance(raw, str) and raw.strip():
            stem = raw.strip().rsplit(".", 1)[0]
            return _safe(stem)
        return _safe(step_id)

    def _filename(ext: str) -> str:
        stem = _stem()
        if stem == _safe(step_id):
            return f"{stem}.{ext}"
        return f"{_safe(step_id)}_{stem}.{ext}"

    def add(kind: str, mime: str, filename: str, content: bytes) -> None:
        artifact_id = uuid.uuid4().hex
        try:
            _write_bytes(run_dir / step_id / filename, content)
        except OSError:
            logger.exception("artifact write failed run=%s step=%s", run_id, step_id)
            return
        out.append(
            {
                "artifact_id": artifact_id,
                "kind": kind,
                "mime": mime,
                "step_id": step_id,
                "filename": filename,
                "url": f"/v1/runs/{run_id}/artifacts/{artifact_id}",
            }
        )

    svg = data.get("svg")
    if isinstance(svg, str) and svg.lstrip().startswith("<svg"):
        import hashlib as _hashlib

        digest = _hashlib.sha256(svg.encode("utf-8")).hexdigest()
        if seen_charts is not None:
            if digest in seen_charts:
                logger.info(
                    "artifact duplicate chart skipped run=%s step=%s",
                    run_id, step_id,
                )
            else:
                seen_charts.add(digest)
                add("chart", MIME_SVG, f"{step_id}.svg", svg.encode("utf-8"))
        else:
            add("chart", MIME_SVG, f"{step_id}.svg", svg.encode("utf-8"))

    image_b64 = data.get("image_b64")
    if isinstance(image_b64, str) and image_b64:
        raw_mime = data.get("mime")
        mime = raw_mime if isinstance(raw_mime, str) and raw_mime else MIME_PNG_FALLBACK
        ext = "png" if "png" in mime else ("svg" if "svg" in mime else "bin")
        try:
            add("image", mime, f"{step_id}.{ext}", base64.b64decode(image_b64))
        except ValueError:
            logger.warning("artifact image_b64 undecodable run=%s step=%s", run_id, step_id)

    docx_b64 = data.get("docx_b64")
    if isinstance(docx_b64, str) and docx_b64:
        try:
            add("document", MIME_DOCX, _filename("docx"), base64.b64decode(docx_b64))
        except ValueError:
            logger.warning("artifact docx_b64 undecodable run=%s step=%s", run_id, step_id)

    pdf_b64 = data.get("pdf_b64")
    if isinstance(pdf_b64, str) and pdf_b64:
        try:
            add("document", MIME_PDF, _filename("pdf"), base64.b64decode(pdf_b64))
        except ValueError:
            logger.warning("artifact pdf_b64 undecodable run=%s step=%s", run_id, step_id)

    markdown = data.get("markdown")
    if isinstance(markdown, str) and markdown and ("docx_b64" in data or "pdf_b64" in data):
        # Only a file artifact when it rides with rendered binaries; a bare
        # markdown string is just step output — EXCEPT doc.convert output,
        # which is a verbatim file conversion (source_file_id marks it).
        add("document", MIME_MARKDOWN, _filename("md"), markdown.encode("utf-8"))
    elif (
        isinstance(markdown, str)
        and markdown
        and isinstance(data.get("source_file_id"), str)
    ):
        add("document", MIME_MARKDOWN, _filename("md"), markdown.encode("utf-8"))

    rows = data.get("rows")
    if isinstance(rows, list) and "row_count" in data:
        add("data", MIME_JSON, f"{step_id}.json", json.dumps(rows).encode("utf-8"))

    file_b64 = data.get("file_b64")
    if isinstance(file_b64, str) and file_b64:
        raw_name = data.get("filename")
        filename = (
            _safe(str(raw_name)) if isinstance(raw_name, str) and raw_name.strip()
            else f"{step_id}.txt"
        )
        raw_mime = data.get("mime")
        mime = (
            str(raw_mime)
            if isinstance(raw_mime, str) and raw_mime.strip()
            else "text/plain"
        )
        try:
            add("document", mime, filename, base64.b64decode(file_b64))
        except ValueError:
            logger.warning("artifact file_b64 undecodable run=%s step=%s", run_id, step_id)

    return out


def _write_index(run_dir: Path, artifacts: list[dict]) -> None:
    index = {
        a["artifact_id"]: {
            "step_id": a["step_id"],
            "filename": a["filename"],
            "kind": a["kind"],
            "mime": a["mime"],
        }
        for a in artifacts
    }
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")
    except OSError:
        logger.exception("artifact index write failed dir=%s", run_dir)


def resolve_artifact(
    *, upload_dir: str, corpus_id: str, run_id: str, artifact_id: str
) -> tuple[Path, str, str] | None:
    """Resolve an artifact_id to (path, filename, mime); None when unknown.

    Used by `GET /v1/runs/{run_id}/artifacts/{artifact_id}`. The lookup is
    confined to the run's own directory (index.json), so one run can never
    address another run's files — no path traversal via crafted ids.
    """
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", artifact_id or ""):
        return None
    run_dir = Path(upload_dir) / str(corpus_id) / "artifacts" / str(run_id)
    try:
        index = json.loads((run_dir / "index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    entry = index.get(artifact_id)
    if not isinstance(entry, dict):
        return None
    step_id = _safe(str(entry.get("step_id", "")))
    filename = _safe(str(entry.get("filename", "")))
    path = run_dir / step_id / filename
    try:
        if not path.is_file():
            return None
    except OSError:
        return None
    mime = str(entry.get("mime") or "application/octet-stream")
    return path, filename, mime


__all__ = [
    "collect_artifacts",
    "resolve_artifact",
]
