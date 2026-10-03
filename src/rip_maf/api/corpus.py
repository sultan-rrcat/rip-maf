"""Corpus API — the Open WebUI Pipe's ingest surface.

A corpus is the RAG scope unit, keyed `(owner_ref, corpus_ref)` where
`corpus_ref` is the Open WebUI chat id. Ingest wraps the existing pipeline
(`services.ingest.run_rag_pipeline`); answering reuses `POST /v1/runs`.
"""
from __future__ import annotations

import logging
import os
from uuid import uuid4

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from rip_maf.auth import Principal, require_principal
from rip_maf.core.config import settings
from rip_maf.core.db import pg_connection
from rip_maf.services.ingest import run_rag_pipeline

logger = logging.getLogger("api.corpus")

router = APIRouter()

_CHUNK_SIZE = 8192


class CorpusRequest(BaseModel):
    chat_id: str
    corpus_name: str | None = None


class CorpusResponse(BaseModel):
    corpus_id: str
    created: bool


def _get_or_create_corpus(
    owner_ref: str, chat_id: str, corpus_name: str | None
) -> tuple[str, bool]:
    corpus_ref = str(chat_id)
    with pg_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT corpus_id FROM corpora WHERE owner_ref = %s AND corpus_ref = %s",
            (owner_ref, corpus_ref),
        )
        row = cur.fetchone()
        if row:
            return str(row[0]), False
        name = corpus_name or f"OWUI chat {corpus_ref}"
        cur.execute(
            "INSERT INTO corpora (name, owner_ref, corpus_ref) "
            "VALUES (%s, %s, %s) RETURNING corpus_id",
            (name, owner_ref, corpus_ref),
        )
        row = cur.fetchone()
    return str(row[0]), True


def _verify_ownership(corpus_id: str, owner_ref: str) -> None:
    with pg_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM corpora WHERE corpus_id = %s AND owner_ref = %s",
            (corpus_id, owner_ref),
        )
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Corpus not found")


@router.post("/v1/corpus", response_model=CorpusResponse)
def ensure_corpus(
    body: CorpusRequest,
    principal: Principal = Depends(require_principal),  # noqa: B008
):
    """Idempotently resolve the corpus for one Open WebUI chat."""
    if not body.chat_id.strip():
        raise HTTPException(status_code=422, detail="chat_id is required")
    corpus_id, created = _get_or_create_corpus(
        principal.owner_ref, body.chat_id.strip(), body.corpus_name
    )
    return CorpusResponse(corpus_id=corpus_id, created=created)


@router.post("/v1/corpus/{corpus_id}/files")
async def ingest_file(
    corpus_id: str,
    file: UploadFile = File(...),  # noqa: B008
    principal: Principal = Depends(require_principal),  # noqa: B008
):
    """Ingest one uploaded file: save → parse → embed → ready (inline)."""
    _verify_ownership(corpus_id, principal.owner_ref)

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"File type '{ext}' not allowed. Allowed: {settings.allowed_extensions}",
        )

    from rip_maf.tools.rag_query import get_rag_singleton

    rag = get_rag_singleton()
    if rag is None:
        raise HTTPException(
            status_code=503,
            detail="RAG models not loaded (BGE weights missing) — retry after restart",
        )

    file_id = str(uuid4())
    corpus_path = os.path.join(settings.upload_dir, corpus_id)
    os.makedirs(corpus_path, exist_ok=True)
    file_path = os.path.join(corpus_path, f"{file_id}{ext}")

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    file_size = 0
    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(_CHUNK_SIZE):
                file_size += len(chunk)
                if file_size > max_bytes:
                    await f.close()
                    os.remove(file_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {settings.max_upload_size_mb} MB limit",
                    )
                await f.write(chunk)
    except HTTPException:
        raise
    except Exception:
        logger.exception("ingest upload failed corpus=%s", corpus_id)
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail="Failed to save file") from None

    try:
        with pg_connection() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO files (file_id, corpus_id, file_name, file_size, "
                "file_status) VALUES (%s, %s, %s, %s, 'processing')",
                (file_id, corpus_id, file.filename, file_size),
            )
    except Exception:
        logger.exception("ingest DB insert failed file=%s", file_id)
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500, detail="Failed to save file metadata"
        ) from None

    await run_rag_pipeline(file_id, rag)

    with pg_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT file_status FROM files WHERE file_id = %s", (file_id,))
        row = cur.fetchone()
    status = row[0] if row else "error"
    return {
        "id": file_id,
        "name": file.filename,
        "size": file_size,
        "status": status,
    }
