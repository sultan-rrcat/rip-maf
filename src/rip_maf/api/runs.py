"""Run lifecycle routes (Phase 4.1).

Q1 locked contract, copied verbatim in shape:
- `POST /v1/runs` body is exactly `{corpus_id: UUID, message: str}` →
  `202 {run_id}` bare JSON. No auth/quota/tenant, no BFF envelope on `/v1/*`.
- Errors: `404` unknown `corpus_id`, `422` empty message.
- `GET /v1/runs/{id}` — run detail (Postgres row is truth).
- `GET /v1/runs/{id}/events` — SSE; replays persisted `run_events ORDER BY
  seq` (structural + final text, no deltas — Q35) with `id:<seq>`, then
  streams live. Frontend dedupes by `seq`. No `?last_event_id=` in v1.
- `POST /v1/runs/{id}/cancel` — cooperative cancel (idempotent; a late
  cancel never overwrites a final state).
- `GET /v1/runs/{id}/artifacts/{artifact_id}` — Q34 file download.

Full `/v1/...` paths inline (RIP convention — routers mount unprefixed).
"""
from __future__ import annotations

import json
import logging
import queue
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, field_validator

from rip_maf.api.deps import get_run_manager
from rip_maf.artifacts import resolve_artifact
from rip_maf.auth import Principal, require_principal
from rip_maf.core.config import settings
from rip_maf.core.db import pg_connection
from rip_maf.runs import store as run_store

logger = logging.getLogger("api.runs")

router = APIRouter()


class CreateRunRequest(BaseModel):
    corpus_id: UUID
    message: str  # min_length=1 after strip; 422 on empty
    # rip-maf: caller-supplied conversation turns (Open WebUI owns history).
    # When present the worker is stateless (no DB message/summary read).
    history: list[dict] | None = None

    @field_validator("message")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be empty")
        return v


class CreateRunResponse(BaseModel):
    run_id: UUID  # bare JSON, NO BFF envelope on /v1/*


class RunDetail(BaseModel):
    run_id: str
    corpus_id: str
    status: str
    goal: str | None = None


class CancelRunResponse(BaseModel):
    run_id: str
    status: str
    already_done: bool


def _verify_corpus_ownership(corpus_id: str, owner_ref: str) -> None:
    """Raise 404 if corpus not found or not owned by user (no 403 leak)."""
    with pg_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM corpora WHERE corpus_id = %s AND owner_ref = %s",
            (corpus_id, owner_ref),
        )
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Corpus not found")


def _verify_run_ownership(run_id: str, owner_ref: str) -> None:
    """Raise 404 if run not found or not owned by user (no 403 leak)."""
    with pg_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM runs r
            JOIN corpora n ON n.corpus_id = r.corpus_id
            WHERE r.id = %s AND n.owner_ref = %s
        """,
            (run_id, owner_ref),
        )
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Run not found")


def _frame(seq: Any, type: str, run_id: str, data: dict) -> str:
    """One SSE frame: `id: <seq>` + flat JSON payload (type/run_id/seq up)."""
    payload = {"type": type, "run_id": run_id, "seq": seq, **(data or {})}
    return f"id: {seq}\ndata: {json.dumps(payload)}\n\n"


@router.post("/v1/runs", response_model=CreateRunResponse, status_code=202)
def create_run(
    body: CreateRunRequest,
    manager=Depends(get_run_manager),  # noqa: B008 - FastAPI Depends-in-default is canonical
    principal: Principal = Depends(require_principal),  # noqa: B008
):
    try:
        _verify_corpus_ownership(str(body.corpus_id), principal.owner_ref)
        record = manager.create_run(
            str(body.corpus_id), body.message, history=body.history
        )
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(
            status_code=429,
            detail=str(e),
            headers={"Retry-After": "30"},
        ) from e
    logger.info("created run id=%s corpus=%s", record.run_id, record.corpus_id)
    return CreateRunResponse(run_id=UUID(record.run_id))


@router.get("/v1/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: str, principal: Principal = Depends(require_principal)):  # noqa: B008
    _verify_run_ownership(run_id, principal.owner_ref)
    row = run_store.get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunDetail(
        run_id=row.id, corpus_id=row.corpus_id, status=row.status, goal=row.goal
    )


@router.post("/v1/runs/{run_id}/cancel", response_model=CancelRunResponse)
def cancel_run(
    run_id: str,
    manager=Depends(get_run_manager),  # noqa: B008 - FastAPI Depends-in-default is canonical
    principal: Principal = Depends(require_principal),  # noqa: B008
):
    _verify_run_ownership(run_id, principal.owner_ref)
    handled, already_done = manager.cancel_run(run_id)
    if not handled:
        raise HTTPException(status_code=404, detail="Run not found")
    row = run_store.get_run(run_id)
    status = row.status if row else "unknown"
    return CancelRunResponse(run_id=run_id, status=status, already_done=already_done)


@router.get("/v1/runs/{run_id}/events")
def run_events(
    run_id: str,
    request: Request,
    manager=Depends(get_run_manager),  # noqa: B008 - FastAPI Depends-in-default is canonical
    principal: Principal = Depends(require_principal),  # noqa: B008
):
    _verify_run_ownership(run_id, principal.owner_ref)
    try:
        events, live, _done = manager.subscribe(run_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Run not found") from None

    async def stream():
        for ev in events:
            yield _frame(ev.seq, ev.event_type, run_id, ev.payload or {})
        while True:
            if await request.is_disconnected():
                return
            try:
                item = live.get(timeout=15)
            except queue.Empty:
                yield ": heartbeat\n\n"
                continue
            if item is None:
                return
            yield _frame(item["seq"], item["type"], run_id, item.get("data") or {})

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.get("/v1/runs/{run_id}/artifacts/{artifact_id}")
def download_artifact(
    run_id: str,
    artifact_id: str,
    principal: Principal = Depends(require_principal),  # noqa: B008
):
    _verify_run_ownership(run_id, principal.owner_ref)
    row = run_store.get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    resolved = resolve_artifact(
        upload_dir=settings.upload_dir,
        corpus_id=row.corpus_id,
        run_id=run_id,
        artifact_id=artifact_id,
    )
    if resolved is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    path, filename, mime = resolved
    return FileResponse(path, filename=filename, media_type=mime)
