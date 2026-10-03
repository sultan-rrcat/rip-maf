"""Postgres-backed runs store (Phase 3.3).

Persists runs + their structural SSE event log to the `runs` / `run_events`
tables (schema.sql) via `core.db.pg_connection`, so runs survive page
refresh and reconnects replay in order. NOT a copy of Athena's SQLite store;
corpus = conversation, so no conversations table is involved.

Q35 persistence rule lives in append_event: `delta` token events are
live-streamed only and NEVER written to run_events (skipped events consume
no sequence number, so replay has no gaps). Reconnect clients resume text
from `step_completed` / `summary` payloads.

Sequence numbers are monotonic per run: append_event locks the parent run
row (SELECT ... FOR UPDATE) inside its transaction, so concurrent writers
cannot reuse a seq. Every call opens its own connection — durability across
connections/processes is inherent.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from psycopg2.extras import Json, RealDictCursor

from rip_maf.core.db import pg_connection

logger = logging.getLogger("store.runs")

VALID_STATUSES = ("pending", "running", "completed", "failed", "cancelled")

# Live-only event type: streamed to connected SSE clients, never persisted.
_LIVE_ONLY = "delta"


class Run:
    """One orchestration unit (mirrors the `runs` row)."""

    def __init__(
        self,
        id: str,
        corpus_id: str,
        status: str,
        goal: str | None = None,
        plan: dict | None = None,
        result: dict | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ):
        self.id = str(id)
        self.corpus_id = str(corpus_id)
        self.status = status
        self.goal = goal
        self.plan = plan
        self.result = result
        self.created_at = created_at
        self.updated_at = updated_at

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Run(id={self.id!r}, status={self.status!r})"


class RunEvent:
    """One persisted SSE event (mirrors the `run_events` row)."""

    def __init__(
        self,
        id: int,
        run_id: str,
        seq: int,
        event_type: str,
        payload: dict | None = None,
        created_at: datetime | None = None,
    ):
        self.id = id
        self.run_id = str(run_id)
        self.seq = seq
        self.event_type = event_type
        self.payload = payload if payload is not None else {}
        self.created_at = created_at

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"RunEvent(run_id={self.run_id!r}, "
            f"seq={self.seq}, type={self.event_type!r})"
        )


def _row_to_run(row: dict) -> Run:
    return Run(
        id=row["id"],
        corpus_id=row["corpus_id"],
        status=row["status"],
        goal=row.get("goal"),
        plan=row.get("plan"),
        result=row.get("result"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _row_to_event(row: dict) -> RunEvent:
    return RunEvent(
        id=row["id"],
        run_id=row["run_id"],
        seq=row["seq"],
        event_type=row["event_type"],
        payload=row.get("payload"),
        created_at=row.get("created_at"),
    )


def create_run(corpus_id: str) -> Run:
    """Insert a `pending` run for a corpus; fail-honest on FK violation."""
    with pg_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
                INSERT INTO runs (corpus_id)
                VALUES (%s)
                RETURNING id, corpus_id, status, goal, plan, result,
                          created_at, updated_at
                """,
            (str(corpus_id),),
        )
        row = cur.fetchone()
    run = _row_to_run(row)
    logger.info("run created id=%s corpus=%s", run.id, run.corpus_id)
    return run


def get_run(run_id: str) -> Run | None:
    """Fetch one run; None when unknown (fail-soft for GET/detail paths)."""
    with pg_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
                SELECT id, corpus_id, status, goal, plan, result,
                       created_at, updated_at
                FROM runs WHERE id = %s
                """,
            (str(run_id),),
        )
        row = cur.fetchone()
    return _row_to_run(row) if row else None


def list_runs(corpus_id: str) -> list[Run]:
    """All runs of a corpus, oldest first."""
    with pg_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
                SELECT id, corpus_id, status, goal, plan, result,
                       created_at, updated_at
                FROM runs WHERE corpus_id = %s ORDER BY created_at, id
                """,
            (str(corpus_id),),
        )
        rows = cur.fetchall()
    return [_row_to_run(r) for r in rows]


def update_run(
    run_id: str,
    *,
    status: str | None = None,
    goal: str | None = None,
    plan: dict | None = None,
    result: dict | None = None,
) -> Run:
    """Patch run fields; KeyError on unknown run, ValueError on bad status."""
    if status is not None and status not in VALID_STATUSES:
        raise ValueError(
            f"status {status!r} must be one of {', '.join(VALID_STATUSES)}"
        )
    assignments: list[str] = ["updated_at = CURRENT_TIMESTAMP"]
    values: list[Any] = []
    if status is not None:
        assignments.append("status = %s")
        values.append(status)
    if goal is not None:
        assignments.append("goal = %s")
        values.append(goal)
    if plan is not None:
        assignments.append("plan = %s")
        values.append(Json(plan))
    if result is not None:
        assignments.append("result = %s")
        values.append(Json(result))
    values.append(str(run_id))
    with pg_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            f"""
                UPDATE runs SET {', '.join(assignments)}
                WHERE id = %s
                RETURNING id, corpus_id, status, goal, plan, result,
                          created_at, updated_at
                """,
            values,
        )
        row = cur.fetchone()
    if row is None:
        raise KeyError(f"Unknown run: {run_id}")
    return _row_to_run(row)


def cancel_run(run_id: str) -> bool:
    """Terminal transition for the stop button: pending/running → cancelled.

    Returns True when the run was actually cancelled; False when it was
    already terminal (completed/failed/cancelled) or unknown — a late cancel
    must never overwrite a final state.
    """
    with pg_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
                UPDATE runs
                SET status = 'cancelled', updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND status IN ('pending', 'running')
                """,
            (str(run_id),),
        )
        cancelled = cur.rowcount > 0
    if cancelled:
        logger.info("run cancelled id=%s", run_id)
    return cancelled


def append_event(
    run_id: str, event_type: str, payload: dict | None = None
) -> RunEvent | None:
    """Persist one structural SSE event with the next monotonic seq.

    `delta` events return None without touching the DB (Q35: live-only).
    The parent run row is locked first, so the seq stays gap-free per run.
    KeyError on unknown run.
    """
    if event_type == _LIVE_ONLY:
        return None
    with pg_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT id FROM runs WHERE id = %s FOR UPDATE", (str(run_id),))
        if cur.fetchone() is None:
            raise KeyError(f"Unknown run: {run_id}")
        cur.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq "
            "FROM run_events WHERE run_id = %s",
            (str(run_id),),
        )
        seq = cur.fetchone()["next_seq"]
        cur.execute(
            """
                INSERT INTO run_events (run_id, seq, event_type, payload)
                VALUES (%s, %s, %s, %s)
                RETURNING id, run_id, seq, event_type, payload, created_at
                """,
            (str(run_id), seq, event_type, Json(payload or {})),
        )
        return _row_to_event(cur.fetchone())


def list_events(run_id: str) -> list[RunEvent]:
    """Replay a run's persisted events in seq order (structural only)."""
    with pg_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
                SELECT id, run_id, seq, event_type, payload, created_at
                FROM run_events WHERE run_id = %s ORDER BY seq
                """,
            (str(run_id),),
        )
        rows = cur.fetchall()
    return [_row_to_event(r) for r in rows]
