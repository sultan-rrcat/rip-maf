"""Canonical envelopes for the BFF contract.

Copied from Athena (Phase 4.1), adapted for RIP: the `approval_required`
event is gone (no approval gate — tools run directly) and `sources` is a
first-class event (Q32: emitted after `rag.query` completes).

Two directions, one owner (the BFF layer):
- RequestEnvelope: the canonical INBOUND shape (kept minimal — Q1 locks
  `POST /v1/runs` to exactly `{corpus_id, message}`, no envelope there).
- EventEnvelope: the canonical OUTBOUND shape — every SSE message carries a
  per-run monotonic `seq` that doubles as the SSE `id:` line, so a
  reconnecting client can dedupe by `seq` (no `?last_event_id=` in v1).

Event types (locked SSE vocabulary):
- "run_started":    the run was accepted and a worker began
- "plan":           the validated goal + steps (emitted plan-time)
- "step_started":   a step node began executing
- "step_completed": one step finished, with its result
- "delta":          streamed text token — LIVE ONLY, never persisted (Q35)
- "sources":        citations after a `rag.query` step (Q32)
- "artifacts":      file artifacts as download URLs (Q34; charts render
  inline as <img> over the same URL — never inline markup/base64)
- "summary":        the final aggregated answer text (≠ conversation_summary;
  chart steps contribute a placeholder, never raw SVG)
- "run_completed":  terminal success envelope
- "error":          the run aborted before/without a summary
- "cancelled":      terminal envelope after cooperative cancel
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel

EventType = Literal[
    "run_started",
    "plan",
    "step_started",
    "step_completed",
    "delta",
    "sources",
    "artifacts",
    "summary",
    "run_completed",
    "error",
    "cancelled",
]


class RequestEnvelope(BaseModel):
    message: str
    model: str | None = None
    trace_id: str | None = None
    conversation_id: str | None = None


class EventEnvelope(BaseModel):
    seq: int | str  # monotonic per run (1-based ints); live-only deltas use "N.K"
    run_id: str
    type: EventType
    data: dict
    ts: str  # ISO-8601 UTC timestamp

    @classmethod
    def build(
        cls, seq: int | str, run_id: str, type: EventType, data: dict
    ) -> EventEnvelope:
        return cls(
            seq=seq,
            run_id=run_id,
            type=type,
            data=data,
            ts=datetime.now(UTC).isoformat(),
        )

    def to_sse(self) -> str:
        # SSE frame with an id (== seq): "id: <seq>\ndata: <json>\n\n"
        payload = json.dumps(self.model_dump())
        return f"id: {self.seq}\ndata: {payload}\n\n"
