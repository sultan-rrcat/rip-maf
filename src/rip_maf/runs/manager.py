"""Run worker (Q31 REWRITE — not Athena's manager verbatim).

One run = one user message = one worker thread driving
`orchestrator.run()` to completion. The Q31 contract:

1. **Always orchestrate** — no complexity classifier, no direct shortcut.
2. **Never write `messages`** — the frontend owns user/assistant rows (Q22).
3. **Load memory before run** — `corpora.conversation_summary` +
   `summary_message_count` + `messages` → `build_memory_context()` →
   `context=memory.as_prompt()`. A fold failure degrades to `context=None`,
   never kills the run.
4. **Persist memory after run** — updated summary/count back to `corpora`.
5. **Emit `sources` on `rag.query`** — per completed rag step, via
   `extract_sources()` (Q32); persisted like any structural event.
6. **Event persistence (Q35)** — structural events go to `run_events` and
   are replayed from Postgres on reconnect (survives restart/eviction);
   `delta` is fanned out to live subscribers only, never INSERTed. Live
   deltas carry fractional seqs (`"<persisted>.<k>"`) which provably never
   collide with the 1-based persisted seqs, so the persisted log stays
   gap-free while every live frame still has a unique id.
7. **File artifacts (Q34)** — tool outputs written to disk; SSE `artifacts`
   carries download URLs, never inline base64.

Terminal sequence: `update_run` row first (so `GET /v1/runs/{id}` sees the
final state), then the terminal SSE event, then close. A late cancel never
overwrites a final state (`store.cancel_run` is conditional).
"""
from __future__ import annotations

import logging
import queue
import threading

from rip_maf.artifacts import collect_artifacts
from rip_maf.core.db import pg_connection
from rip_maf.envelope import EventType
from rip_maf.observability.langfuse import (
    flush as langfuse_flush,
)
from rip_maf.observability.langfuse import (
    manual_span,
    request_attributes,
    truncate,
)
from rip_maf.orchestration.memory import build_history_context
from rip_maf.orchestration.orchestrator import OrchestrationError
from rip_maf.runs import store as run_store
from rip_maf.services.chat import extract_sources

logger = logging.getLogger("runs.manager")

# Registry cap (completed runs included): keeps the process bounded without
# a sweeper thread; oldest live handles are dropped first. Durability is in
# Postgres — replay always reads `run_events`, never this registry.
MAX_RUNS = 50

# Concurrency limit: max 4 active run workers (matches single Ollama + 2-parallel-writes rule)
MAX_ACTIVE_RUNS = 4

# Run timeout: 10 minutes fixed
RUN_TIMEOUT_S = 600

_SENTINEL = None  # stream-end marker for subscriber queues

_TERMINAL = ("completed", "failed", "cancelled")


class RunRecord:
    """Live handle for one run: cancel flag, subscribers, done flag."""

    def __init__(
        self,
        run_id: str,
        corpus_id: str,
        message: str,
        history: list[dict] | None = None,
    ):
        self.run_id = run_id
        self.corpus_id = corpus_id
        self.message = message
        # rip-maf stateless mode: caller-supplied history (Open WebUI owns the
        # conversation). None = legacy DB-memory path (React frontend).
        self.history = history
        self.cancel_event = threading.Event()
        self._lock = threading.Lock()
        self._subscribers: set[queue.Queue] = set()
        self._done = threading.Event()
        # Delta seq state: last persisted seq + per-gap counter. Touched only
        # by the run's single worker thread — no lock needed.
        self._last_seq = 0
        self._delta_n = 0

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def cancel(self) -> None:
        self.cancel_event.set()

    def _close(self) -> None:
        """Terminal: wake every subscriber queue; late subscribers see done."""
        with self._lock:
            for sub in self._subscribers:
                sub.put(_SENTINEL)
            self._done.set()

    def _detach(self, sub: queue.Queue) -> None:
        with self._lock:
            self._subscribers.discard(sub)

    def fan_out(self, event: dict) -> None:
        """Push one event dict to every live subscriber (never blocks)."""
        with self._lock:
            for sub in self._subscribers:
                sub.put(event)

    def attach(self) -> tuple[queue.Queue, bool]:
        """Attach a live subscriber; returns (queue, done_snapshot).

        The lock is held across the done-check + registration so no terminal
        event slips between a replay snapshot and going live (the SSE route
        replays from Postgres first, then calls this).
        """
        with self._lock:
            live: queue.Queue = queue.Queue(maxsize=1000)
            done_snapshot = self._done.is_set()
            if done_snapshot:
                live.put(_SENTINEL)
            else:
                self._subscribers.add(live)
            return live, done_snapshot


class RunManager:
    """Registry of live runs + the Q31 worker-thread spawner."""

    def __init__(self, *, provider, orchestrator, upload_dir: str | None = None):
        self._provider = provider
        self._orchestrator = orchestrator
        self._upload_dir = upload_dir  # None → settings.upload_dir read live
        self._runs: dict[str, RunRecord] = {}
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(MAX_ACTIVE_RUNS)

    @property
    def _root(self) -> str:
        if self._upload_dir is not None:
            return self._upload_dir
        from rip_maf.core.config import settings

        return settings.upload_dir

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._runs.get(str(run_id))

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for r in self._runs.values() if not r.done)

    def create_run(
        self,
        corpus_id: str,
        message: str,
        history: list[dict] | None = None,
    ) -> RunRecord:
        """Create a `pending` run row and spawn its worker (Q1 contract).

        ValueError on empty message (defense in depth — routes 422 first);
        LookupError on unknown corpus (routes map to 404).
        RuntimeError if max active runs reached (routes map to 429).

        `history` (rip-maf): caller-supplied conversation turns; when given,
        the worker is stateless and ignores the DB message/summary tables.
        """
        text = (message or "").strip()
        if not text:
            raise ValueError("message must not be empty")
        corpus_id = str(corpus_id)
        if not self._corpus_exists(corpus_id):
            raise LookupError(f"Unknown corpus: {corpus_id}")

        if not self._semaphore.acquire(blocking=False):
            raise RuntimeError("Max concurrent runs reached")

        try:
            row = run_store.create_run(corpus_id)
            run_store.update_run(row.id, status="running")
            record = RunRecord(
                run_id=row.id,
                corpus_id=corpus_id,
                message=text,
                history=history,
            )
            with self._lock:
                while len(self._runs) >= MAX_RUNS:
                    oldest = next(iter(self._runs))
                    logger.info("evicting run %s (registry full)", oldest)
                    del self._runs[oldest]
                self._runs[record.run_id] = record
            threading.Thread(
                target=self._worker,
                args=(record,),
                name=f"rip-run-{record.run_id[:8]}",
                daemon=True,
            ).start()
            logger.info("run started id=%s corpus=%s", record.run_id, corpus_id)
            return record
        except Exception:
            self._semaphore.release()
            raise

    def cancel_run(self, run_id: str) -> tuple[bool, bool]:
        """Cooperative cancel. Returns (handled, already_done).

        handled=False means unknown (no live record, no row — routes 404).
        already_done=True means the run was already terminal; a late cancel
        never overwrites a final state.
        """
        run_id = str(run_id)
        record = self.get(run_id)
        if record is not None:
            record.cancel()
        row = run_store.get_run(run_id)
        if row is None:
            return False, False
        if row.status in _TERMINAL:
            return True, True
        run_store.cancel_run(run_id)
        logger.info("run cancel requested id=%s", run_id)
        return True, False

    def subscribe(self, run_id: str):
        """Replay-from-Postgres snapshot point + live queue for SSE.

        Returns (events, queue, done). `events` are `RunEvent`s in seq order
        (structural only — deltas were never stored, Q35). Raises LookupError
        on unknown run. Works after eviction/restart: replay needs no live
        record, only the row.
        """
        run_id = str(run_id)
        if run_store.get_run(run_id) is None and self.get(run_id) is None:
            raise LookupError(f"Unknown run: {run_id}")
        events = run_store.list_events(run_id)
        record = self.get(run_id)
        if record is None:
            # Evicted or restarted run: replay is complete, end right after.
            live: queue.Queue = queue.Queue()
            live.put(_SENTINEL)
            return events, live, True
        live, done = record.attach()
        return events, live, done

    # -- worker ------------------------------------------------------------

    def _publish(
        self, record: RunRecord, type: EventType, data: dict, *, persist: bool = True
    ) -> dict:
        """Persist (unless live-only) + fan out one event; return it.

        The returned dict is `{"seq", "type", "data"}` — routes format the
        SSE frame from it, and replay rows are reshaped to the same form.
        """
        if type == "delta" or not persist:
            # Live-only: fractional seq provably outside the persisted
            # 1-based space — unique per frame, zero DB touch (Q35).
            record._delta_n += 1
            event = {
                "seq": f"{record._last_seq}.{record._delta_n}",
                "type": type,
                "data": data,
            }
            record.fan_out(event)
            return event
        stored = run_store.append_event(record.run_id, type, data)
        record._last_seq = stored.seq
        record._delta_n = 0
        event = {"seq": stored.seq, "type": type, "data": data}
        record.fan_out(event)
        return event

    def _on_event(self, record: RunRecord, d: dict) -> None:
        """Orchestrator callback: persist structural events, stream deltas."""
        type = d.get("type")
        data = {k: v for k, v in d.items() if k != "type"}
        self._publish(record, type, data)

    def _worker(self, record: RunRecord) -> None:
        # One Langfuse trace per run (Athena pattern: one trace per
        # request). session_id = corpus (one corpus = one conversation),
        # so a corpus's traces group into one Langfuse session. Opened
        # HERE in the worker thread: the plan graph copies this thread's
        # contextvars per node, so engine/step/generation spans auto-parent.
        # All no-ops when tracing is disabled.

        # Fixed 10-minute run timeout: cancel cooperatively via cancel_event.
        timeout_timer = threading.Timer(RUN_TIMEOUT_S, self._on_timeout, args=(record,))
        timeout_timer.daemon = True
        timeout_timer.start()

        try:
            self._publish(
                record,
                "run_started",
                {"run_id": record.run_id, "corpus_id": record.corpus_id,
                 "message": record.message},
            )
            with request_attributes(
                session_id=record.corpus_id,
                user_id="local",
                metadata={"route": "runs", "run_id": record.run_id},
                tags=["feature:runs"],
                trace_name="run",
            ), manual_span(
                "run", as_type="span", input=truncate(record.message, 2000)
            ) as run_obs:
                self._run_traced(record, run_obs)
        except Exception as e:
            logger.exception("run %s crashed", record.run_id)
            self._finish_failed(record, str(e))
        finally:
            timeout_timer.cancel()
            langfuse_flush()
            record._close()
            self._semaphore.release()

    def _on_timeout(self, record: RunRecord) -> None:
        """Cancel a run that exceeded the fixed timeout."""
        logger.warning("run %s timed out after %ds", record.run_id, RUN_TIMEOUT_S)
        record.cancel()

    def _run_traced(self, record: RunRecord, run_obs) -> None:
        """Worker body inside the trace root (split for readability)."""
        try:
            context, corpus_context = self._load_memory(record)
            try:
                result = self._orchestrator.run(
                    record.message,
                    record.corpus_id,
                    on_event=lambda d: self._on_event(record, d),
                    context=context,
                    cancel_event=record.cancel_event,
                    corpus_context=corpus_context,
                )
            except OrchestrationError as e:
                if record.cancel_event.is_set():
                    self._finish_cancelled(record, str(e))
                else:
                    self._finish_failed(record, str(e))
                run_obs.update(output={"status": "failed", "error": str(e)[:500]})
                return

            # Q32: one `sources` event per completed rag.query step, in plan
            # order, via extract_sources() over the step's retrieved chunks.
            for step in result.step_results or []:
                if (
                    getattr(step.agent_id, "value", step.agent_id) == "rag.query"
                    and getattr(step.status, "value", step.status) == "success"
                    and isinstance(getattr(step, "data", None), dict)
                    and isinstance(step.data.get("results"), list)
                ):
                    sources = extract_sources({"results": step.data["results"]})
                    if sources:
                        self._publish(
                            record, "sources",
                            {"step_id": step.step_id, "sources": sources},
                        )

            # Q34: tool outputs to disk; SSE carries download URLs only.
            artifacts = collect_artifacts(
                result.step_results or [],
                upload_dir=self._root,
                corpus_id=record.corpus_id,
                run_id=record.run_id,
            )
            if artifacts:
                self._publish(record, "artifacts", {"artifacts": artifacts})

            self._publish(
                record,
                "summary",
                {
                    "content": result.summary,
                    "status": result.status,
                    "plan_incomplete": result.plan_incomplete,
                    "conflicts": result.conflicts,
                    "needs_clarification": result.needs_clarification,
                    "shown": list(getattr(result, "shown", []) or []),
                    "hidden": list(getattr(result, "hidden", []) or []),
                    "visibility": dict(getattr(result, "visibility", {}) or {}),
                },
            )
            terminal = "completed" if result.status in ("success", "partial") else "failed"
            run_store.update_run(
                record.run_id,
                status=terminal,
                goal=result.goal,
                plan={"plan_id": result.plan_id, "goal": result.goal},
                result={
                    "summary": result.summary,
                    "status": result.status,
                    "plan_incomplete": result.plan_incomplete,
                    "conflicts": result.conflicts,
                    "needs_clarification": result.needs_clarification,
                    "shown": list(getattr(result, "shown", []) or []),
                    "hidden": list(getattr(result, "hidden", []) or []),
                    "visibility": dict(getattr(result, "visibility", {}) or {}),
                },
            )
            self._publish(record, "run_completed",
                           {"status": result.status, "run_id": record.run_id})
            run_obs.update(output={
                "status": result.status,
                "summary": truncate(result.summary, 2000),
            })
        except Exception as e:
            logger.exception("run %s crashed", record.run_id)
            run_obs.update(output={"status": "failed", "error": str(e)[:500]})
            self._finish_failed(record, str(e))

    def _finish_failed(self, record: RunRecord, message: str) -> None:
        try:
            run_store.update_run(record.run_id, status="failed")
        except Exception:
            logger.exception("run %s: failed-state persist failed", record.run_id)
        try:
            self._publish(record, "error", {"message": message})
        except Exception:
            logger.exception("run %s: error event persist failed", record.run_id)

    def _finish_cancelled(self, record: RunRecord, reason: str) -> None:
        try:
            run_store.cancel_run(record.run_id)
        except Exception:
            logger.exception("run %s: cancelled-state persist failed", record.run_id)
        try:
            self._publish(record, "cancelled", {"reason": reason})
        except Exception:
            logger.exception("run %s: cancelled event persist failed", record.run_id)

    # -- memory (Q31 steps 3-4) --------------------------------------------

    def _corpus_exists(self, corpus_id: str) -> bool:
        with pg_connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM corpora WHERE corpus_id = %s", (corpus_id,)
            )
            return cur.fetchone() is not None

    def _load_memory(self, record: RunRecord) -> tuple[str | None, str | None]:
        """Render the caller-supplied history + the corpus file inventory.

        Open WebUI owns conversation history, so RIP stores none: the Pipe
        forwards prior turns and they are rendered verbatim (newest window).
        Returns (prompt context, corpus file snapshot); the snapshot is e.g.
        "2 file(s): a.pdf (ready)" or "(no documents)" and degrades to None.
        """
        context = build_history_context(record.history or [])
        return context, self._load_file_snapshot(record.corpus_id)

    def _load_file_snapshot(self, corpus_id: str) -> str | None:
        """Best-effort file inventory string for the planner (None on failure)."""
        try:
            with pg_connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT file_id, file_name, file_status FROM files "
                    "WHERE corpus_id = %s ORDER BY created_at ASC",
                    (str(corpus_id),),
                )
                rows = cur.fetchall()
        except Exception:
            logger.exception("file snapshot failed, continuing without it")
            return None
        if not rows:
            return "(no documents)"
        parts = []
        for r in rows:
            fid, name, status = str(r[0]), r[1] or "?", r[2] or "?"
            parts.append(f"{name} [{status}] id={fid}")
        return f"{len(parts)} file(s): " + "; ".join(parts)

def get_run_manager() -> RunManager:
    """Process-wide singleton — runs must be addressable across requests.

    Canonical home is `app.api.deps` (routes resolve the manager through it,
    so FastAPI `dependency_overrides` work); this delegates there so the two
    never diverge into separate instances.
    """
    from rip_maf.api.deps import get_run_manager as _deps_manager

    manager = _deps_manager()
    assert isinstance(manager, RunManager)
    return manager


def set_run_manager(manager: RunManager) -> None:
    """Install the process manager (main.py lifespan; tests).

    There is no uninstall-single-field path by design — teardown uses
    `app.api.deps.reset()`. Passing None raises (use `deps.reset()`).
    """
    if manager is None:
        raise ValueError("use app.api.deps.reset() to clear the runtime")
    from rip_maf.api import deps as _deps

    _deps.configure(run_manager=manager)


__all__ = ["RunManager", "RunRecord", "get_run_manager", "set_run_manager"]
