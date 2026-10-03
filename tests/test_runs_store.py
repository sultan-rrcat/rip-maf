"""Phase 3.3 — runs store: Postgres CRUD + Q35 replay.

Needs a live Postgres with schema.sql applied (compose `postgres`
service; from the host: DB_HOST=localhost DB_PORT=5433 DB_NAME=rip
DB_USER=rip DB_PASSWORD=rippass). Skips when unreachable. Run with
``pytest backend/tests/test_runs_store.py -q --noconftest`` until the
torch env is repaired — the shared conftest imports app.main, which needs
sentence_transformers.
"""

from __future__ import annotations

import pytest

try:
    import psycopg2
    from psycopg2 import errors as _pg_errors

    from rip_maf.core.db import pg_connection
    from rip_maf.runs import store

    _IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001 - import probe; pragma: no cover
    psycopg2 = None  # type: ignore[assignment]
    _pg_errors = None  # type: ignore[assignment]
    pg_connection = None  # type: ignore[assignment]
    store = None  # type: ignore[assignment]
    _IMPORT_ERROR = e


def _db_up() -> bool:
    if pg_connection is None:
        return False
    try:
        with pg_connection():
            pass
        return True
    except Exception:  # noqa: BLE001 - any connect failure means "down"
        return False


needs_db = pytest.mark.skipif(
    _IMPORT_ERROR is not None or not _db_up(),
    reason=f"postgres unreachable ({_IMPORT_ERROR or 'connect failed'})",
)


@pytest.fixture()
def corpus_id():
    with pg_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO corpora (name, owner_ref, corpus_ref) "
            "VALUES (%s, %s, %s) RETURNING corpus_id",
            ("store-test", "pytest", "store-test"),
        )
        nb = str(cur.fetchone()[0])
    yield nb
    with pg_connection() as conn, conn.cursor() as cur:
        # CASCADE wipes this corpus's files/messages/runs/run_events.
        cur.execute("DELETE FROM corpora WHERE corpus_id = %s", (nb,))


@needs_db
class TestRuns:
    def test_create_run_pending(self, corpus_id):
        run = store.create_run(corpus_id)
        assert run.status == "pending"
        assert run.corpus_id == corpus_id
        assert run.goal is None and run.plan is None
        assert run.created_at is not None

    def test_create_run_unknown_corpus_fails(self):
        with pytest.raises(_pg_errors.ForeignKeyViolation):
            store.create_run("00000000-0000-0000-0000-000000000000")

    def test_get_run_roundtrip(self, corpus_id):
        created = store.create_run(corpus_id)
        fetched = store.get_run(created.id)
        assert fetched is not None and fetched.id == created.id
        assert fetched.status == "pending"

    def test_get_run_unknown_returns_none(self):
        assert store.get_run("00000000-0000-0000-0000-000000000000") is None

    def test_update_run_fields(self, corpus_id):
        run = store.create_run(corpus_id)
        updated = store.update_run(
            run.id, status="running", goal="answer things",
            plan={"steps": []}, result={"summary": "x"},
        )
        assert updated.status == "running"
        assert updated.goal == "answer things"
        assert updated.plan == {"steps": []} and updated.result == {"summary": "x"}
        assert store.get_run(run.id).status == "running"

    def test_update_run_bad_status(self, corpus_id):
        run = store.create_run(corpus_id)
        with pytest.raises(ValueError):
            store.update_run(run.id, status="exploding")

    def test_update_run_unknown_raises(self):
        with pytest.raises(KeyError):
            store.update_run("00000000-0000-0000-0000-000000000000", status="failed")

    def test_cancel_pending_run(self, corpus_id):
        run = store.create_run(corpus_id)
        assert store.cancel_run(run.id) is True
        assert store.get_run(run.id).status == "cancelled"

    def test_cancel_terminal_run_is_noop(self, corpus_id):
        run = store.create_run(corpus_id)
        store.update_run(run.id, status="completed")
        assert store.cancel_run(run.id) is False
        assert store.get_run(run.id).status == "completed"

    def test_cancel_unknown_run_is_noop(self):
        assert store.cancel_run("00000000-0000-0000-0000-000000000000") is False

    def test_list_runs_isolated_and_ordered(self, corpus_id):
        first = store.create_run(corpus_id)
        second = store.create_run(corpus_id)
        ids = [r.id for r in store.list_runs(corpus_id)]
        assert ids == [first.id, second.id]
        with pg_connection() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO corpora (name, owner_ref, corpus_ref) "
                "VALUES (%s, %s, %s) RETURNING corpus_id",
                ("store-other", "pytest", "store-other"),
            )
            other = str(cur.fetchone()[0])
        try:
            assert [r.id for r in store.list_runs(other)] == []
        finally:
            with pg_connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM corpora WHERE corpus_id = %s", (other,)
                )


@needs_db
class TestRunEvents:
    def test_append_and_replay_in_order(self, corpus_id):
        run = store.create_run(corpus_id)
        e1 = store.append_event(run.id, "run_started", {"run_id": run.id})
        e2 = store.append_event(run.id, "plan", {"goal": "g", "steps": []})
        e3 = store.append_event(
            run.id, "step_completed",
            {"step_id": "1", "status": "success", "output": "hi"},
        )
        assert [e.seq for e in (e1, e2, e3)] == [1, 2, 3]
        replayed = store.list_events(run.id)
        assert [e.event_type for e in replayed] == [
            "run_started", "plan", "step_completed",
        ]
        assert [e.seq for e in replayed] == [1, 2, 3]
        assert replayed[2].payload["output"] == "hi"

    def test_delta_not_persisted_no_gap(self, corpus_id):
        run = store.create_run(corpus_id)
        store.append_event(run.id, "step_started", {"step_id": "1"})
        assert store.append_event(run.id, "delta", {"content": "tok"}) is None
        store.append_event(run.id, "step_completed", {"step_id": "1"})
        replayed = store.list_events(run.id)
        assert [e.event_type for e in replayed] == ["step_started", "step_completed"]
        assert [e.seq for e in replayed] == [1, 2]  # skipped delta consumes no seq

    def test_append_unknown_run_raises(self):
        with pytest.raises(KeyError):
            store.append_event(
                "00000000-0000-0000-0000-000000000000", "plan", {}
            )

    def test_events_visible_across_connections(self, corpus_id):
        # Every store call opens its own connection: create + append here,
        # read back through fresh connections — durability, not buffers.
        run = store.create_run(corpus_id)
        store.append_event(run.id, "summary", {"content": "answer"})
        assert store.get_run(run.id).status == "pending"
        replayed = store.list_events(run.id)
        assert len(replayed) == 1 and replayed[0].payload["content"] == "answer"
