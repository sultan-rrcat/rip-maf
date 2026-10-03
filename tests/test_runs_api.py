"""Phase 4.1 — runs API + Q31 worker + Q32/Q34/Q35 events.

Needs a live Postgres with schema.sql applied (compose `postgres`
service; from the host: DB_HOST=127.0.0.1 DB_PORT=5433 — note 127.0.0.1,
`localhost` costs ~21s/connect on Windows). Skips when unreachable. Run
with ``pytest backend/tests/test_runs_api.py -q --noconftest`` until the
torch env is repaired.

Strategy: REAL Postgres store + REAL FastAPI routes, FAKE orchestrator and
provider (no LLM calls). The worker thread runs for real, so event
persistence, SSE replay, sources, artifacts, memory and cancel are all
exercised end to end.
"""

from __future__ import annotations

import json
import threading
import time
import uuid

import pytest

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rip_maf.agents.base import StepStatus
    from rip_maf.api.deps import get_run_manager
    from rip_maf.api.runs import router as runs_router
    from rip_maf.auth import Principal, require_principal
    from rip_maf.core.db import pg_connection
    from rip_maf.orchestration.orchestrator import (
        OrchestrationError,
        OrchestrationResult,
    )
    from rip_maf.orchestration.results import StepResult
    from rip_maf.runs import store
    from rip_maf.runs.manager import RunManager

    _IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001 - import probe; pragma: no cover
    _IMPORT_ERROR = e


def _db_up() -> bool:
    if _IMPORT_ERROR is not None:
        return False
    try:
        with pg_connection():
            pass
        return True
    except Exception:  # noqa: BLE001 - any connect failure means "down"
        return False


needs_db = pytest.mark.skipif(
    _IMPORT_ERROR is not None or not _db_up(),
    reason=f"runs api needs postgres ({_IMPORT_ERROR or 'connect failed'})",
)


class FakeProvider:
    def generate(self, model=None, messages=None, max_tokens=None, **kw):
        return "folded summary (fake)"


def _rag_step():
    return StepResult(
        step_id="s1",
        agent_id="rag.query",
        status=StepStatus.SUCCESS,
        output="chunks...",
        data={
            "results": [
                {"source": "doc.pdf", "section": "Intro", "content": "hello"},
                {"source": "doc.pdf", "section": "Intro", "content": "hello"},
                {"source": "other.pdf", "section": "Body", "content": "world"},
            ],
            "query": "hello",
        },
    )


def _chart_step():
    return StepResult(
        step_id="s2",
        agent_id="plot.chart",
        status=StepStatus.SUCCESS,
        output="<svg>...</svg>",
        data={"svg": "<svg xmlns='http://www.w3.org/2000/svg'></svg>"},
    )


def _result():
    return OrchestrationResult(
        trace_id="t1",
        plan_id="p1",
        goal="answer hello",
        step_results=[_rag_step(), _chart_step()],
        summary="final answer",
        status="success",
        plan_incomplete=False,
        conflicts=[],
        needs_clarification=False,
    )


class FakeOrchestrator:
    """Emits a realistic event chain, then returns a fixed result."""

    def run(self, request_text, corpus_id, on_event=None, context=None,
            cancel_event=None, **_kw):
        emit = on_event or (lambda d: None)
        emit({"type": "plan", "plan_id": "p1", "goal": "answer hello",
              "steps": [{"step_id": "s1", "executor": "rag.query",
                         "depends_on": []}]})
        emit({"type": "step_started", "step_id": "s1", "executor_id": "rag.query"})
        emit({"type": "delta", "step_id": "s1", "content": "hel"})
        emit({"type": "delta", "step_id": "s1", "content": "lo"})
        emit({"type": "step_completed", "step_id": "s1", "status": "success",
              "output": "chunks..."})
        return _result()


class BlockingOrchestrator:
    """Blocks until released or cancelled (cancel-path test)."""

    def __init__(self):
        self.release = threading.Event()

    def run(self, request_text, corpus_id, on_event=None, context=None,
            cancel_event=None, **_kw):
        emit = on_event or (lambda d: None)
        emit({"type": "plan", "plan_id": "p1", "goal": "g", "steps": []})
        while not self.release.is_set():
            if cancel_event is not None and cancel_event.is_set():
                raise OrchestrationError("run cancelled")
            time.sleep(0.02)
        return _result()


class BoomOrchestrator:
    def run(self, request_text, corpus_id, on_event=None, context=None,
            cancel_event=None, **_kw):
        raise RuntimeError("boom")


@pytest.fixture()
def test_user():
    """Owner for the fixture corpora (/v1/* is owner-scoped)."""
    owner_ref = f"runs-api-{uuid.uuid4().hex[:8]}"
    yield Principal(owner_ref=owner_ref, name=owner_ref)


@pytest.fixture()
def corpus_id(test_user):
    with pg_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO corpora (name, owner_ref, corpus_ref) "
            "VALUES (%s, %s, %s) RETURNING corpus_id",
            ("runs-api-test", test_user.owner_ref, uuid.uuid4().hex),
        )
        nb = str(cur.fetchone()[0])
    yield nb
    with pg_connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM corpora WHERE corpus_id = %s", (nb,))


@pytest.fixture()
def uploads(tmp_path, monkeypatch):
    """Point settings.upload_dir at a tmp dir (routes read settings live,
    mirroring production where worker and routes share the same root)."""
    from rip_maf.core.config import settings

    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    return str(tmp_path)


@pytest.fixture()
def manager(uploads):
    return RunManager(
        provider=FakeProvider(),
        orchestrator=FakeOrchestrator(),
    )


def _build_app(manager, user):
    """Bare /v1 app with run-manager + auth overridden (no login flow)."""
    app = FastAPI()
    app.include_router(runs_router)
    app.dependency_overrides[get_run_manager] = lambda: manager
    app.dependency_overrides[require_principal] = lambda: user
    return app


@pytest.fixture()
def client(manager, test_user):
    with TestClient(_build_app(manager, test_user)) as c:
        yield c


def _wait_done(run_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        row = store.get_run(run_id)
        if row is not None and row.status in (
            "completed", "failed", "cancelled",
        ):
            return row
        time.sleep(0.1)
    raise AssertionError(f"run {run_id} did not finish in {timeout}s")


def _frames(text):
    """Parse SSE text into [(id, data-dict)]."""
    out = []
    for chunk in text.split("\n\n"):
        lines = [ln for ln in chunk.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        sid, data = None, None
        for ln in lines:
            if ln.startswith("id:"):
                sid = ln[3:].strip()
            elif ln.startswith("data:"):
                data = json.loads(ln[5:].strip())
        if data is not None:
            out.append((sid, data))
    return out


@needs_db
class TestCreateAndReplay:
    def test_create_run_202(self, client, corpus_id):
        r = client.post(
            "/v1/runs", json={"corpus_id": corpus_id, "message": "hello"}
        )
        assert r.status_code == 202
        run_id = r.json()["run_id"]
        assert run_id and set(r.json()) == {"run_id"}  # bare, no envelope
        row = _wait_done(run_id)
        assert row.status == "completed" and row.goal == "answer hello"

    def test_events_replay_no_deltas(self, client, corpus_id):
        run_id = client.post(
            "/v1/runs", json={"corpus_id": corpus_id, "message": "hello"}
        ).json()["run_id"]
        _wait_done(run_id)
        r = client.get(f"/v1/runs/{run_id}/events")
        assert r.status_code == 200
        frames = _frames(r.text)
        types = [d["type"] for _, d in frames]
        for expected in (
            "run_started", "plan", "step_started", "step_completed",
            "sources", "summary", "artifacts", "run_completed",
        ):
            assert expected in types, f"missing {expected} in {types}"
        assert "delta" not in types  # Q35: live-only, never replayed
        seqs = [s for s, _ in frames]
        assert seqs == sorted(seqs, key=int)  # gap-free persisted order
        assert all(d["run_id"] == run_id for _, d in frames)

    def test_sources_shape(self, client, corpus_id):
        run_id = client.post(
            "/v1/runs", json={"corpus_id": corpus_id, "message": "hello"}
        ).json()["run_id"]
        _wait_done(run_id)
        frames = _frames(client.get(f"/v1/runs/{run_id}/events").text)
        src = [d for _, d in frames if d["type"] == "sources"]
        assert len(src) == 1 and src[0]["step_id"] == "s1"
        # Q32 via extract_sources(): deduped (set-based, order not promised).
        assert sorted(map(json.dumps, src[0]["sources"],)) == sorted(
            map(json.dumps, [
                {"source": "doc.pdf", "section": "Intro"},
                {"source": "other.pdf", "section": "Body"},
            ])
        )

    def test_store_has_no_deltas(self, client, corpus_id):
        run_id = client.post(
            "/v1/runs", json={"corpus_id": corpus_id, "message": "hello"}
        ).json()["run_id"]
        _wait_done(run_id)
        types = [e.event_type for e in store.list_events(run_id)]
        assert "delta" not in types

    def test_detail(self, client, corpus_id):
        run_id = client.post(
            "/v1/runs", json={"corpus_id": corpus_id, "message": "hello"}
        ).json()["run_id"]
        _wait_done(run_id)
        r = client.get(f"/v1/runs/{run_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["run_id"] == run_id and body["status"] == "completed"
        assert body["corpus_id"] == corpus_id


@needs_db
class TestValidation:
    def test_empty_message_422(self, client, corpus_id):
        r = client.post(
            "/v1/runs", json={"corpus_id": corpus_id, "message": "   "}
        )
        assert r.status_code == 422

    def test_unknown_corpus_404(self, client):
        r = client.post(
            "/v1/runs",
            json={"corpus_id": str(uuid.uuid4()), "message": "hi"},
        )
        assert r.status_code == 404

    def test_unknown_run_404(self, client):
        bad = str(uuid.uuid4())
        assert client.get(f"/v1/runs/{bad}").status_code == 404
        assert client.get(f"/v1/runs/{bad}/events").status_code == 404
        assert client.post(f"/v1/runs/{bad}/cancel").status_code == 404


@needs_db
class TestArtifacts:
    def test_artifact_download(self, client, corpus_id):
        run_id = client.post(
            "/v1/runs", json={"corpus_id": corpus_id, "message": "hello"}
        ).json()["run_id"]
        _wait_done(run_id)
        frames = _frames(client.get(f"/v1/runs/{run_id}/events").text)
        arts = [d for _, d in frames if d["type"] == "artifacts"]
        assert len(arts) == 1
        entry = arts[0]["artifacts"][0]
        assert set(entry) >= {"artifact_id", "kind", "filename", "url"}
        assert "base64" not in json.dumps(arts[0])  # Q34: links, not bytes
        dl = client.get(entry["url"])
        assert dl.status_code == 200
        assert "<svg" in dl.text

    def test_unknown_artifact_404(self, client, corpus_id):
        run_id = client.post(
            "/v1/runs", json={"corpus_id": corpus_id, "message": "hello"}
        ).json()["run_id"]
        _wait_done(run_id)
        r = client.get(f"/v1/runs/{run_id}/artifacts/nope")
        assert r.status_code == 404


@needs_db
class TestStatelessHistory:
    def test_run_accepts_caller_history(self, client, corpus_id):
        """Open WebUI owns history; the run stores none of its own."""
        run_id = client.post(
            "/v1/runs",
            json={
                "corpus_id": corpus_id,
                "message": "hello",
                "history": [
                    {"role": "user", "content": "previous question"},
                    {"role": "assistant", "content": "previous answer"},
                ],
            },
        ).json()["run_id"]
        _wait_done(run_id)
        detail = client.get(f"/v1/runs/{run_id}").json()
        assert detail["status"] in ("completed", "failed")


@needs_db
class TestCancel:
    def test_cancel_running_run(self, corpus_id, uploads, test_user):
        orch = BlockingOrchestrator()
        manager = RunManager(
            provider=FakeProvider(), orchestrator=orch,
        )
        app = _build_app(manager, test_user)
        with TestClient(app) as client:
            run_id = client.post(
                "/v1/runs", json={"corpus_id": corpus_id, "message": "hi"}
            ).json()["run_id"]
            time.sleep(0.5)  # let the worker block inside orchestration
            r = client.post(f"/v1/runs/{run_id}/cancel")
            assert r.status_code == 200
            body = r.json()
            assert body["run_id"] == run_id and body["already_done"] is False
            row = _wait_done(run_id)
            assert row.status == "cancelled"
            assert body["status"] == "cancelled"
            # The row flips to 'cancelled' before the worker publishes the
            # terminal event; poll briefly for it to land.
            types: list[str] = []
            for _ in range(50):
                types = [e.event_type for e in store.list_events(run_id)]
                if "cancelled" in types:
                    break
                time.sleep(0.1)
            assert "cancelled" in types and "run_completed" not in types

    def test_cancel_finished_run_already_done(self, client, corpus_id):
        run_id = client.post(
            "/v1/runs", json={"corpus_id": corpus_id, "message": "hello"}
        ).json()["run_id"]
        _wait_done(run_id)
        r = client.post(f"/v1/runs/{run_id}/cancel")
        assert r.status_code == 200 and r.json()["already_done"] is True
        assert store.get_run(run_id).status == "completed"  # never overwritten


@needs_db
class TestFailure:
    def test_crash_becomes_error(self, corpus_id, uploads, test_user):
        manager = RunManager(
            provider=FakeProvider(), orchestrator=BoomOrchestrator(),
        )
        app = _build_app(manager, test_user)
        with TestClient(app) as client:
            run_id = client.post(
                "/v1/runs", json={"corpus_id": corpus_id, "message": "hi"}
            ).json()["run_id"]
            row = _wait_done(run_id)
            assert row.status == "failed"
            types = [e.event_type for e in store.list_events(run_id)]
            assert "error" in types and "run_completed" not in types


@needs_db
class TestLiveDeltas:
    def test_delta_live_only_fractional_seq(self, corpus_id, uploads):
        orch = BlockingOrchestrator()
        manager = RunManager(
            provider=FakeProvider(), orchestrator=orch,
        )
        record = manager.create_run(corpus_id, "hi")
        time.sleep(0.5)  # worker is blocked; run_started+plan persisted
        events, live, done = manager.subscribe(record.run_id)
        assert done is False
        assert [e.event_type for e in events] == ["run_started", "plan"]
        # Inject live deltas straight through the worker's publish path.
        manager._publish(record, "delta", {"step_id": "s1", "content": "tok"})
        frame = live.get(timeout=5)
        assert frame["type"] == "delta" and "." in str(frame["seq"])  # N.K
        orch.release.set()
        while True:  # drain to close
            item = live.get(timeout=15)
            if item is None:
                break
        _wait_done(record.run_id)
        assert "delta" not in [e.event_type for e in store.list_events(record.run_id)]



