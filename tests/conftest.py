"""Shared fixtures for integration tests.

Real stack, no mocks: real Postgres (same DB as the app), real ML models
(via the app lifespan), real HTTP through TestClient, plus a real Ollama
probe for the prompt tests (skipped when unreachable).

There is no login flow: every request carries the service-key headers.
"""

import os
import shutil
import sys
import uuid

import httpx
import pytest
from fastapi.testclient import TestClient

# Make `src/` importable when pytest runs from the project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from rip_maf.core.config import Settings, settings
from rip_maf.core.db import pg_connection

TEST_SERVICE_KEY = "test-service-key"
AUTH_HEADERS = {
    "X-RIP-Service-Key": TEST_SERVICE_KEY,
    "X-RIP-User-Id": "pytest-user",
    "X-RIP-User-Name": "pytest",
}
settings.rip_service_key = TEST_SERVICE_KEY


def _apply_schema():
    """Apply schema.sql idempotently (CREATE TABLE/INDEX IF NOT EXISTS)."""
    schema_path = os.path.join(PROJECT_ROOT, "schema.sql")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    with pg_connection() as conn, conn.cursor() as cur:
        cur.execute(schema_sql)


@pytest.fixture(scope="session")
def client():
    """TestClient with lifespan executed: real VectorRAG models loaded once."""
    try:
        _apply_schema()
    except Exception as e:  # noqa: BLE001 - any connect/DDL failure means "down"
        pytest.skip(f"Postgres unreachable, skipping integration tests: {e}")
    from rip_maf.main import app

    with TestClient(app, headers=AUTH_HEADERS) as test_client:
        yield test_client

    # Deterministic teardown: release torch models / CUDA cache before exit.
    try:
        import gc

        import torch

        app.state.rag = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001, S110 - teardown probes must never fail the suite
        pass


@pytest.fixture()
def test_corpus(client):
    """A fresh corpus per test; cleaned up afterwards."""
    chat_id = f"pytest-{uuid.uuid4().hex[:8]}"
    response = client.post("/v1/corpus", json={"chat_id": chat_id})
    assert response.status_code == 200, response.text
    corpus_id = response.json()["corpus_id"]
    yield corpus_id
    # Teardown: DB cascade handles files/embeddings.
    try:
        with pg_connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM corpora WHERE corpus_id = %s", (corpus_id,))
    except Exception:  # noqa: BLE001, S110 - best-effort teardown
        pass
    corpus_dir = os.path.join(settings.upload_dir, corpus_id)
    shutil.rmtree(corpus_dir, ignore_errors=True)


def llm_available() -> bool:
    """Probe the real Ollama endpoint; prompt tests skip when it is down."""
    try:
        base_url = Settings().ollama_base_url.rstrip("/")
    except Exception:  # noqa: BLE001 - any settings failure means "down"
        return False
    if not base_url:
        return False
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=10, trust_env=False)
        return response.status_code < 500
    except Exception:  # noqa: BLE001 - any probe failure means "down"
        return False


needs_llm = pytest.mark.skipif(
    not llm_available(), reason="Ollama endpoint unavailable"
)
