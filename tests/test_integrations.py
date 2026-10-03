"""Unit tests for the service-key auth + corpus surface (no DB / no LLM)."""
from __future__ import annotations

import os
import sys
import uuid

import pytest
from starlette.requests import Request

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from rip_maf.api.runs import CreateRunRequest
from rip_maf.auth import (
    SERVICE_KEY_HEADER,
    SERVICE_USER_ID_HEADER,
    Principal,
    require_principal,
)


def _request(headers: dict[str, str]) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [
                (k.lower().encode("latin-1"), v.encode("latin-1"))
                for k, v in headers.items()
            ],
        }
    )


# ── service-key principal ───────────────────────────────────────────────────


def test_principal_rejects_missing_key(monkeypatch):
    monkeypatch.setattr("rip_maf.auth.settings.rip_service_key", "secret")
    with pytest.raises(Exception) as exc:
        require_principal(_request({}))
    assert getattr(exc.value, "status_code", None) == 401


def test_principal_rejects_bad_key(monkeypatch):
    monkeypatch.setattr("rip_maf.auth.settings.rip_service_key", "secret")
    with pytest.raises(Exception) as exc:
        require_principal(_request({SERVICE_KEY_HEADER: "wrong"}))
    assert getattr(exc.value, "status_code", None) == 401


def test_principal_requires_user_id(monkeypatch):
    monkeypatch.setattr("rip_maf.auth.settings.rip_service_key", "secret")
    with pytest.raises(Exception) as exc:
        require_principal(_request({SERVICE_KEY_HEADER: "secret"}))
    assert getattr(exc.value, "status_code", None) == 401


def test_principal_maps_user_with_valid_key(monkeypatch):
    monkeypatch.setattr("rip_maf.auth.settings.rip_service_key", "secret")
    principal = require_principal(
        _request(
            {
                SERVICE_KEY_HEADER: "secret",
                SERVICE_USER_ID_HEADER: "owui-123",
                "X-RIP-User-Name": "Alice",
            }
        )
    )
    assert isinstance(principal, Principal)
    assert principal.owner_ref == "owui-123"
    assert principal.name == "Alice"


# ── CreateRunRequest history ────────────────────────────────────────────────


def test_create_run_request_accepts_history():
    body = CreateRunRequest.model_validate(
        {
            "corpus_id": str(uuid.uuid4()),
            "message": "hi",
            "history": [{"role": "user", "content": "a"}],
        }
    )
    assert body.history == [{"role": "user", "content": "a"}]


def test_create_run_request_rejects_empty_message():
    with pytest.raises(ValueError):
        CreateRunRequest.model_validate(
            {"corpus_id": str(uuid.uuid4()), "message": "   "}
        )


# ── corpus ensure (fake DB) ─────────────────────────────────────────────────


class _FakeCursor:
    def __init__(self):
        self._next = None

    def execute(self, sql, params=None):
        up = sql.strip().upper()
        if "SELECT CORPUS_ID" in up:
            self._next = None  # not found → create
        elif "INSERT INTO CORPORA" in up:
            self._next = (str(uuid.uuid4()),)

    def fetchone(self):
        return self._next

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeConn:
    def cursor(self):
        return _FakeCursor()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_ensure_corpus_creates_when_missing(monkeypatch):
    from rip_maf.api import corpus as corpus_api

    monkeypatch.setattr(corpus_api, "pg_connection", lambda: _FakeConn())
    corpus_id, created = corpus_api._get_or_create_corpus("owner-1", "chat-9", None)
    assert created is True
    assert corpus_id
