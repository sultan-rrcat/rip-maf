"""Unit tests for DB host normalization + connect timeout (no live DB)."""

import logging
import os
import sys

import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import psycopg2

from rip_maf.core import db  # isort: skip



def test_localhost_normalized():
    from rip_maf.core.config import Settings

    assert Settings(db_host="localhost").db_host == "127.0.0.1"
    assert Settings(db_host="LOCALHOST").db_host == "127.0.0.1"


def test_non_localhost_preserved():
    from rip_maf.core.config import Settings

    assert Settings(db_host="postgres").db_host == "postgres"
    assert Settings(db_host="127.0.0.1").db_host == "127.0.0.1"
    assert Settings().db_host == "127.0.0.1"
    assert Settings().db_connect_timeout_s == 5


def test_db_params_timeout_and_masked_log(monkeypatch, caplog):
    captured = {}

    def fake_connect(**kwargs):
        captured.update(kwargs)
        raise psycopg2.OperationalError("down")

    monkeypatch.setattr(psycopg2, "connect", fake_connect)
    monkeypatch.setattr(db, "_pool", None)
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PASSWORD", "supersecret")

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(psycopg2.OperationalError),
        db.pg_connection(),
    ):
        pass

    assert captured["host"] == "127.0.0.1"
    assert captured["connect_timeout"] == 5
    log_text = caplog.text
    assert "supersecret" not in log_text
    assert "Password: ***" in log_text
