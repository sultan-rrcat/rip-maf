"""Unit tests for degraded Ollama init (no live server)."""

import os
import sys

import httpx
import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from rip_maf.core.config import settings
from rip_maf.providers.ollama import OllamaProvider


class _Resp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {"models": [{"name": "m1"}]}

    def json(self):
        return self._payload


def test_init_unreachable_never_raises(monkeypatch):
    def fake_get(self, *a, **k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    p = OllamaProvider()
    assert p._reachable is False
    assert p.served_model("anything") == settings.ollama_default_model
    with pytest.raises(ValueError, match="Could not connect"):
        p.ensure_ready()


def test_init_lists_models(monkeypatch):
    def fake_get(self, *a, **k):
        return _Resp()

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    p = OllamaProvider()
    assert p._reachable is True
    assert p.served_model("m1") == "m1"
    assert p.served_model("unknown") == settings.ollama_default_model
