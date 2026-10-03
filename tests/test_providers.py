"""Phase 2.1 — providers: ThinkFilter unit tests + Ollama live round-trip.

Live tests use the small local model (default ``openbmb/minicpm5-2b:latest``,
override with ``RIP_TEST_MODEL``) and skip when Ollama is down. Run with
``pytest backend/tests/test_providers.py -q --noconftest`` until the torch
env is repaired (Phase 6) — the shared conftest imports app.main, which
needs sentence_transformers.
"""

import os

import httpx
import pytest

from rip_maf.core.config import settings
from rip_maf.providers.base import ModelProvider
from rip_maf.providers.ollama import OllamaProvider
from rip_maf.providers.streaming import ThinkFilter, strip_think

TEST_MODEL = os.environ.get("RIP_TEST_MODEL", "openbmb/minicpm5-2b:latest")


def _present_model() -> str | None:
    """A model actually pulled on this Ollama host (env override wins).

    Hardcoding one model name breaks on hosts with different pulls —
    resolution falls back to the default by design, so tests must use
    whatever the server lists.
    """
    try:
        ids = [m["id"] for m in OllamaProvider().list_available_models()]
    except Exception:  # noqa: BLE001 - probe; needs_ollama skips anyway
        return None
    if TEST_MODEL in ids:
        return TEST_MODEL
    return ids[0] if ids else None


def _ollama_up() -> bool:
    try:
        r = httpx.get(
            f"{settings.ollama_base_url}/api/tags",
            timeout=5.0,
            trust_env=False,
        )
        return r.status_code == 200
    except httpx.RequestError:
        return False


needs_ollama = pytest.mark.skipif(
    not _ollama_up(), reason="Ollama down — live provider tests skipped"
)


# --- ThinkFilter / strip_think (pure, no Ollama) ---


class TestStripThink:
    def test_closed_block_removed(self):
        assert strip_think("<think>hmm</think>hello") == "hello"

    def test_unclosed_block_drops_tail(self):
        assert strip_think("hello<think>hmm") == "hello"

    def test_plain_text_untouched(self):
        assert strip_think("just text") == "just text"


class TestThinkFilter:
    def test_closed_block_split_across_chunks(self):
        f = ThinkFilter()
        out = f.feed("a<thi") + f.feed("nk>hid") + f.feed("den</th") + f.feed("ink>b")
        assert out + f.close() == "ab"

    def test_unclosed_block_dropped_at_close(self):
        f = ThinkFilter()
        out = f.feed("visible<think>reasoning forever")
        assert out + f.close() == "visible"

    def test_plain_text_flows(self):
        f = ThinkFilter()
        assert f.feed("hello ") + f.feed("world") + f.close() == "hello world"

    def test_feed_may_return_empty(self):
        # feed() may return "" — callers must skip empties (stream contract).
        f = ThinkFilter()
        out = "".join(
            f.feed(chunk) for chunk in ["<", "think>", "x", "</think>", "y"]
        )
        assert out + f.close() == "y"


# --- Live Ollama round-trip (minicpm5) ---


@needs_ollama
class TestOllamaProvider:
    def test_is_model_provider(self):
        assert isinstance(OllamaProvider(), ModelProvider)

    def test_model_resolution(self):
        p = OllamaProvider()
        present = _present_model()
        assert present is not None, "Ollama lists no pulled models"
        assert p.served_model(present) == present
        assert p.served_model("definitely-not-pulled:latest") == (
            settings.ollama_default_model
        )

    def test_list_available_models(self):
        models = OllamaProvider().list_available_models()
        assert models
        assert any(m["id"] == _present_model() for m in models)

    def test_generate_round_trip(self):
        p = OllamaProvider()
        text = p.generate(
            TEST_MODEL,
            [{"role": "user", "content": "Reply with exactly: hello rip"}],
            max_tokens=256,
        )
        assert "hello rip" in text
        assert "<think>" not in text

    def test_generate_stream_round_trip(self):
        p = OllamaProvider()
        chunks = list(
            p.generate_stream(
                TEST_MODEL,
                [{"role": "user", "content": "Reply with exactly: hello rip"}],
                max_tokens=256,
            )
        )
        assert chunks and all(chunks)  # non-empty-chunk contract
        assert "hello rip" in "".join(chunks)
