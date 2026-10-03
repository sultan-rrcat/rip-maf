"""Observability tests: disabled-mode no-ops + tracing wrapper delegation.

No Langfuse server needed — everything here runs with
langfuse_enabled=False (the default), proving the Athena-ported helpers
cost nothing and change nothing when tracing is off.
"""
import pytest

from rip_maf.observability import langfuse as lf
from rip_maf.providers.base import ModelProvider
from rip_maf.providers.tracing import TracingProvider, wrap_provider


@pytest.fixture(autouse=True)
def _force_tracing_disabled(monkeypatch):
    """These tests assert disabled-mode behavior — pin it.

    The repo .env ships LANGFUSE_ENABLED=true and earlier tests run the
    app lifespan (init_langfuse), so without this the assertions depend
    on ambient env and test order instead of the code under test.
    """
    from rip_maf.core.config import settings

    monkeypatch.setattr(settings, "langfuse_enabled", False)
    monkeypatch.setattr(lf, "_client", None)


class _FakeProvider(ModelProvider):
    def __init__(self):
        self.calls = []
        self._usage = {"input": 3, "output": 5, "total": 8}

    @property
    def last_usage(self):
        return self._usage

    def generate(self, model, messages, *, temperature=0.2, max_tokens=None):
        self.calls.append(("generate", model))
        return "hello"

    def generate_stream(self, model, messages, *, temperature=0.2, max_tokens=None):
        self.calls.append(("generate_stream", model))
        yield "hel"
        yield "lo"

    def generate_structured(self, model, messages, schema, *, temperature=0.0):
        self.calls.append(("generate_structured", model))
        return {"goal": "g", "steps": []}

    def embed(self, model, text):
        raise NotImplementedError("text-generation only")

    def list_available_models(self):
        return [{"id": "fake", "display_name": "fake"}]

    def served_model(self, requested_model):
        return requested_model


def test_tracing_disabled_by_default():
    assert lf.tracing_enabled() is False


def test_observe_is_identity_when_disabled():
    @lf.observe(name="x")
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_manual_span_disabled_yields_stub():
    with lf.manual_span("s", input="in") as obs:
        assert obs.id is None
        obs.update(output="out")  # no-op, must not raise


def test_manual_generation_disabled_yields_stub():
    with lf.manual_generation("g", input="in") as gen:
        assert gen.id is None
        gen.update(output="out")


def test_request_attributes_disabled_noop():
    with lf.request_attributes(session_id="n", user_id="u"):
        pass


def test_update_generation_disabled_noop():
    lf.update_generation(output="x")


def test_flush_disabled_noop():
    lf.flush()


def test_truncate_bounds():
    assert lf.truncate(None) is None
    assert lf.truncate("abcdef", 3) == "abc"
    assert len(lf.truncate({"k": "v" * 5000})) <= 2000


def test_wrap_provider_passthrough_when_disabled():
    inner = _FakeProvider()
    assert wrap_provider(inner) is inner


def test_wrap_provider_idempotent():
    inner = _FakeProvider()
    wrapped = TracingProvider(inner)
    assert wrap_provider(wrapped) is wrapped


def test_tracing_provider_delegates_generate():
    inner = _FakeProvider()
    wrapped = TracingProvider(inner)
    assert wrapped.generate("m", [{"role": "user", "content": "hi"}]) == "hello"
    assert ("generate", "m") in inner.calls


def test_tracing_provider_delegates_stream():
    inner = _FakeProvider()
    wrapped = TracingProvider(inner)
    assert "".join(wrapped.generate_stream("m", [])) == "hello"


def test_tracing_provider_delegates_structured():
    inner = _FakeProvider()
    wrapped = TracingProvider(inner)
    out = wrapped.generate_structured("m", [], {}, temperature=0.0)
    assert out == {"goal": "g", "steps": []}


def test_tracing_provider_passthrough_models():
    inner = _FakeProvider()
    wrapped = TracingProvider(inner)
    assert wrapped.served_model("m") == "m"
    assert wrapped.list_available_models() == [{"id": "fake", "display_name": "fake"}]
