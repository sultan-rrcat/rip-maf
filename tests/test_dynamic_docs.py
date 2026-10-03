"""V2(B) — corpus.inspect + doc.convert + planner doc-awareness.

Fake-heavy (no DB, no Docling, no Ollama): pg_connection and the
loader/render boundaries are monkeypatched.
"""

from __future__ import annotations

from contextlib import contextmanager

from rip_maf.agents.base import StepStatus
from rip_maf.artifacts import collect_artifacts
from rip_maf.orchestration.results import StepResult
from rip_maf.tools.base import ToolRequest
from rip_maf.tools.corpus_inspect import CorpusInspectTool
from rip_maf.tools.doc_convert import DocConvertTool
from rip_maf.tools.registry import get_default_tool_registry


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, *a, **k):
        return None

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self, *a, **k):
        return _FakeCursor(self._rows)


class TestCorpusInspect:
    def test_lists_files(self):
        import rip_maf.core.db as db_mod

        rows = [
            ("fid-1", "a.pdf", 10, "ready"),
            ("fid-2", "b.pdf", 20, "processing"),
        ]
        real = db_mod.pg_connection

        @contextmanager
        def fake():
            yield _FakeConn(rows)

        db_mod.pg_connection = fake
        try:
            resp = CorpusInspectTool().execute(
                ToolRequest(tool_id="corpus.inspect", input={"corpus_id": "nb-1"})
            )
        finally:
            db_mod.pg_connection = real
        assert resp.ok
        assert len(resp.data["files"]) == 2
        assert resp.data["files"][0]["file_id"] == "fid-1"
        assert "a.pdf" in (resp.output or "")

    def test_missing_corpus_id(self):
        resp = CorpusInspectTool().execute(
            ToolRequest(tool_id="corpus.inspect", input={})
        )
        assert not resp.ok and "corpus_id" in (resp.error or "")

    def test_registered(self):
        reg = get_default_tool_registry(rag=object())
        assert "corpus.inspect" in reg
        assert "doc.convert" in reg
        assert len(reg) == 5


class TestDocConvert:
    def _tool(self):
        return DocConvertTool()

    def test_missing_file_id(self):
        resp = self._tool().execute(
            ToolRequest(
                tool_id="doc.convert",
                input={"corpus_id": "nb", "target_format": "md"},
            )
        )
        assert not resp.ok and "file_id" in (resp.error or "")

    def test_bad_target_format(self):
        resp = self._tool().execute(
            ToolRequest(
                tool_id="doc.convert",
                input={"corpus_id": "nb", "file_id": "f", "target_format": "txt"},
            )
        )
        assert not resp.ok and "target_format" in (resp.error or "")

    def test_unknown_file_id(self, monkeypatch):
        import rip_maf.tools.doc_convert as dc_mod

        monkeypatch.setattr(
            dc_mod, "_resolve_source", lambda nb, fid: (_ for _ in ()).throw(ValueError("unknown file_id: x"))
        )
        resp = self._tool().execute(
            ToolRequest(
                tool_id="doc.convert",
                input={"corpus_id": "nb", "file_id": "x", "target_format": "md"},
            )
        )
        assert not resp.ok and "unknown file_id" in (resp.error or "")

    def test_md_lossless(self, monkeypatch):
        import rip_maf.tools.doc_convert as dc_mod

        monkeypatch.setattr(
            dc_mod, "_resolve_source", lambda nb, fid: ("/tmp/a.pdf", "a.pdf", ".pdf")
        )
        monkeypatch.setattr(dc_mod, "_load_markdown", lambda path, ext: "# T\n\nbody text")
        resp = self._tool().execute(
            ToolRequest(
                tool_id="doc.convert",
                input={"corpus_id": "nb", "file_id": "fid-1", "target_format": "md"},
            )
        )
        assert resp.ok
        assert resp.output == "# T\n\nbody text"
        assert resp.data["source_file_id"] == "fid-1"
        assert resp.data["target_format"] == "md"

    def test_docx_target(self, monkeypatch):
        import rip_maf.tools.doc_convert as dc_mod

        monkeypatch.setattr(
            dc_mod, "_resolve_source", lambda nb, fid: ("/tmp/a.pdf", "a.pdf", ".pdf")
        )
        monkeypatch.setattr(dc_mod, "_load_markdown", lambda path, ext: "hello")
        resp = self._tool().execute(
            ToolRequest(
                tool_id="doc.convert",
                input={"corpus_id": "nb", "file_id": "fid-1", "target_format": "docx"},
            )
        )
        assert resp.ok and resp.data.get("docx_b64")

    def test_no_rag_no_llm(self, monkeypatch):
        """Convert path touches only resolve+load+render — patch all three."""
        import rip_maf.tools.doc_convert as dc_mod

        seen = {}

        def fake_resolve(nb, fid):
            seen["resolve"] = (nb, fid)
            return ("/tmp/a.pdf", "a.pdf", ".pdf")

        def fake_load(path, ext):
            seen["load"] = path
            return "full text here"

        monkeypatch.setattr(dc_mod, "_resolve_source", fake_resolve)
        monkeypatch.setattr(dc_mod, "_load_markdown", fake_load)
        resp = self._tool().execute(
            ToolRequest(
                tool_id="doc.convert",
                input={"corpus_id": "nb", "file_id": "fid-9", "target_format": "md"},
            )
        )
        assert resp.ok and seen == {"resolve": ("nb", "fid-9"), "load": "/tmp/a.pdf"}

    def test_file_name_alias(self, monkeypatch):
        import rip_maf.tools.doc_convert as dc_mod

        monkeypatch.setattr(dc_mod, "_resolve_by_name", lambda nb, name: ("fid-1", "/tmp/a.pdf", "a.pdf", ".pdf"))
        monkeypatch.setattr(dc_mod, "_load_markdown", lambda path, ext: "txt")
        resp = self._tool().execute(
            ToolRequest(
                tool_id="doc.convert",
                input={"corpus_id": "nb", "file_name": "a.pdf", "target_format": "md"},
            )
        )
        assert resp.ok and resp.data["source_file_name"] == "a.pdf"

    def test_convert_all_star(self, monkeypatch):
        import os

        import rip_maf.tools.doc_convert as dc_mod

        def fake_list_ready(nb):
            return [("fid-1", "a.pdf", ".pdf"), ("fid-2", "b.pdf", ".pdf")]

        monkeypatch.setattr(dc_mod, "_list_ready", fake_list_ready)
        monkeypatch.setattr(dc_mod, "_disk_path", lambda nb, fid, ext: f"/tmp/{fid}{ext}")
        monkeypatch.setattr(os.path, "isfile", lambda p: True)
        def fake_convert(nb, fid, path, name, ext, target):
            return ("content", {"source_file_id": fid, "source_file_name": name, "target_format": target, "markdown": "x"})
        monkeypatch.setattr(dc_mod, "_convert_one", fake_convert)
        resp = self._tool().execute(
            ToolRequest(
                tool_id="doc.convert",
                input={"corpus_id": "nb", "file_id": "*", "target_format": "md"},
            )
        )
        assert resp.ok
        conv = resp.data.get("conversions", [])
        assert len(conv) == 2
        assert conv[0]["source_file_id"] == "fid-1"
        assert conv[1]["source_file_id"] == "fid-2"


class TestConvertArtifacts:
    def test_md_with_source_id_becomes_file(self, tmp_path):
        step = StepResult(
            step_id="2",
            agent_id="doc.convert",
            status=StepStatus.SUCCESS,
            output="# T",
            data={"markdown": "# T", "source_file_id": "fid-1"},
        )
        found = collect_artifacts(
            [step], upload_dir=str(tmp_path), corpus_id="nb", run_id="run-1"
        )
        assert any(a["filename"].endswith(".md") for a in found)

    def test_bare_markdown_without_source_stays_text(self, tmp_path):
        step = StepResult(
            step_id="2",
            agent_id="reasoning",
            status=StepStatus.SUCCESS,
            output="# T",
            data={"markdown": "# T"},
        )
        found = collect_artifacts(
            [step], upload_dir=str(tmp_path), corpus_id="nb", run_id="run-2"
        )
        assert found == []

    def test_duplicate_chart_svg_collected_once(self, tmp_path):
        # Trace 07fb4f59 r2/r3: identical SVG bytes collected twice, so
        # the frontend Artifacts panel showed the same plot twice.
        svg = "<svg xmlns='x'><title>same chart</title></svg>"
        other = "<svg xmlns='x'><title>other chart</title></svg>"
        steps = [
            StepResult(
                step_id="r2", agent_id="plot.chart", status=StepStatus.SUCCESS,
                output=svg, data={"svg": svg},
            ),
            StepResult(
                step_id="r3", agent_id="plot.chart", status=StepStatus.SUCCESS,
                output=svg, data={"svg": svg},
            ),
            StepResult(
                step_id="r5", agent_id="plot.chart", status=StepStatus.SUCCESS,
                output=other, data={"svg": other},
            ),
        ]
        found = collect_artifacts(
            steps, upload_dir=str(tmp_path), corpus_id="nb", run_id="run-3"
        )
        charts = [a for a in found if a["kind"] == "chart"]
        assert len(charts) == 2
        assert {a["step_id"] for a in charts} == {"r2", "r5"}


class _FakeProvider:
    def __init__(self, payload):
        self._payload = payload
        self.seen_messages = None

    def generate_structured(self, model, messages, schema, temperature=0):
        self.seen_messages = messages
        return dict(self._payload)


class TestRouterDocAwareness:
    def test_router_prompt_covers_convert_slots(self):
        from rip_maf.orchestration.router import Router

        provider = _FakeProvider(
            {"intent": "convert_one", "queries": [], "confidence": 0.9,
             "file_hint": "a.pdf", "target_format": "md"}
        )
        result = Router(provider).route("convert a.pdf to md please")
        assert result.file_hint == "a.pdf" and result.target_format == "md"
        system = provider.seen_messages[0]["content"]
        assert "file_hint" in system and "target_format" in system

    def test_react_prompt_renders_snapshot(self):
        from rip_maf.agents.registry import get_default_agent_registry
        from rip_maf.orchestration.react import run_react
        from rip_maf.providers.base import ModelProvider
        from rip_maf.tools.registry import get_default_tool_registry

        seen: list = []

        class _Probe(ModelProvider):
            def generate(self, model, messages, *, temperature=0.2, max_tokens=None):
                return "ok"

            def generate_structured(self, model, messages, schema, *, temperature=0.0):
                seen.append(messages)
                return {"thought": "done", "executor": "reasoning",
                        "input": {}, "is_final": True, "answer": "ok"}

            def embed(self, model: str, text: str) -> list[float]:
                raise NotImplementedError("test fake")

            def list_available_models(self) -> list[dict]:
                return [{"id": "fake"}]

        probe = _Probe()
        agents = get_default_agent_registry(probe)
        tools = get_default_tool_registry()
        run_react(
            "what do docs say?", probe, agents, tools,
            trace_id="t", corpus_id="nb-1",
            corpus_context="1 file(s): a.pdf [ready] id=fid-1",
        )
        system = seen[0][0]["content"]
        assert "a.pdf" in system

    def test_no_docs_react_snapshot(self):
        from rip_maf.agents.registry import get_default_agent_registry
        from rip_maf.orchestration.react import run_react
        from rip_maf.providers.base import ModelProvider
        from rip_maf.tools.registry import get_default_tool_registry

        seen: list = []

        class _Probe(ModelProvider):
            def generate(self, model, messages, *, temperature=0.2, max_tokens=None):
                return "ok"

            def generate_structured(self, model, messages, schema, *, temperature=0.0):
                seen.append(messages)
                return {"thought": "done", "executor": "reasoning",
                        "input": {}, "is_final": True, "answer": "ok"}

            def embed(self, model: str, text: str) -> list[float]:
                raise NotImplementedError("test fake")

            def list_available_models(self) -> list[dict]:
                return [{"id": "fake"}]

        probe = _Probe()
        agents = get_default_agent_registry(probe)
        tools = get_default_tool_registry()
        run_react(
            "what do docs say?", probe, agents, tools,
            trace_id="t", corpus_id="nb-1",
        )
        assert "(no documents)" in seen[0][0]["content"]
