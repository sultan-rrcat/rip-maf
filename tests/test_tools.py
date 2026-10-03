"""Tools: registry/executor + the five tools.

Live where possible (plot = stdlib, doc = installed libs); rag.query uses an
injected fake (real VectorRAG needs the torch env), corpus.inspect fakes its
DB cursor.
"""

from __future__ import annotations

import pytest

from rip_maf.tools.base import Tool, ToolRequest, ToolResponse
from rip_maf.tools.executor import execute_tool
from rip_maf.tools.rag_query import RagQueryTool, bind_rag_singleton, rag_query
from rip_maf.tools.registry import ToolRegistry, get_default_tool_registry

EXPECTED_IDS = [
    "corpus.inspect",
    "doc.convert",
    "doc.generate",
    "plot.chart",
    "rag.query",
]


class FakeRAG:
    """Stand-in for VectorRAG (same retrieve_context shape)."""

    def __init__(self, results: list[dict] | None = None):
        self.seen: list[tuple] = []
        self.results = results if results is not None else [
            {"content": "c1", "source": "f.pdf", "section": "H1", "rerank_score": 0.9},
            {"content": "c2", "source": "f.pdf", "section": "H1", "rerank_score": 0.8},
            {"content": "c3", "source": "g.pdf", "section": "H2", "rerank_score": 0.7},
        ]

    def retrieve_context(
        self, corpus_id, query, top_k=4, file_id=None, file_name=None, mode="specific"
    ):
        self.seen.append((corpus_id, query, top_k, file_id, mode))
        return {"query": query, "mode": mode, "results": list(self.results)}


class WholeFileRAG(FakeRAG):
    """FakeRAG that also implements the whole-file shortcut."""

    #: Sentinel: "no override given" so an explicit `whole=None` (shortcut
    #: declines) stays distinguishable from the default whole-file hit.
    _DEFAULT = object()

    def __init__(self, whole=_DEFAULT):
        super().__init__()
        self._whole = (
            [
                {"content": "whole a", "source": "f.pdf", "section": "H1", "rerank_score": None},
                {"content": "whole b", "source": "f.pdf", "section": "H1", "rerank_score": None},
                {"content": "whole c", "source": "f.pdf", "section": "H2", "rerank_score": None},
            ]
            if whole is self._DEFAULT
            else whole
        )
        self.ranked_calls = 0
        self.whole_file_calls: list[tuple] = []

    def retrieve_whole_file(self, corpus_id, *, file_id=None, file_name=None):
        self.whole_file_calls.append((corpus_id, file_id, file_name))
        return self._whole

    def retrieve_context(self, *args, **kwargs):
        self.ranked_calls += 1
        return super().retrieve_context(*args, **kwargs)


class SpyProvider:
    """Counts planner invocations; returns one sub-query."""

    def __init__(self):
        self.calls = 0

    def generate_structured(self, model, messages, schema, temperature=0):
        self.calls += 1
        return {"queries": ["sub one"]}


# --- Registry / executor ---


class TestRegistry:
    def test_all_tools_registered(self):
        reg = get_default_tool_registry(rag=FakeRAG())
        assert sorted(t["tool_id"] for t in reg.manifest()) == EXPECTED_IDS
        assert len(reg) == 5

    def test_all_inherit_tool(self):
        reg = get_default_tool_registry(rag=FakeRAG())
        for tool_id in EXPECTED_IDS:
            assert isinstance(reg.get(tool_id), Tool)

    def test_unknown_tool_raises(self):
        with pytest.raises(KeyError):
            get_default_tool_registry().get("nope.tool")

    def test_no_approval_gate_in_executor(self):
        # Signature must not contain an approval parameter (merge decision).
        import inspect

        assert "approved" not in inspect.signature(execute_tool).parameters

    def test_sandboxed_tool_runs_without_approval(self):
        reg = get_default_tool_registry(rag=FakeRAG())
        resp = execute_tool(
            reg, "plot.chart",
            {"chart_type": "bar", "labels": ["a"], "values": [1]},
        )
        assert resp.ok

    def test_raising_tool_becomes_ok_false(self):
        class Boom(Tool):
            tool_id = "boom.tool"
            name = "Boom"
            description = "raises"
            effect_class = "read-only"  # type: ignore[assignment]

            def execute(self, request: ToolRequest) -> ToolResponse:
                raise RuntimeError("kaboom")

        reg = ToolRegistry()
        reg.register(Boom())
        resp = execute_tool(reg, "boom.tool", {})
        assert not resp.ok and "kaboom" in (resp.error or "")

    def test_identity_mismatch_rejected(self):
        class Liar(Tool):
            tool_id = "liar.tool"
            name = "Liar"
            description = "wrong id"
            effect_class = "read-only"  # type: ignore[assignment]

            def execute(self, request: ToolRequest) -> ToolResponse:
                return ToolResponse(tool_id="other.tool", ok=True)

        reg = ToolRegistry()
        reg.register(Liar())
        resp = execute_tool(reg, "liar.tool", {})
        assert not resp.ok and "mismatch" in (resp.error or "")


# --- rag.query ---


class TestRagQuery:
    def test_returns_chunks_with_sources(self):
        reg = get_default_tool_registry(rag=FakeRAG())
        resp = execute_tool(reg, "rag.query", {"corpus_id": "nb-1", "query": "hello"})
        assert resp.ok
        assert len(resp.data["results"]) == 3
        # extract_sources shape for the Q32 SSE event, deduped.
        assert sorted(resp.data["sources"], key=lambda s: s["source"]) == [
            {"source": "f.pdf", "section": "H1"},
            {"source": "g.pdf", "section": "H2"},
        ]
        assert "[1 (f.pdf)]" in (resp.output or "")

    def test_corpus_id_scoped_and_top_k_default(self):
        fake = FakeRAG()
        reg = get_default_tool_registry(rag=fake)
        execute_tool(reg, "rag.query", {"corpus_id": "nb-9", "query": "q"})
        assert fake.seen and fake.seen[0][0] == "nb-9" and fake.seen[0][2] == 4
        assert fake.seen[0][4] == "specific"

    def test_file_id_and_mode_forwarded(self):
        fake = FakeRAG()
        reg = get_default_tool_registry(rag=fake)
        resp = execute_tool(
            reg,
            "rag.query",
            {"corpus_id": "nb-1", "query": "q", "file_id": "fid-1", "mode": "overview"},
        )
        assert resp.ok
        assert fake.seen[0][3] == "fid-1" and fake.seen[0][4] == "overview"
        assert resp.data["file_id"] == "fid-1" and resp.data["mode"] == "overview"

    def test_bad_mode_rejected(self):
        reg = get_default_tool_registry(rag=FakeRAG())
        resp = execute_tool(
            reg, "rag.query", {"corpus_id": "nb-1", "query": "q", "mode": "bogus"}
        )
        assert not resp.ok and "mode" in (resp.error or "")

    def test_placeholder_file_id_rejected(self):
        reg = get_default_tool_registry(rag=FakeRAG())
        resp = execute_tool(
            reg, "rag.query", {"corpus_id": "nb-1", "query": "q", "file_id": "{{1}}"}
        )
        assert not resp.ok and "placeholder" in (resp.error or "")

    def test_missing_corpus_id_fails_honest(self):
        reg = get_default_tool_registry(rag=FakeRAG())
        resp = execute_tool(reg, "rag.query", {"query": "hello"})
        assert not resp.ok and "corpus_id" in (resp.error or "")

    def test_missing_query_fails_honest(self):
        reg = get_default_tool_registry(rag=FakeRAG())
        resp = execute_tool(reg, "rag.query", {"corpus_id": "nb-1"})
        assert not resp.ok and "query" in (resp.error or "")

    def test_unbound_singleton_fails_honest(self):
        from rip_maf.tools import rag_query as rq_mod

        old = rq_mod._rag_singleton
        rq_mod._rag_singleton = None
        try:
            tool = RagQueryTool()
            req = ToolRequest(
                tool_id="rag.query", input={"corpus_id": "nb", "query": "q"}
            )
            resp = tool.execute(req)
            assert not resp.ok
        finally:
            rq_mod._rag_singleton = old

    def test_module_function_uses_injected_rag(self):
        fake = FakeRAG()
        out = rag_query("nb-1", "hello", rag=fake)
        assert len(out) == 3 and out[0]["source"] == "f.pdf"

    def test_whole_file_shortcut_skips_ranked_retrieval(self):
        """Whole-file hit returns every chunk and never calls the planner."""
        rag = WholeFileRAG()
        provider = SpyProvider()
        tool = RagQueryTool(rag=rag, provider=provider)
        resp = tool.execute(
            ToolRequest(tool_id="rag.query", input={"corpus_id": "nb-1", "query": "hi"})
        )
        assert resp.ok
        assert len(resp.data["results"]) == 3
        assert resp.data["whole_file"] is True
        assert [r["content"] for r in resp.data["results"]] == [
            "whole a",
            "whole b",
            "whole c",
        ]
        # The whole point: no embed/vector/FTS/RRF/rerank, no planner LLM call.
        assert rag.ranked_calls == 0
        assert provider.calls == 0
        assert resp.data["generated_queries"] == ["hi"]
        # Sources still carry every chunk's section (SSE `sources` stays
        # intact). `extract_sources` dedupes via a set, so order is not
        # guaranteed — compare as a set.
        assert {tuple(sorted(s.items())) for s in resp.data["sources"]} == {
            (("section", "H1"), ("source", "f.pdf")),
            (("section", "H2"), ("source", "f.pdf")),
        }
        assert "whole a" in resp.output

    def test_whole_file_shortcut_scopes_to_file_id(self):
        rag = WholeFileRAG()
        tool = RagQueryTool(rag=rag)
        tool.execute(
            ToolRequest(
                tool_id="rag.query",
                input={"corpus_id": "nb-1", "query": "hi", "file_id": "f1"},
            )
        )
        assert rag.whole_file_calls == [("nb-1", "f1", None)]

    def test_shortcut_disabled_falls_back_to_ranked(self):
        """None (over budget) keeps today's path exactly."""
        rag = WholeFileRAG(whole=None)
        provider = SpyProvider()
        tool = RagQueryTool(rag=rag, provider=provider)
        resp = tool.execute(
            ToolRequest(tool_id="rag.query", input={"corpus_id": "nb-1", "query": "hi"})
        )
        assert resp.ok
        assert resp.data["whole_file"] is False
        assert rag.ranked_calls == 1
        assert provider.calls == 1

    def test_shortcut_error_does_not_fail_retrieval(self):
        """A raising shortcut must not fail the tool."""

        class BoomRAG(WholeFileRAG):
            def retrieve_whole_file(self, corpus_id, *, file_id=None, file_name=None):
                raise RuntimeError("db down")

        tool = RagQueryTool(rag=BoomRAG())
        resp = tool.execute(
            ToolRequest(tool_id="rag.query", input={"corpus_id": "nb-1", "query": "hi"})
        )
        assert resp.ok
        assert resp.data["whole_file"] is False

    def test_legacy_double_without_shortcut_still_runs(self):
        """Plain FakeRAG exposes no retrieve_whole_file (back-compat)."""
        rag = FakeRAG()
        tool = RagQueryTool(rag=rag)
        resp = tool.execute(
            ToolRequest(tool_id="rag.query", input={"corpus_id": "nb-1", "query": "hi"})
        )
        assert resp.ok
        assert resp.data["whole_file"] is False
        assert rag.seen == [("nb-1", "hi", 4, None, "specific")]

    def test_module_singleton_binding(self):
        fake = FakeRAG()
        bind_rag_singleton(fake)
        try:
            assert len(rag_query("nb-1", "hello")) == 3
        finally:
            bind_rag_singleton(None)  # type: ignore[arg-type]


# --- plot.chart ---


class TestToolContract:
    """The contract lives with the tool (ADR-035).

    `execute()` and any pre-flight caller must reach the same verdict, and a
    caller must never re-implement a rule — the drift that left ReAct's copy
    of plot.chart's shape four revisions behind the tool's.
    """

    def _registry(self):
        return get_default_tool_registry()

    def test_preflight_and_execute_agree(self):
        from rip_maf.orchestration.react import _validate_react_input

        registry = self._registry()
        cases = [
            ("plot.chart", {"chart_type": "line", "labels": ["a"],
                            "values": [[1, 2]]}),
            ("plot.chart", {"chart_type": "line", "labels": ["a"],
                            "values": [1], "series": [{"label": "s",
                                                        "values": [1]}]}),
            ("plot.chart", {"labels": ["a"], "values": [1]}),
            ("plot.chart", {"chart_type": "line", "labels": ["a"],
                            "values": [1], "china_values": [2]}),
            ("rag.query", {"query": "x", "mode": "bogus"}),
            ("rag.query", {"query": "x", "file_id": "{{1}}"}),
            ("doc.convert", {"target_format": "txt", "file_id": "f"}),
            ("doc.generate", {"title": "t", "sections": [{"heading": "h"}]}),
        ]
        for tool_id, payload in cases:
            # corpus_id is injected by the engine, so execute() sees it —
            # without it the injection guard (a different failure) fires first.
            injected = {**payload, "corpus_id": "nb-1"}
            hint = _validate_react_input(tool_id, payload, registry)
            resp = execute_tool(registry, tool_id, injected)
            assert not resp.ok, (tool_id, payload)
            assert hint is not None, (tool_id, payload)
            assert resp.error in hint, (tool_id, resp.error, hint)

    def test_every_tool_declares_an_example_and_closes_its_keys(self):
        for manifest_entry in self._registry().manifest():
            tool = self._registry().get(manifest_entry["tool_id"])
            assert manifest_entry["input_schema"].get(
                "additionalProperties"
            ) is False, manifest_entry["tool_id"]
            assert tool.input_example.strip(), manifest_entry["tool_id"]

    def test_engine_injected_keys_are_not_unknown(self):
        # run_plan_graph injects corpus_id AND expected_output_type into
        # every resolved step input; a closed key set must tolerate both.
        for tool_id, payload in (
            ("rag.query", {"query": "x"}),
            ("doc.convert", {"file_id": "f", "target_format": "md"}),
            ("plot.chart", {"chart_type": "bar", "labels": ["a"], "values": [1]}),
            ("corpus.inspect", {}),
        ):
            tool = self._registry().get(tool_id)
            injected = {**payload, "corpus_id": "nb", "expected_output_type": "x"}
            assert tool.validate_input(injected) is None, tool_id

    def test_a_placeholder_in_file_id_is_still_rejected(self):
        # Exempting engine-owned keys must not exempt invented ones.
        tool = self._registry().get("rag.query")
        assert tool.validate_input(
            {"query": "x", "corpus_id": "nb", "file_id": "{{1}}"}
        ) is not None


class TestPlotChart:
    def test_bar_renders_svg(self):
        reg = get_default_tool_registry()
        resp = execute_tool(
            reg, "plot.chart",
            {"chart_type": "bar", "labels": ["a", "b"], "values": [1, 2], "title": "T"},
        )
        assert resp.ok
        assert resp.output and resp.output.startswith("<svg")
        assert resp.data["point_count"] == 2

    def test_line_renders_svg(self):
        reg = get_default_tool_registry()
        resp = execute_tool(
            reg, "plot.chart",
            {"chart_type": "line", "labels": ["a", "b"], "values": [1, 2]},
        )
        assert resp.ok and "<polyline" in (resp.output or "")

    def test_bad_chart_type_rejected(self):
        reg = get_default_tool_registry()
        resp = execute_tool(
            reg, "plot.chart",
            {"chart_type": "pie", "labels": ["a"], "values": [1]},
        )
        assert not resp.ok

    def test_csv_string_values_split(self):
        # Placeholder-resolved whole-text output arrives as ONE string.
        reg = get_default_tool_registry()
        resp = execute_tool(
            reg, "plot.chart",
            {"chart_type": "bar", "labels": ["a", "b"], "values": ["0.82, 0.88"]},
        )
        assert resp.ok and resp.data["point_count"] == 2

    def test_stray_commas_tolerated(self):
        reg = get_default_tool_registry()
        resp = execute_tool(
            reg, "plot.chart",
            {"chart_type": "bar", "labels": ["a", "b"], "values": [",0.82,", "0.88,"]},
        )
        assert resp.ok and resp.data["point_count"] == 2

    def test_garbage_values_still_rejected(self):
        reg = get_default_tool_registry()
        resp = execute_tool(
            reg, "plot.chart",
            {"chart_type": "bar", "labels": ["a"], "values": ["not a number"]},
        )
        assert not resp.ok and "must all be numbers" in (resp.error or "")

    def test_length_checked_after_split(self):
        reg = get_default_tool_registry()
        resp = execute_tool(
            reg, "plot.chart",
            {"chart_type": "bar", "labels": ["a", "b"], "values": ["1, 2, 3"]},
        )
        assert not resp.ok and "same length" in (resp.error or "")

    def test_multiseries_line_renders_with_legend(self):
        reg = get_default_tool_registry()
        resp = execute_tool(
            reg, "plot.chart",
            {"chart_type": "line", "labels": ["2000", "2010", "2020"],
             "series": [{"label": "USA", "values": [10.0, 15.0, 21.0]},
                        {"label": "China", "values": ["1.2, 6.0, 14.7"]}],
             "title": "GDP (approximate)"},
        )
        assert resp.ok
        assert (resp.output or "").count("<polyline") == 2
        assert ">USA<" in (resp.output or "") and ">China<" in (resp.output or "")
        assert resp.data["series_count"] == 2 and resp.data["point_count"] == 3

    def test_multiseries_bar_groups(self):
        reg = get_default_tool_registry()
        resp = execute_tool(
            reg, "plot.chart",
            {"chart_type": "bar", "labels": ["a", "b"],
             "series": [{"label": "x", "values": [1, 2]},
                        {"label": "y", "values": [3, 4]}]},
        )
        assert resp.ok and (resp.output or "").startswith("<svg")

    def test_values_and_series_conflict_rejected(self):
        reg = get_default_tool_registry()
        resp = execute_tool(
            reg, "plot.chart",
            {"chart_type": "bar", "labels": ["a"], "values": [1],
             "series": [{"label": "x", "values": [1]}]},
        )
        assert not resp.ok and "never both" in (resp.error or "")

    def test_missing_values_and_series_rejected(self):
        reg = get_default_tool_registry()
        resp = execute_tool(
            reg, "plot.chart",
            {"chart_type": "bar", "labels": ["a"]},
        )
        assert not resp.ok and "non-empty array" in (resp.error or "")

    def test_series_length_mismatch_rejected(self):
        reg = get_default_tool_registry()
        resp = execute_tool(
            reg, "plot.chart",
            {"chart_type": "line", "labels": ["a", "b"],
             "series": [{"label": "x", "values": [1]}]},
        )
        assert not resp.ok and "but 2 labels" in (resp.error or "")

    def test_too_many_series_rejected(self):
        reg = get_default_tool_registry()
        resp = execute_tool(
            reg, "plot.chart",
            {"chart_type": "line", "labels": ["a"],
             "series": [{"label": f"s{i}", "values": [1]} for i in range(6)]},
        )
        assert not resp.ok and "at most 5 series" in (resp.error or "")

    def test_series_entry_without_label_rejected(self):
        reg = get_default_tool_registry()
        resp = execute_tool(
            reg, "plot.chart",
            {"chart_type": "line", "labels": ["a"],
             "series": [{"label": "", "values": [1]}]},
        )
        assert not resp.ok and "label" in (resp.error or "")

    def test_missing_title_defaults_from_labels(self):
        # Untitled calls still render a heading derived from the data
        # (trace affdbbd4: ReAct omitted title on 3 of 4 charts).
        reg = get_default_tool_registry()
        resp = execute_tool(
            reg, "plot.chart",
            {"chart_type": "bar", "labels": ["FASDD_CV", "AgniNetra"],
             "values": [0.5, 0.7]},
        )
        assert resp.ok
        assert "FASDD_CV vs AgniNetra" in (resp.output or "")

    def test_missing_title_defaults_from_series(self):
        reg = get_default_tool_registry()
        resp = execute_tool(
            reg, "plot.chart",
            {"chart_type": "bar", "labels": ["FASDD_CV", "AgniNetra"],
             "series": [{"label": "s", "values": [0.5, 0.7]},
                        {"label": "n", "values": [0.4, 0.6]}]},
        )
        assert resp.ok
        assert "s, n by FASDD_CV vs AgniNetra" in (resp.output or "")

    def test_explicit_title_kept(self):
        reg = get_default_tool_registry()
        resp = execute_tool(
            reg, "plot.chart",
            {"chart_type": "bar", "labels": ["a", "b"], "values": [1, 2],
             "title": "Custom"},
        )
        assert resp.ok and ">Custom<" in (resp.output or "")


# --- doc.generate (live libs) ---


class TestDocGenerate:
    def test_renders_all_formats(self):
        reg = get_default_tool_registry()
        resp = execute_tool(
            reg, "doc.generate",
            {"title": "T", "sections": [{"heading": "H", "body": "B"}]},
        )
        assert resp.ok
        assert (resp.output or "").startswith("# T")
        assert resp.data["docx_b64"] and resp.data["pdf_b64"]

    def test_missing_title_rejected(self):
        reg = get_default_tool_registry()
        resp = execute_tool(reg, "doc.generate", {"sections": []})
        assert not resp.ok

    def test_neither_path_rejected(self):
        reg = get_default_tool_registry()
        resp = execute_tool(reg, "doc.generate", {"title": "T"})
        assert not resp.ok and "sections" in (resp.error or "")

    def test_both_paths_rejected(self):
        reg = get_default_tool_registry()
        resp = execute_tool(
            reg, "doc.generate",
            {"title": "T", "sections": [{"heading": "H", "body": "B"}],
             "content": "raw"},
        )
        assert not resp.ok and "never both" in (resp.error or "")


class TestDocGenerateVerbatim:
    """Verbatim file path: content + format/filename, byte-for-byte."""

    def test_csv_by_filename(self):
        import base64

        reg = get_default_tool_registry()
        resp = execute_tool(
            reg, "doc.generate",
            {"content": "a,b\n1,2\n", "filename": "result.csv"},
        )
        assert resp.ok
        assert resp.output == "a,b\n1,2\n"
        assert resp.data["filename"] == "result.csv"
        assert resp.data["mime"] == "text/csv"
        assert base64.b64decode(resp.data["file_b64"]).decode() == "a,b\n1,2\n"

    def test_code_file_by_format(self):
        reg = get_default_tool_registry()
        resp = execute_tool(
            reg, "doc.generate",
            {"content": "print('hi')\n", "format": "py"},
        )
        assert resp.ok
        assert resp.data["filename"] == "document.py"
        assert resp.data["mime"] == "text/x-python"

    def test_default_is_txt(self):
        reg = get_default_tool_registry()
        resp = execute_tool(reg, "doc.generate", {"content": "hello"})
        assert resp.ok and resp.data["filename"] == "document.txt"

    def test_filename_format_mismatch_rejected(self):
        reg = get_default_tool_registry()
        resp = execute_tool(
            reg, "doc.generate",
            {"content": "x", "filename": "n.txt", "format": "csv"},
        )
        assert not resp.ok and "disagree" in (resp.error or "")

    def test_executable_extension_rejected(self):
        reg = get_default_tool_registry()
        resp = execute_tool(
            reg, "doc.generate", {"content": "x", "filename": "run.exe"}
        )
        assert not resp.ok and "extension" in (resp.error or "")

    def test_empty_content_rejected(self):
        reg = get_default_tool_registry()
        resp = execute_tool(reg, "doc.generate", {"content": ""})
        assert not resp.ok and "content" in (resp.error or "")

    def test_verbatim_collects_as_artifact(self, tmp_path):
        import base64

        from rip_maf.agents.base import StepStatus
        from rip_maf.artifacts import collect_artifacts
        from rip_maf.orchestration.results import StepResult

        result = StepResult(
            step_id="s1", agent_id="doc.generate", status=StepStatus.SUCCESS,
            output="a,b\n1,2\n",
            data={
                "file_b64": base64.b64encode(b"a,b\n1,2\n").decode(),
                "filename": "result.csv", "mime": "text/csv",
            },
        )
        found = collect_artifacts(
            [result], upload_dir=str(tmp_path), corpus_id="nb", run_id="r",
        )
        assert found and found[0]["filename"] == "result.csv"
        assert found[0]["mime"] == "text/csv"



