"""Plan Validator — deterministic safety gate between the Planner and execution.

The Planner is an LLM; it can produce valid-looking JSON that is still wrong
(nonexistent agent/tool, dependency cycle, oversized plan). The Validator is
the trust boundary: PURE rules, NO LLM. It raises a typed ValidationError on
any failure so the caller can fall back safely (fail honest).

Side-effecting tool steps PASS validation by design: RIP is local
single-user, so tools execute directly with no approval gate.
"""
from __future__ import annotations

import logging
import re

from rip_maf.agents.registry import AgentRegistry
from rip_maf.core.config import settings
from rip_maf.orchestration.plan import Plan
from rip_maf.tools.registry import ToolRegistry

logger = logging.getLogger("orchestration.validator")

#: Placeholder references inside step inputs, e.g. "{{2}}". Same shape as
#: plan_graph._PLACEHOLDER (kept local: the validator must not import the
#: execution graph).
_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_-]+)\s*\}\}")

#: Prose signal of an ungrounded fan-in (trace ecd93eb4): the message talks
#: about retrieved chunks / numbered steps without a {{id}} placeholder.
#: Checked only when rag.query chunks siblings exist (see
#: _check_prose_grounding), so ordinary "step N" prose stays legal.
_PROSE_STEP_REF = re.compile(
    r"(retrieved\s+chunks?|chunks?\s+from\s+steps?|steps?\s+\d+|from\s+step\s+\d+)",
    re.IGNORECASE,
)

#: Known expected_output_type vocabulary. "summary" is a legacy/planner
#: variant of "answer" (both are terminal prose). "observation" is the
#: ReAct internal type for a non-terminal agent step (scratchpad, never
#: the answer). Unknown types are allowed (forward-compatible — the
#: aggregator treats them as SHOW) but logged.
_KNOWN_OUTPUT_TYPES = frozenset({
    "chunks", "answer", "numbers", "chart", "document", "text",
    "clarification", "summary", "observation",
})

#: Terminal prose types that can ground a doc.generate report.
_ANSWER_TYPES = frozenset({"answer", "summary", "text"})


def _contains_placeholder(value: object) -> bool:
    """Whether any string in `value` carries a {{id}} placeholder."""
    if isinstance(value, str):
        return bool(_PLACEHOLDER.search(value))
    if isinstance(value, dict):
        return any(_contains_placeholder(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_placeholder(v) for v in value)
    return False


class PlanValidationError(ValueError):
    """Raised when a plan fails validation. Message is the reason."""


class PlanValidator:
    def __init__(
        self,
        registry: AgentRegistry,
        tool_registry: ToolRegistry | None = None,
        *,
        max_steps: int | None = None,
    ):
        self._registry = registry
        self._tool_registry = tool_registry or ToolRegistry()
        self._max_steps = max_steps or settings.default_max_plan_steps

    def validate(self, plan: Plan) -> Plan:

        self._check_unique_step_ids(plan) # -> PlanValidationError on duplicate step_id
        self._check_executors_exist(plan) # -> PlanValidationError on unknown agent/tool
        self._check_no_cycles(plan)       # -> PlanValidationError on a cycle
        self._check_budget(plan)          # -> PlanValidationError if too many steps
        self._check_dataflow_grounding(plan)  # -> PlanValidationError on ungrounded plot/report
        self._check_reasoning_grounding(plan)  # -> PlanValidationError on ungrounded agent step
        self._check_prose_grounding(plan)  # -> PlanValidationError on prose step-ref without placeholder
        self._check_placeholder_edges(plan)  # -> PlanValidationError on dangling {{id}} refs
        self._check_parallel_fanout(plan)  # -> PlanValidationError on >5 parallel long writes
        self._check_tool_required_fields(plan)  # -> PlanValidationError on missing required tool input
        self._check_retrieval_output_types(plan)  # -> PlanValidationError on rag.query not typed chunks
        logger.info("plan %s validated (%d steps)", plan.plan_id, len(plan.steps))
        return plan

    def _check_unique_step_ids(self, plan: Plan) -> None:
        ids = [s.step_id for s in plan.steps]
        if len(ids) != len(set(ids)):
            raise PlanValidationError("plan contains duplicate step_ids")

    def _check_executors_exist(self, plan: Plan) -> None:
        manifest = self._registry.manifest()
        known_agent = {agent["agent_id"] for agent in manifest}
        known_tool = {tool["tool_id"] for tool in self._tool_registry.manifest()}

        for step in plan.steps:
            has_agent = bool(step.agent_id)
            has_tool = bool(step.tool_id)
            if has_agent == has_tool:  # both or neither
                hint = ""
                if isinstance(step.input, dict):
                    if "agent_id" in step.input or "tool_id" in step.input:
                        hint = " (agent_id/tool_id are TOP-LEVEL step fields, not input keys — move them up)"
                    elif "step_id" in step.input or "depends_on" in step.input:
                        hint = (
                            f" (input buries step {str(step.input.get('step_id', '?'))!r} — "
                            "each steps[] element is ONE step; move step_id/depends_on/"
                            "expected_output_type out of input to top level)"
                        )
                    else:
                        hint = f" (input keys={sorted(step.input)})"
                raise PlanValidationError(
                    f"step {step.step_id} must set exactly one of agent_id/tool_id{hint}"
                )
            if has_agent and step.agent_id not in known_agent:
                raise PlanValidationError(f"step {step.step_id} references unknown agent")
            if has_tool and step.tool_id not in known_tool:
                raise PlanValidationError(f"step {step.step_id} references unknown tool")

    def _check_no_cycles(self, plan: Plan) -> None:
        step_ids = set()
        for step in plan.steps:
            step_ids.add(step.step_id)

        adjacency = {}
        for step in plan.steps:
            adjacency[step.step_id] = set(step.depends_on)

        for step in plan.steps:
            for dependency in step.depends_on:
                if dependency not in step_ids:
                    raise PlanValidationError(
                        f"step {step.step_id} depends on non existent step {dependency}"
                    )

        visiting = set()
        visited = set()

        def dfs(step_id: str) -> None:
            if step_id in visiting:
                raise PlanValidationError("cycle detected")
            if step_id in visited:
                return
            visiting.add(step_id)
            for dependency in adjacency[step_id]:
                dfs(dependency)
            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in step_ids:
            dfs(step_id)

    def _check_budget(self, plan: Plan) -> None:
        if len(plan.steps) > self._max_steps:
            raise PlanValidationError(f"plan has {len(plan.steps)}")

    def _check_dataflow_grounding(self, plan: Plan) -> None:
        """Reject ungrounded plot/report steps (fail-honest, no fake charts).

        - plot.chart with dependencies must reference upstream via a
          {{{id}}} placeholder in `values` (literals + deps = hallucinated
          chart). Standalone plots with literal numbers stay legal.
        - plot.chart `values` placeholders must resolve to a numbers-type
          step: prose/chunks cannot parse as floats at runtime. Direct
          dependence on rag.query chunks is rejected for the same reason.
        - doc.generate with dependencies must have an upstream
          answer/summary/text step; standalone reports with full sections
          stay legal.
        """
        by_id = {s.step_id: s for s in plan.steps}
        for step in plan.steps:
            eot = (step.expected_output_type or "text").lower()
            if eot not in _KNOWN_OUTPUT_TYPES:
                logger.warning(
                    "plan %s step %s has unknown expected_output_type %r",
                    plan.plan_id, step.step_id, step.expected_output_type,
                )

        def upstream(step_id: str) -> set[str]:
            seen: set[str] = set()
            stack = list(by_id[step_id].depends_on)
            while stack:
                dep = stack.pop()
                if dep in seen or dep not in by_id:
                    continue
                seen.add(dep)
                stack.extend(by_id[dep].depends_on)
            return seen

        def placeholders_in(value: object) -> set[str]:
            found: set[str] = set()
            if isinstance(value, str):
                found.update(_PLACEHOLDER.findall(value))
            elif isinstance(value, dict):
                for v in value.values():
                    found.update(placeholders_in(v))
            elif isinstance(value, list):
                for v in value:
                    found.update(placeholders_in(v))
            return found

        for step in plan.steps:
            if step.tool_id == "plot.chart":
                step_input = step.input if isinstance(step.input, dict) else {}
                values = step_input.get("values")
                series = step_input.get("series")
                if values is not None and series is not None:
                    raise PlanValidationError(
                        f"step {step.step_id} (plot.chart) passes both 'values' "
                        "and 'series' — pass exactly one (single series vs "
                        "multi-series comparison)"
                    )
                if values is None and series is None:
                    raise PlanValidationError(
                        f"step {step.step_id} (plot.chart) is missing required "
                        "input field 'values' (or 'series' for multi-series) — "
                        "refusing to execute an unguarded tool call"
                    )
                if series is not None and (
                    not isinstance(series, list) or not series
                ):
                    raise PlanValidationError(
                        f"step {step.step_id} (plot.chart) 'series' must be a "
                        "non-empty array of {label, values} objects"
                    )
                series_values: list[object] = []
                if isinstance(series, list):
                    for entry in series:
                        if not isinstance(entry, dict):
                            raise PlanValidationError(
                                f"step {step.step_id} (plot.chart) 'series' "
                                "entries must be {label, values} objects"
                            )
                        if not str(entry.get("label", "")).strip():
                            raise PlanValidationError(
                                f"step {step.step_id} (plot.chart) 'series' "
                                "entries need a non-empty string 'label'"
                            )
                        entry_values = entry.get("values")
                        if not isinstance(entry_values, list) or not entry_values:
                            raise PlanValidationError(
                                f"step {step.step_id} (plot.chart) series "
                                f"{entry.get('label')!r} needs a non-empty "
                                "'values' array"
                            )
                        series_values.append(entry_values)
                value_lists = (
                    [values] if isinstance(values, list) else []
                ) + series_values
                refs: set[str] = set()
                for value_list in value_lists:
                    refs.update(placeholders_in(value_list))
                if step.depends_on:
                    if not refs:
                        raise PlanValidationError(
                            f"step {step.step_id} (plot.chart) depends on "
                            f"{sorted(step.depends_on)} but its values carry no "
                            "{{{id}}} placeholder — dependent plots must reference "
                            "upstream numbers, never hardcoded literals"
                        )
                    for value_list in value_lists:
                        assert isinstance(value_list, list)
                        for v in value_list:
                            if isinstance(v, str) and _PLACEHOLDER.search(v) and not _PLACEHOLDER.fullmatch(v.strip()):
                                raise PlanValidationError(
                                    f"step {step.step_id} (plot.chart) values element "
                                    f"{v!r} mixes a placeholder with surrounding text — "
                                    "each values element must be a number or a lone "
                                    "{{{id}}} placeholder"
                                )
                    if len(refs) > len(value_lists):
                        raise PlanValidationError(
                            f"step {step.step_id} (plot.chart) values reference "
                            f"multiple upstream steps {sorted(refs)} in one series — "
                            "fan them into ONE merging numbers step first, then "
                            "reference only it (one placeholder per series)"
                        )
                    for ref in refs:
                        target = by_id.get(ref)
                        if target is None:
                            continue  # unknown dep: _check_no_cycles already rejects
                        target_eot = (target.expected_output_type or "text").lower()
                        if target_eot != "numbers":
                            raise PlanValidationError(
                                f"step {step.step_id} (plot.chart) values reference "
                                f"step {ref} ({target_eot or 'text'}), but plot values "
                                "must reference a numbers-producing step"
                            )
                    direct = [by_id[d] for d in step.depends_on if d in by_id]
                    if any(
                        d.tool_id == "rag.query"
                        and (d.expected_output_type or "").lower() == "chunks"
                        for d in direct
                    ):
                        raise PlanValidationError(
                            f"step {step.step_id} (plot.chart) depends directly on "
                            "rag.query chunks — route through a numbers-producing "
                            "reasoning step instead"
                        )
            elif step.tool_id == "doc.generate":
                if step.depends_on:
                    ups = upstream(step.step_id)
                    if not any(
                        (by_id[u].expected_output_type or "text").lower() in _ANSWER_TYPES
                        for u in ups
                    ):
                        raise PlanValidationError(
                            f"step {step.step_id} (doc.generate) depends on "
                            f"{sorted(step.depends_on)} with no upstream "
                            "answer/summary/text step — reports must be grounded "
                            "in an answer step"
                        )

    def _check_reasoning_grounding(self, plan: Plan) -> None:
        """Reject agent steps that declare dependencies but use no placeholder.

        An agent step with non-empty depends_on must reference at least one
        transitive upstream step via a {{id}} placeholder in its input values
        (typically the `message`). Without it the step executes on prose
        alone — e.g. three "write 5 MCQs" steps off rag.query that never see
        the retrieved chunks and ask the user to re-upload.
        """
        by_id = {s.step_id: s for s in plan.steps}

        def upstream(step_id: str) -> set[str]:
            seen: set[str] = set()
            stack = list(by_id[step_id].depends_on)
            while stack:
                dep = stack.pop()
                if dep in seen or dep not in by_id:
                    continue
                seen.add(dep)
                stack.extend(by_id[dep].depends_on)
            return seen

        def placeholders_in(value: object) -> set[str]:
            found: set[str] = set()
            if isinstance(value, str):
                found.update(_PLACEHOLDER.findall(value))
            elif isinstance(value, dict):
                for v in value.values():
                    found.update(placeholders_in(v))
            elif isinstance(value, list):
                for v in value:
                    found.update(placeholders_in(v))
            return found

        for step in plan.steps:
            if step.tool_id or not step.depends_on:
                continue
            refs = placeholders_in(step.input)
            ups = upstream(step.step_id)
            if not (refs & ups):
                raise PlanValidationError(
                    f"step {step.step_id} ({step.agent_id or 'agent'}) depends on "
                    f"{sorted(step.depends_on)} but its input carries no "
                    "{{{id}}} placeholder — dependent agent steps must reference "
                    "upstream output (e.g. \"... using {{1}} ...\")"
                )

    def _check_prose_grounding(self, plan: Plan) -> None:
        """Reject agent steps that name upstream steps in prose without a placeholder.

        Live trace ecd93eb4: a reasoning step with empty depends_on said
        "Using the retrieved chunks from step 1 ... and step 2 ..." with no
        {{1}} {{2}}. It ran in parallel with the rag.query steps, saw no
        chunks, and asked the user to re-upload. The structural
        _check_reasoning_grounding cannot see this shape (no depends_on), so
        match the prose signal — but ONLY when the plan actually holds
        rag.query chunks siblings, to avoid false positives on benign
        "step N" prose.
        """
        has_chunks = any(
            s.tool_id == "rag.query"
            and (s.expected_output_type or "").lower() == "chunks"
            for s in plan.steps
        )
        if not has_chunks:
            return
        for step in plan.steps:
            if step.tool_id:
                continue
            stack: list[object] = [step.input]
            texts: list[str] = []
            while stack:
                value = stack.pop()
                if isinstance(value, str):
                    texts.append(value)
                elif isinstance(value, dict):
                    stack.extend(value.values())
                elif isinstance(value, list):
                    stack.extend(value)
            body = " ".join(texts)
            if _PLACEHOLDER.search(body):
                continue  # structurally grounded; other checks own the edges
            if _PROSE_STEP_REF.search(body):
                raise PlanValidationError(
                    f"step {step.step_id} ({step.agent_id or 'agent'}) mentions "
                    "upstream steps in prose but carries no {{{id}}} placeholder "
                    "— add depends_on plus {{1}} {{2}} references so the chunks "
                    "are injected before it runs"
                )

    def _check_placeholder_edges(self, plan: Plan) -> None:
        """Reject {{id}} placeholders with no matching ordering edge.

        Placeholders are the engine's only dataflow mechanism
        (plan_graph._resolve_value): a step referencing {{1}} without
        depends_on ["1"] runs in the same super-step as step 1, so the
        placeholder never resolves and the literal "{{1}}" reaches the LLM
        (observed: "I don't have the retrieved chunks"). Plan.from_model
        auto-wires this shape, so reaching here means a dangling reference
        to a non-existent step — fail honest instead of executing garbled.
        """
        by_id = {s.step_id: s for s in plan.steps}

        def placeholders_in(value: object) -> set[str]:
            found: set[str] = set()
            if isinstance(value, str):
                found.update(_PLACEHOLDER.findall(value))
            elif isinstance(value, dict):
                for v in value.values():
                    found.update(placeholders_in(v))
            elif isinstance(value, list):
                for v in value:
                    found.update(placeholders_in(v))
            return found

        def upstream(step_id: str) -> set[str]:
            seen: set[str] = set()
            stack = list(by_id[step_id].depends_on)
            while stack:
                dep = stack.pop()
                if dep in seen or dep not in by_id:
                    continue
                seen.add(dep)
                stack.extend(by_id[dep].depends_on)
            return seen

        for step in plan.steps:
            refs = placeholders_in(step.input)
            if not refs:
                continue
            ups = upstream(step.step_id)
            missing = refs - ups - {step.step_id}
            unknown = {r for r in refs if r not in by_id}
            if unknown:
                raise PlanValidationError(
                    f"step {step.step_id} references unknown step(s) "
                    f"{sorted(unknown)} via {{{{id}}}} placeholder — "
                    "placeholders must reference real step_ids"
                )
            if missing:
                raise PlanValidationError(
                    f"step {step.step_id} uses {{{{id}}}} placeholder(s) "
                    f"{sorted(missing)} with no ordering edge — add "
                    f"{sorted(missing)} to depends_on so upstream runs first"
                )

    def _check_parallel_fanout(self, plan: Plan) -> None:
        """Reject >5 parallel long-text agent steps off the same parent.

        The backend is a single local Ollama server: concurrent long writes
        contend for it (previously observed 25-77s per write on qwen2.5:14b,
        hence the old cap of 2). The cap is raised to 5 for the current
        model/host, which sustains at least five parallel writes. Beyond
        that, long writes must be chained sequentially or collapsed into
        one step. Machine outputs (numbers/chunks) stay exempt so the
        numbers+answer pattern in Example E keeps passing.
        """
        groups: dict[tuple[str, ...], list] = {}
        for step in plan.steps:
            if step.tool_id or not step.depends_on:
                continue
            eot = (step.expected_output_type or "text").lower()
            if eot in ("numbers", "chunks"):
                continue
            key = tuple(sorted(step.depends_on))
            groups.setdefault(key, []).append(step)
        for key, siblings in groups.items():
            if len(siblings) > 5:
                ids = sorted(s.step_id for s in siblings)
                raise PlanValidationError(
                    f"steps {ids} fan out {len(siblings)} parallel long-text "
                    f"agent steps off {list(key)} — chain long writes "
                    "sequentially (2 depends_on [\"1\"], 3 depends_on [\"2\"], ...) "
                    "or collapse them into ONE step"
                )

    def _check_tool_required_fields(self, plan: Plan) -> None:
        """Reject tool steps missing their schema-required input fields.

        Catches malformed planner output before execution (e.g. doc.generate
        with {"message": ...} instead of {"title": ..., "sections": ...}).
        `corpus_id` is exempt: the engine injects run-scoped truth at
        execution time, so the planner must never emit it. A value containing
        a {{id}} placeholder counts as present — it resolves at runtime.

        doc.generate's either/or contract (report vs verbatim file) cannot
        be a static `required` list, so its full input contract is checked
        through the tool itself — unless the input carries placeholders,
        which resolve at runtime past what a static contract can judge.
        """
        schemas: dict[str, dict] = {}
        for tool in self._tool_registry.manifest():
            schemas[tool["tool_id"]] = tool.get("input_schema", {}) or {}
        for step in plan.steps:
            if not step.tool_id:
                continue
            schema = schemas.get(step.tool_id, {})
            required = schema.get("required", []) or []
            if required and isinstance(step.input, dict):
                for field in required:
                    if field == "corpus_id":
                        continue
                    value = step.input.get(field)
                    if value is None:
                        raise PlanValidationError(
                            f"step {step.step_id} ({step.tool_id}) is missing required "
                            f"input field {field!r} — refusing to execute an "
                            "unguarded tool call"
                        )
                    if isinstance(value, str):
                        if not value.strip():
                            raise PlanValidationError(
                                f"step {step.step_id} ({step.tool_id}) has empty "
                                f"required input field {field!r}"
                            )
                    elif isinstance(value, list) and not value:
                        raise PlanValidationError(
                            f"step {step.step_id} ({step.tool_id}) has empty "
                            f"required input field {field!r}"
                        )
            if step.tool_id == "doc.generate" and isinstance(step.input, dict):
                if _contains_placeholder(step.input):
                    continue  # runtime-resolved; execution validates
                try:
                    tool = self._tool_registry.get(step.tool_id)
                except KeyError:
                    continue
                err = tool.validate_input(step.input)
                if err is not None:
                    raise PlanValidationError(
                        f"step {step.step_id} (doc.generate) {err} — refusing "
                        "to execute an unguarded tool call"
                    )

    def _check_retrieval_output_types(self, plan: Plan) -> None:
        """Enforce intermediate output typing so HIDE works downstream.

        The aggregator hides `chunks`/`numbers` and shows everything else.
        A rag.query typed as `text`/`answer` promotes raw chunks into the
        user-visible summary (observed live: "Step 1 (rag.query): <ToC...>"
        prefixing the MCQ answer). Fail honest at plan time.
        """
        for step in plan.steps:
            if step.tool_id == "rag.query":
                eot = (step.expected_output_type or "").lower()
                if eot != "chunks":
                    raise PlanValidationError(
                        f"step {step.step_id} (rag.query) must declare "
                        f"expected_output_type \"chunks\" (got "
                        f"{step.expected_output_type!r}) — raw chunks are "
                        "intermediates and must stay hidden from the answer"
                    )
                if not isinstance(step.input, dict):
                    continue
                file_id = step.input.get("file_id")
                if file_id is not None:
                    if not isinstance(file_id, str) or not file_id.strip():
                        raise PlanValidationError(
                            f"step {step.step_id} (rag.query) has empty "
                            f"file_id — use a literal snapshot id or omit it"
                        )
                    if "{{" in file_id or "}}" in file_id:
                        raise PlanValidationError(
                            f"step {step.step_id} (rag.query) file_id must be "
                            f"a literal snapshot id, never a {{{{id}}}} placeholder"
                        )
                mode = step.input.get("mode")
                if mode is not None and (
                    not isinstance(mode, str)
                    or mode.strip().lower() not in ("specific", "overview")
                ):
                    raise PlanValidationError(
                        f"step {step.step_id} (rag.query) mode must be "
                        f"'specific' or 'overview' (got {mode!r})"
                    )
