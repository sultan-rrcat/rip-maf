"""Execution-plan data model.

A Plan is a small DAG: ordered steps, each naming an agent, its inputs, and
its dependencies. The Planner emits it (structured output), the Validator
checks it, the Execution Engine runs it. These pydantic models ARE the
contract — keep them in one place.
"""
from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel, Field

logger = logging.getLogger("orchestration.plan")


class PlanStep(BaseModel):
    """One DAG node: EITHER an agent step (agent_id) OR a tool step (tool_id).

    Exactly-one-of is enforced by the Validator (fail-honest
    PlanValidationError), not by construction — so malformed planner output
    surfaces as a validation failure with a clear message, never a crash.
    """

    step_id: str
    agent_id: str = ""
    tool_id: str | None = None
    input: dict = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    expected_output_type: str = "text"

    @property
    def kind(self) -> str:
        return "tool" if self.tool_id else "agent"

    @property
    def executor_id(self) -> str:
        return self.tool_id or self.agent_id


#: Step-wiring keys that belong at step top level, never inside `input`.
_WIRING_KEYS = ("step_id", "depends_on", "expected_output_type")


def _hoist_wiring_keys(step: dict, plan_id: str, step_id: str) -> None:
    """Hoist step-wiring keys buried in `input` to top-level step fields.

    Observed live (ornith-1.5:9b): the planner emits the right shape but
    nests depends_on/expected_output_type inside input — twice in a row,
    exhausting the retry budget on an otherwise executable plan. Mutates
    `step` in place: a wiring key found in input moves up when the top
    level lacks it (same repair philosophy as depends_on auto-wire and
    the rag.query chunks default); an input copy EQUAL to the top-level
    value is dropped as a harmless echo; DIFFERENT values are a
    contradiction and raise ValueError (engine retry contract) naming
    the conflict. A differing input step_id is the old whole-step-nesting
    shape and keeps the "buries step" message. Non-string/non-list
    nested values are left for the Validator's buried-step backstop.
    """
    raw_input = step.get("input")
    if not isinstance(raw_input, dict):
        return
    for key in _WIRING_KEYS:
        if key not in raw_input:
            continue
        nested = raw_input[key]
        top = step.get(key)
        if key == "depends_on":
            if not isinstance(nested, list) or not all(isinstance(d, str) for d in nested):
                continue  # malformed: validator backstop rejects by key
            if not isinstance(top, list) or not top:
                step[key] = list(nested)
                del raw_input[key]
                logger.info(
                    "plan %s step %s: hoisted nested %s to top level",
                    plan_id, step_id, key,
                )
            elif list(top) != list(nested):
                raise ValueError(
                    f"step {step_id} contradicts itself: top-level "
                    f"depends_on {top!r} vs input depends_on {nested!r} — "
                    "keep exactly one"
                )
            else:
                del raw_input[key]
            continue
        if not isinstance(nested, str):
            continue  # malformed: validator backstop rejects by key
        if top is None or (isinstance(top, str) and not top):
            step[key] = nested
            del raw_input[key]
            logger.info(
                "plan %s step %s: hoisted nested %s to top level",
                plan_id, step_id, key,
            )
        elif top != nested:
            if key == "step_id":
                raise ValueError(
                    f"step {step_id} buries step {nested!r} inside its "
                    "input (input keys belong at step top level) — move "
                    "agent_id/tool_id, step_id, depends_on and "
                    "expected_output_type OUT of input"
                )
            raise ValueError(
                f"step {step_id} contradicts itself: top-level "
                f"{key} {top!r} vs input {key} {nested!r} — keep exactly one"
            )
        else:
            del raw_input[key]


def _hoist_nested_executor(step: dict, plan_id: str, step_id: str) -> None:
    """Hoist executor ids buried in `input` to top-level step fields.

    Mutates `step` in place. Only fires when the top level names NO
    executor (empty agent_id and empty/None tool_id) and `input` holds
    exactly one truthy string under "agent_id"/"tool_id" — that key is
    moved up and deleted from input (no tool/agent schema uses those keys,
    so the move is lossless). Anything else (both present, non-strings,
    top level already set) is left for the Validator to reject.
    """
    if step.get("agent_id") or step.get("tool_id"):
        return
    raw_input = step.get("input")
    if not isinstance(raw_input, dict):
        return
    nested_agent = raw_input.get("agent_id")
    nested_tool = raw_input.get("tool_id")
    has_agent = isinstance(nested_agent, str) and bool(nested_agent.strip())
    has_tool = isinstance(nested_tool, str) and bool(nested_tool.strip())
    if has_agent == has_tool:  # both or neither: not our repair shape
        return
    if has_agent:
        step["agent_id"] = nested_agent.strip()
    else:
        step["tool_id"] = nested_tool.strip()
    del raw_input["agent_id" if has_agent else "tool_id"]
    logger.info(
        "plan %s step %s: hoisted nested %s to top level",
        plan_id, step_id, "agent_id" if has_agent else "tool_id",
    )


class Plan(BaseModel):
    plan_id: str
    goal: str
    steps: list[PlanStep] = Field(default_factory=list)

    def is_trivial(self) -> bool:
        return len(self.steps) == 0

    @classmethod
    def from_model(cls, plan_id: str, goal: str, raw_steps: list[dict]) -> Plan:
        """Build a Plan from raw step JSON, deterministically fixing
        common LLM slips (and failing honest with an actionable message
        on the rest):
          - non-object array elements (a stray `"step_id": "3"` string where a
            step object belongs — observed live): rejected as ValueError naming
            the position instead of validating a phantom step;
          - step-wiring keys nested inside `input` (depends_on,
            expected_output_type, step_id — observed live, ornith-1.5:9b
            nests them twice running): hoisted to top level when absent
            there, dropped when equal (harmless echo), rejected as
            ValueError on contradiction (keeps the "buries step" message
            for a differing input step_id — the old whole-step-nesting
            shape). Executor ids are TOP-LEVEL step fields, never input
            keys;
          - executor ids nested inside `input` ({"input": {"tool_id": ...}})
            instead of top-level step fields: hoisted when exactly one is
            present and the top level has neither (observed live: the model
            buries tool_id/agent_id in input, validator then rejects the
            step — hoisting turns a fail-honest abort into an executable
            plan; both-present stays rejected);
          - omitted `expected_output_type` on rag.query steps: filled with
            "chunks" (observed live: the model emits the right shape but drops
            the optional key, and the default "text" would promote raw chunks
            into the answer). Explicitly wrong values are left untouched for
            the Validator to reject;
          - duplicate step_ids: first occurrence keeps its id, later ones get
            a numeric suffix;
          - exact-duplicate steps (same agent_id + same input): the later copy
            is DROPPED so it never executes or costs money, and any depends_on
            reference to it is redirected to the kept twin;
          - prefixed dependency ids ("step_1" where the id is "1"): normalized
            to the real id when a trailing-number match exists — the same
            parse-time repair philosophy as the dedup above. Genuinely unknown
            references are left untouched for the Validator to reject.
        """
        for i, raw in enumerate(raw_steps):
            if not isinstance(raw, dict):
                # ValueError (not TypeError): callers catch ValueError, so
                # this surfaces as an honest failure instead of a run crash.
                raise ValueError(  # noqa: TRY004 - honest-failure contract needs ValueError
                    f"step {i + 1} is not an object (got {type(raw).__name__} "
                    f"{str(raw)[:120]!r}) — each steps[] element must be an "
                    "object with step_id plus exactly one of agent_id/tool_id "
                    "as TOP-LEVEL keys"
                )
        raw_ids = {r.get("step_id") or "step" for r in raw_steps}
        _PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z0-9_-]+)\s*\}\}")

        def _refs_in(value: object) -> set[str]:
            found: set[str] = set()
            if isinstance(value, str):
                found.update(_PLACEHOLDER_RE.findall(value))
            elif isinstance(value, dict):
                for v in value.values():
                    found.update(_refs_in(v))
            elif isinstance(value, list):
                for v in value:
                    found.update(_refs_in(v))
            return found

        def canon_dep(dep: str) -> str:
            if dep in raw_ids:
                return dep
            m = re.search(r"(\d+)$", dep)
            if m and m.group(1) in raw_ids:
                return m.group(1)
            return dep

        resolved: dict[str, str] = {}  # raw id -> kept id (renames AND drops)
        used: set[str] = set()
        seen: dict[tuple[str, str, str], str] = {}  # (agent_id, tool_id, input) -> kept step_id
        steps: list[PlanStep] = []

        for raw in raw_steps:
            old = raw.get("step_id") or "step"
            if old not in resolved:
                new = old
                while new in used:
                    new = f"{new}_x"
            else:
                n = 2
                new = f"{old}_{n}"
                while new in used or new in resolved.values():
                    n += 1
                    new = f"{old}_{n}"

            step = dict(raw)
            step["step_id"] = new
            if isinstance(step.get("input"), dict):
                step["input"] = dict(step["input"])  # hoist mutates; don't touch caller's dict
            _hoist_wiring_keys(step, plan_id, new)
            _hoist_nested_executor(step, plan_id, new)
            if step.get("tool_id") == "rag.query" and "expected_output_type" not in step:
                step["expected_output_type"] = "chunks"
                logger.info(
                    "plan %s step %s: defaulted omitted expected_output_type to chunks",
                    plan_id, new,
                )
            step["depends_on"] = [
                resolved.get(canon_dep(d), canon_dep(d))
                for d in step.get("depends_on", [])
            ]
            # Auto-wire: a {{id}} placeholder referencing a known step implies
            # an ordering edge. The planner (ornith-1.5:9b, observed live)
            # emits grounded messages but omits depends_on, so the steps run
            # in parallel and the placeholder never resolves (literal "{{1}}"
            # reaches the LLM, which asks to re-upload). Same repair
            # philosophy as hoist/dedup above: add missing edges, leave
            # genuinely unknown refs for the Validator to reject.
            if isinstance(step.get("input"), dict):
                for ref in _refs_in(step["input"]):
                    target = canon_dep(ref)
                    wired = resolved.get(target, target)
                    if (wired in raw_ids or wired in used) and wired not in step["depends_on"]:
                        step["depends_on"].append(wired)
                        logger.info(
                            "plan %s step %s: auto-wired depends_on %s from placeholder",
                            plan_id, new, wired,
                        )
            candidate = PlanStep(**step)

            sig = (
                candidate.agent_id,
                candidate.tool_id or "",
                json.dumps(candidate.input, sort_keys=True),
            )
            twin = seen.get(sig)
            if twin is not None:
                resolved[old] = twin  # drop: redirect references to the twin
                continue
            resolved[old] = new
            used.add(new)
            seen[sig] = new
            steps.append(candidate)

        return cls(plan_id=plan_id, goal=goal, steps=steps)
