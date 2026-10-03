"""Aggregator — deterministic assembly of step outputs (Q36).

Turns a plan's per-step outputs into ONE final answer with NO LLM synthesis
step (merge decision: cuts cost and latency; predictable and testable).

Type-aware deterministic rules (ADR-023 as amended):
- HIDE intermediate outputs (`expected_output_type` in
  chunks/numbers/observation, or the `corpus.inspect` freshness probe)
  unless they are the only output (anti-blank fallback);
- SHOW terminal outputs (answer/summary/text/document/chart/clarification,
  plus any unknown type — fail-visible, never fail-blank); chart/SVG
  outputs aggregate to a short placeholder (the SVG bytes travel via the
  SSE `artifacts` event as a download URL and render inline as <img>);
- single SHOW → verbatim (no `Step N` prefix, preserves chat tone);
  multi-SHOW → labeled concatenation (`Step <id> (<executor>): ...`,
  same placeholder per chart step);
- a step needing clarification → its question verbatim (never mangled);
- all steps failed (or nothing executed) → the joined error strings;
- partial runs append every failure line so successes never mask failures.
- `shown` / `hidden` / `visibility` expose the SHOW/HIDE decision per step
  (answer bubble = SHOW only; Steps panel = all, collapsed; Langfuse map).

`conflicts` is always empty (no LLM to detect contradictions).
"""
from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

from rip_maf.agents.base import StepStatus
from rip_maf.orchestration.plan import Plan
from rip_maf.orchestration.results import ExecutionResult, StepResult

logger = logging.getLogger("orchestration.aggregator")

#: Unresolved {{id}} placeholders must never reach the user (trace cfbaa9c3:
#: a reasoning step answered "I don't have access to {{1}} and {{2}}...").
#: Outputs carrying them are treated as ungrounded failures, not answers.
_UNRESOLVED_PLACEHOLDER = re.compile(r"\{\{\s*[A-Za-z0-9_-]+\s*\}\}")

#: Placeholder replacing raw chart SVG in the user-visible summary. The SVG
#: bytes stay on StepResult.output (placeholder resolution, traces) and are
#: delivered as a file via the SSE `artifacts` event; the summary (and the
#: persisted assistant message) carries only this short text so the chat
#: never renders raw `<svg>` markup.
CHART_PLACEHOLDER = "Chart generated — see Artifacts below."


def _summarizable(output: str | None) -> str:
    """Return placeholder when output carries chart SVG, else verbatim."""
    text = output or ""
    if "<svg" in text.lower():
        return CHART_PLACEHOLDER
    return text


class AggregationResult(BaseModel):
    status: str  # "success" | "partial" | "failed"
    plan_incomplete: bool
    summary: str
    conflicts: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    #: SHOW/HIDE decision per step_id ("show" | "hide"), in plan order terms.
    #: Drives the frontend answer bubble (SHOW only) vs the collapsed-all
    #: Steps panel, and the Langfuse aggregate-span visibility map.
    shown: list[str] = Field(default_factory=list)
    hidden: list[str] = Field(default_factory=list)
    visibility: dict[str, str] = Field(default_factory=dict)


class Aggregator:
    def aggregate(self, plan: Plan, result: ExecutionResult) -> AggregationResult:
        successful = [r for r in result.step_results if r.status is StepStatus.SUCCESS]
        failed = [r for r in result.step_results if r.status is StepStatus.FAILURE]
        if not result.step_results:
            return AggregationResult(
                status="failed",
                plan_incomplete=True,
                summary="No steps were executed.",
            )

        if len(successful) == len(result.step_results):
            status = "success"
        elif successful:
            status = "partial"
        else:
            status = "failed"

        plan_incomplete = not result.succeeded

        # A step asking for clarification returns its question verbatim —
        # synthesis would mangle the question.
        clarification = next(
            (r for r in result.step_results if r.needs_clarification), None
        )
        if clarification is not None:
            return AggregationResult(
                status="success",
                plan_incomplete=False,
                summary=clarification.output or "",
                needs_clarification=True,
                shown=[clarification.step_id],
                hidden=[r.step_id for r in result.step_results
                        if r.step_id != clarification.step_id],
                visibility={r.step_id: ("show" if r.step_id == clarification.step_id else "hide")
                            for r in result.step_results},
            )

        # All failed: join the error strings honestly.
        if status == "failed":
            summary = "; ".join(
                f"Step {r.step_id} ({r.agent_id}) failed: {r.error}" for r in failed
            )
            logger.info(
                "aggregated plan=%s status=failed failures=%d",
                plan.plan_id, len(failed),
            )
            return AggregationResult(
                status="failed", plan_incomplete=True, summary=summary,
                shown=[],
                hidden=[r.step_id for r in result.step_results],
                visibility={r.step_id: "hide" for r in result.step_results},
            )

        # Type-aware assembly: hide intermediates, show terminals in plan
        # order. Step internals remain available via step_completed events
        # for the UI collapsible; the summary is the persisted answer.
        step_meta = {s.step_id: s for s in plan.steps} if plan.steps else {}
        ordered = (
            [next((r for r in result.step_results if r.step_id == s.step_id), None)
             for s in plan.steps]
            if plan.steps else list(result.step_results)
        )
        ordered_successful = [r for r in ordered if r is not None and r.status is StepStatus.SUCCESS]

        def _hidden(step_id: str) -> bool:
            meta = step_meta.get(step_id)
            if meta is None:
                return False  # unknown step: fail-visible, never fail-blank
            eot = (meta.expected_output_type or "text").lower()
            # "observation" = a non-terminal ReAct agent step: scratchpad
            # kept for the loop, never the user's answer.
            if eot in ("chunks", "numbers", "observation"):
                return True
            return meta.executor_id == "corpus.inspect"

        shown = [r for r in ordered_successful if not _hidden(r.step_id)]
        # Ungrounded-leak guard: a terminal output still carrying {{id}}
        # means placeholder resolution failed upstream (validator bypass or
        # pre-validation plan). Showing it leaks internals ("paste {{1}}");
        # demote to a failure line so the run retries honestly instead.
        leaked = [r for r in shown if _UNRESOLVED_PLACEHOLDER.search(r.output or "")]
        if leaked:
            shown = [r for r in shown if r not in leaked]
            for r in leaked:
                failed.append(
                    StepResult(
                        step_id=r.step_id,
                        agent_id=r.agent_id,
                        status=StepStatus.FAILURE,
                        error="ungrounded response (unresolved placeholders)",
                    )
                )
            # Recompute status: demoted terminals are failures now.
            effective_success = [r for r in ordered_successful if r not in leaked]
            if effective_success and failed:
                status = "partial"
            elif not effective_success:
                status = "failed"
            plan_incomplete = True
        # Duplicate-output guard: identical terminal outputs (trace 07fb4f59
        # plotted the same Avg-Tokens SVG twice, r2 then r3) would render
        # the same plot/text twice in chat. Keep the first, hide the rest —
        # the content is already visible, nothing is lost.
        if len(shown) > 1:
            seen_outputs: set[str] = set()
            deduped = []
            for r in shown:
                key = r.output or ""
                if key in seen_outputs:
                    continue
                seen_outputs.add(key)
                deduped.append(r)
            shown = deduped
        if shown:
            if len(shown) == 1:
                summary = _summarizable(shown[0].output)
            else:
                summary = "\n\n".join(
                    f"Step {r.step_id} ({r.agent_id}): {_summarizable(r.output)}"
                    for r in shown
                )
        elif ordered_successful and not failed:
            # Anti-blank fallback: every success was intermediate (e.g. a
            # lone rag.query or numbers step) and nothing failed — surface
            # the last one. With failures present the errors below are the
            # honest answer; a hidden CSV dump would only add noise.
            summary = _summarizable(ordered_successful[-1].output)
        else:
            summary = ""

        # Partial runs must not hide failures behind successes (e.g. an
        # inspect listing masking failed converts) — append them honestly.
        if failed:
            failures = "\n".join(
                f"Step {r.step_id} ({r.agent_id}) failed: {r.error}" for r in failed
            )
            summary = f"{summary}\n\n{failures}" if summary.strip() else failures

        shown_ids = [r.step_id for r in shown]
        if not shown and ordered_successful and not failed:
            # Anti-blank fallback surfaces the last intermediate — it IS shown.
            shown_ids = [ordered_successful[-1].step_id]
        hidden_ids = [r.step_id for r in ordered_successful if r.step_id not in shown_ids]
        visibility = {r.step_id: ("hide" if r.step_id in hidden_ids else "show")
                      for r in ordered_successful}

        logger.info(
            "aggregated plan=%s status=%s successes=%d shown=%s hidden=%s",
            plan.plan_id, status, len(successful), shown_ids, hidden_ids,
        )
        return AggregationResult(
            status=status,
            plan_incomplete=plan_incomplete,
            summary=summary,
            shown=shown_ids,
            hidden=hidden_ids,
            visibility=visibility,
        )
