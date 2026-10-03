"""Tool ABC — Tier 2 utility capabilities.

An agent is a capability container (model + reasoning); a TOOL is a single-
purpose function (RAG query, chart rendering, file generation).
Tools do one thing honestly or fail explicitly; `rag.query` may call the LLM
solely for retrieval query decomposition (ADR-030 amendment) — no answer synthesis.

Metadata mirrors the Agent pattern so the Router/Builder tool menu renders from one
shape: tool_id / name / description / input_schema / input_example, plus the
effect class:

- "read-only": free to run within budget (rag.query).
- "sandboxed": bounded compute producing data/artifacts (plot.chart,
  doc.generate, doc.convert).
- "side-effecting": outbound or mutating. RIP is local single-user, so
  tools execute directly with no approval gate (the executor runs every
  registered tool unconditionally); the class is kept for manifest honesty.

RIP port: no plugin system — inherit directly from Tool (ADR-017).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from rip_maf.core.classutils import is_abstract
from rip_maf.tools.schema import check_input

ToolEffect = Literal["read-only", "sandboxed", "side-effecting"]

EFFECT_READ_ONLY: ToolEffect = "read-only"
EFFECT_SANDBOXED: ToolEffect = "sandboxed"
EFFECT_SIDE_EFFECTING: ToolEffect = "side-effecting"

EFFECT_CLASSES: tuple[str, ...] = ("read-only", "sandboxed", "side-effecting")


class ToolRequest(BaseModel):
    tool_id: str
    step_id: str = ""
    trace_id: str = ""
    input: dict[str, Any] = Field(default_factory=dict)
    timeout_ms: int = 30000


class ToolResponse(BaseModel):
    tool_id: str
    ok: bool = True
    output: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class Tool(ABC):
    """Deterministic single-purpose capability (Tier 2).

    Concrete tools set class-level metadata; __init_subclass__ enforces the
    required fields — the same pattern as Agent.

    ## The input contract lives here, not in its callers

    `input_schema` + `validate_input` + `input_example` ARE the contract.
    `execute()` must call `validate_input` first, so a tool cannot ship
    rules its own execution ignores; external callers (the ReAct pre-flight,
    anything that wants to fail before spending an execution) go through
    `validate_input` / `explain_invalid` instead of re-implementing rules.
    That duplication is not hypothetical: `plot.chart`'s contract was
    hand-copied into `react_engine.py` four times and drifted until the
    model could not produce a two-series chart (trace `c9b02eef`).

    Why a pre-flight check exists at all, given `execute()` validates:
    economics, not correctness. A step that fails is retried
    `_DEFAULT_MAX_RETRIES` (2) times inside `run_plan_graph`, so a malformed
    call costs three executions and three identical errors in the scratchpad.
    Caught before execution it costs one idle turn and no retries.
    """

    tool_id: str
    name: str
    description: str
    input_schema: ClassVar[dict] = {}
    output_schema: ClassVar[dict] = {}
    #: One copyable call, shown to an LLM that got the shape wrong. The
    #: failure message alone did not teach the 3B model the multi-series
    #: shape (trace c9b02eef); an example did.
    input_example: ClassVar[str] = ""
    effect_class: ToolEffect = EFFECT_READ_ONLY  # type: ignore[assignment]
    requires_approval: bool = False
    cost_class: str = "low"

    _REQUIRED_METADATA: ClassVar[tuple[str, ...]] = ("tool_id", "name", "description")

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if is_abstract(cls):
            return
        missing = [attr for attr in Tool._REQUIRED_METADATA if getattr(cls, attr, None) is None]
        if missing:
            raise TypeError(f"{cls.__name__} must define class attributes: {', '.join(missing)}")
        effect = getattr(cls, "effect_class", None)
        if effect not in EFFECT_CLASSES:
            raise TypeError(
                f"{cls.__name__} effect_class {effect!r} must be one of {', '.join(EFFECT_CLASSES)}"
            )

    def validate_input(self, tool_input: dict[str, Any]) -> str | None:
        """Authoritative input contract. None = valid, else the reason.

        The default enforces the declarative schema (closed key set,
        required fields, types, enums). Override to add cross-field rules a
        schema cannot state, and call `super().validate_input()` first.
        """
        return check_input(self.input_schema, tool_input, self.tool_id)

    def required_for_model(self, tool_input: dict[str, Any]) -> str | None:
        """Stricter than `validate_input`: rules for LLM-authored calls only.

        Direct callers (builders, tests, older clients) may legitimately omit
        what a model should always provide — e.g. `plot.chart` derives a
        heading from the data when `title` is absent, so an untitled chart
        renders, but a model that never writes titles is worth correcting.
        """
        return None

    def explain_invalid(self, tool_input: dict[str, Any]) -> str | None:
        """Failure message + the tool's own example, for LLM feedback.

        Returns None when the input is valid. Callers that want to tell a
        model what shape it got wrong use this; callers that just want to
        reject a call use `validate_input`.
        """
        err = self.required_for_model(tool_input) or self.validate_input(tool_input)
        if err is None:
            return None
        return f"{err}. {self.input_example}" if self.input_example else err

    def invalid_response(self, tool_input: dict[str, Any]) -> ToolResponse:
        """The standard fail-honest response for a rejected input.

        Enforces the runtime contract only — `required_for_model` is advice
        for LLM-authored calls, and `execute()` must honour direct callers
        that legitimately omit it.
        """
        err = self.validate_input(tool_input)
        if err is None:
            return ToolResponse(tool_id=self.tool_id, ok=True)
        return ToolResponse(tool_id=self.tool_id, ok=False, output=None, error=err)

    @abstractmethod
    def execute(self, request: ToolRequest) -> ToolResponse:
        """Run the tool deterministically; never raise — return ok=False."""
