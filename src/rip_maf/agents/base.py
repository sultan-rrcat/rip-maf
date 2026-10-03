from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, Field, model_validator

from rip_maf.core.classutils import is_abstract


class StepStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    LOW_CONFIDENCE = "low_confidence"


class DelegationRequest(BaseModel):
    step_id: str
    trace_id: str
    input: dict[str, Any]
    context_ref: str | None = None
    timeout_ms: int = 30000
    # Streaming sink for partial output (Phase 2): when set, agents whose
    # provider can stream forward text chunks here as they arrive and join
    # them into the final output. Excluded from dumps — a callable is not
    # serializable and must never leak into logs or traces.
    on_delta: Callable[[str], None] | None = Field(default=None, exclude=True)

class DelegationResponse(BaseModel):
    step_id: str
    status: StepStatus = StepStatus.SUCCESS
    output: str | None = None
    confidence: float = Field(default=1.0)
    error: str | None = None
    needs_clarification: bool = False  # true -> ask the user a question, not a failure

    @model_validator(mode="after")
    def validate_error_status(self) -> DelegationResponse:
        is_failure = self.status == StepStatus.FAILURE
        has_error = self.error is not None

        if is_failure and not has_error:
            raise ValueError("error is required when status is failure")
        if not is_failure and has_error:
            raise ValueError("error must be None when status is not failure")

        return self

class Agent(ABC):
    # Required metadata. Declared as class attributes (not abstract @property):
    # every concrete agent sets them as simple class-level constants, and
    # __init_subclass__ enforces that they exist — same guarantee, less ceremony.
    agent_id: str
    name: str
    description: str
    input_schema: ClassVar[dict] = {}
    requires_permission: bool = False
    side_effecting: bool = False
    cost_class: str = "low"

    _REQUIRED_METADATA: ClassVar[tuple[str, ...]] = ("agent_id", "name", "description")

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Abstract intermediates (e.g. plugins.api.AgentPlugin) may omit the
        # metadata; the check binds once the class is concrete — see
        # app.core.classutils.is_abstract for why inspect.isabstract is not used.
        if is_abstract(cls):
            return
        missing = [
            attr for attr in Agent._REQUIRED_METADATA
            if getattr(cls, attr, None) is None
        ]
        if missing:
            raise TypeError(
                f"{cls.__name__} must define class attributes: {', '.join(missing)}"
            )

    @abstractmethod
    def execute(self, request: DelegationRequest) -> DelegationResponse:
        """Execute the agent's core logic for a delegation request."""
