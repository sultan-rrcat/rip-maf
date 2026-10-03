"""Execution result models — the engine's output contract.

These models are consumed across the orchestration layer (aggregator,
orchestrator, plan graph) and the API schemas, so they live in a neutral
module instead of inside either graph implementation. Keeping them
pydantic-clean also keeps graph state serializable for checkpointing.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from rip_maf.agents.base import StepStatus


class StepResult(BaseModel):
    step_id: str
    agent_id: str
    status: StepStatus
    output: str | None = None
    error: str | None = None
    needs_clarification: bool = False
    # Tool step payload: the raw ToolResponse.data dict for successful tool
    # steps (svg/image_b64/docx_b64/results/sources/...). Agent steps leave
    # this empty. Carried through so the run worker can build SSE `sources`
    # and `artifacts` events without re-executing anything.
    data: dict = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    trace_id: str
    step_results: list[StepResult] = []

    @property
    def succeeded(self) -> bool:
        return all(r.status is StepStatus.SUCCESS for r in self.step_results)
