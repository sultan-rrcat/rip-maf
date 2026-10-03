from __future__ import annotations

import logging
import time
from typing import ClassVar

from rip_maf.agents.base import Agent, DelegationRequest, DelegationResponse, StepStatus
from rip_maf.core import constants
from rip_maf.core.config import settings
from rip_maf.providers.base import ModelProvider

logger = logging.getLogger("agents.reasoning")

class ReasoningAgent(Agent):
    agent_id = "reasoning"
    name = "Reasoning Agent"
    description = (
        "General conversational assistance, explanation, and brainstorming; "
        "fallback when no specialized capability fits."
    )
    input_schema: ClassVar[dict] = {"message": "str", "history": "optional list of {role, content}"}
    requires_permission = False
    side_effecting = False
    cost_class = "low"

    def __init__(self, provider: ModelProvider):
        self._provider = provider

    def execute(self, request: DelegationRequest) -> DelegationResponse:
        start = time.perf_counter()
        logger.info("reasoning start step=%s trace=%s", request.step_id, request.trace_id)
        try:
            # 1. Pull `message` (and optional `history`) from request.input
            message = request.input.get("message")
            if not message:
                logger.warning("reasoning missing message step=%s", request.step_id)
                return DelegationResponse(
                    step_id=request.step_id,
                    status=StepStatus.FAILURE,
                    output=None,
                    confidence=constants.CONFIDENCE_LOW,
                    error="Execution failed: 'message' is required in input",
                )
            history = request.input.get("history", [])
            context = request.input.get("context")

            # 2. Build the OpenAI-style `messages` list: conversation context
            #    (short-term memory) first, then history, then current
            messages = []
            if context:
                messages.append({"role": "system", "content": f"Conversation context:\n{context}"})
            messages.extend(
                {"role": turn["role"], "content": turn["content"]}
                for turn in history
                if isinstance(turn, dict) and "role" in turn and "content" in turn
            )
            messages.append({"role": "user", "content": message})

            # 3. Call the provider (streaming when the caller wants deltas)
            if request.on_delta is not None:
                parts: list[str] = []
                for chunk in self._provider.generate_stream(
                    model=settings.ollama_default_model,
                    messages=messages,
                    temperature=settings.default_temperature,
                    max_tokens=settings.default_max_tokens,
                ):
                    parts.append(chunk)
                    request.on_delta(chunk)
                output_text = "".join(parts)
            else:
                output_text = self._provider.generate(
                    model=settings.ollama_default_model,
                    messages=messages,
                    temperature=settings.default_temperature,
                    max_tokens=settings.default_max_tokens,
                )

            # 4. Return success + output + confidence
            result = DelegationResponse(
                step_id=request.step_id,
                status=StepStatus.SUCCESS,
                output=output_text,
                confidence=constants.CONFIDENCE_SUCCESS,
            )
        except Exception as e:
            # 5. Wrap the provider call so ANY exception becomes a failure response.
            logger.exception("reasoning failed step=%s", request.step_id)
            result = DelegationResponse(
                step_id=request.step_id,
                status=StepStatus.FAILURE,
                output=None,
                confidence=constants.CONFIDENCE_LOW,
                error=f"Execution failed: {e!s}",
            )
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "reasoning done step=%s status=%s duration_ms=%.0f",
            request.step_id, result.status.value, duration_ms,
        )
        return result
