"""Coding Agent — writes, explains, reviews, and debugs code.

Mirrors ReasoningAgent's shape, but with a code-specific system prompt and the
shared Ollama default model. RIP port: no plugin system — direct Agent subclass.
"""
from __future__ import annotations

import logging
import time
from typing import ClassVar

from rip_maf.agents.base import Agent, DelegationRequest, DelegationResponse, StepStatus
from rip_maf.core import constants
from rip_maf.core.config import settings
from rip_maf.providers.base import ModelProvider

logger = logging.getLogger("agents.coding")

_SYSTEM_PROMPT = (
    "You are an expert software engineer. Write clean, correct, "
    "well-documented code. Explain your approach and note edge cases."
)


class CodingAgent(Agent):
    agent_id = "coding"
    name = "Coding Agent"
    description = (
        "Write, explain, review, and debug code across languages and frameworks."
    )
    input_schema: ClassVar[dict] = {"message": "str", "language": "optional str"}
    requires_permission = False
    side_effecting = False
    cost_class = "medium"

    def __init__(self, provider: ModelProvider):
        self._provider = provider

    def execute(self, request: DelegationRequest) -> DelegationResponse:
        start = time.perf_counter()
        logger.info("coding start step=%s trace=%s", request.step_id, request.trace_id)
        try:
            message = request.input.get("message")
            if not message:
                logger.warning("coding missing message step=%s", request.step_id)
                return DelegationResponse(
                    step_id=request.step_id,
                    status=StepStatus.FAILURE,
                    output=None,
                    confidence=constants.CONFIDENCE_LOW,
                    error="Execution failed: 'message' is required in input",
                )

            history = request.input.get("history", [])
            language = request.input.get("language")
            context = request.input.get("context")

            # 2. Build the OpenAI-style `messages` list: system prompt,
            #    conversation context (short-term memory), history, then current
            messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
            if context:
                messages.append({"role": "system", "content": f"Conversation context:\n{context}"})
            messages.extend(
                {"role": turn["role"], "content": turn["content"]}
                for turn in history
                if isinstance(turn, dict) and "role" in turn and "content" in turn
            )
            if language:
                messages.append({"role": "user", "content": f"Language: {language}"})
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
            logger.exception("coding failed step=%s", request.step_id)
            result = DelegationResponse(
                step_id=request.step_id,
                status=StepStatus.FAILURE,
                output=None,
                confidence=constants.CONFIDENCE_LOW,
                error=f"Execution failed: {e!s}",
            )
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "coding done step=%s status=%s duration_ms=%.0f",
            request.step_id, result.status.value, duration_ms,
        )

        return result
