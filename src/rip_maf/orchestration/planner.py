"""Planner — thin provider holder (L3 mega-prompt removed).

Planning is now L1 router (sole dispatcher) → L2 deterministic builders →
L3 ReAct. The single-shot DAG mega-prompt was removed: it grew with every
hardening rule and forced intent classification + DAG shape + placeholder
wiring into one call. This class keeps its constructor shape so composition
(main.py, deps.py, tests) is unchanged, and exposes `provider` for the
router and ReAct loop.
"""

from __future__ import annotations

import logging

from rip_maf.agents.registry import AgentRegistry
from rip_maf.core.config import settings
from rip_maf.providers.base import ModelProvider
from rip_maf.tools.registry import ToolRegistry

logger = logging.getLogger("orchestration.planner")


class Planner:
    def __init__(
        self,
        provider: ModelProvider,
        registry: AgentRegistry,
        tool_registry: ToolRegistry | None = None,
    ):
        self._provider = provider
        self._registry = registry
        self._tool_registry = tool_registry or ToolRegistry()
        self._model = settings.ollama_default_model

    @property
    def provider(self) -> ModelProvider:
        """Planning provider, shared by the L1 router and L3 ReAct loop."""
        return self._provider
