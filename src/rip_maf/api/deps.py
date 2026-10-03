"""Shared HTTP-layer dependencies: provider, registries, orchestrator, runs.

No PluginManager — RIP composes a fixed set of objects directly (static
composition). `main.py` lifespan builds the full stack once VectorRAG is
loaded (so `rag.query` gets the lifespan singleton bound) and installs it
via `configure()`; the lazy fallbacks below exist for direct/test use and
fail honest when Ollama is unreachable (rag unbound → `rag.query` errors
at execution, never at import).

Tests install fakes via `configure()` or FastAPI `dependency_overrides`.
"""
from __future__ import annotations

import logging
from typing import Any

from rip_maf.agents.registry import AgentRegistry, get_default_agent_registry
from rip_maf.orchestration.aggregator import Aggregator
from rip_maf.orchestration.orchestrator import Orchestrator
from rip_maf.orchestration.planner import Planner
from rip_maf.orchestration.validator import PlanValidator
from rip_maf.providers.base import ModelProvider
from rip_maf.providers.ollama import OllamaProvider
from rip_maf.providers.tracing import wrap_provider
from rip_maf.tools.registry import ToolRegistry, get_default_tool_registry

logger = logging.getLogger("api.deps")

_provider: ModelProvider | None = None
_agent_registry: AgentRegistry | None = None
_tool_registry: ToolRegistry | None = None
_orchestrator: Orchestrator | None = None
_run_manager: Any | None = None  # app.runs.manager.RunManager (deferred: import cycle)


def configure(
    *,
    provider: ModelProvider | None = None,
    agent_registry: AgentRegistry | None = None,
    tool_registry: ToolRegistry | None = None,
    orchestrator: Orchestrator | None = None,
    run_manager: Any | None = None,
) -> None:
    """Install the process runtime (main.py lifespan; tests)."""
    global _provider, _agent_registry, _tool_registry, _orchestrator, _run_manager
    if provider is not None:
        _provider = provider
    if agent_registry is not None:
        _agent_registry = agent_registry
    if tool_registry is not None:
        _tool_registry = tool_registry
    if orchestrator is not None:
        _orchestrator = orchestrator
    if run_manager is not None:
        _run_manager = run_manager


def reset() -> None:
    """Drop the installed runtime (tests only)."""
    global _provider, _agent_registry, _tool_registry, _orchestrator, _run_manager
    _provider = _agent_registry = _tool_registry = _orchestrator = _run_manager = None


def get_model_provider() -> ModelProvider:
    """The active Ollama provider (fail-honest at build when Ollama is down).

    Tracing-wrapped when langfuse_enabled (no-op wrapper otherwise).
    """
    global _provider
    if _provider is None:
        _provider = wrap_provider(OllamaProvider())
        logger.info("model provider built lazily")
    return _provider


def get_agent_registry() -> AgentRegistry:
    global _agent_registry
    if _agent_registry is None:
        _agent_registry = get_default_agent_registry(get_model_provider())
    return _agent_registry


def get_tool_registry() -> ToolRegistry:
    """Fixed RIP tool set. Lifespan-installed one has `rag` bound; the lazy
    fallback leaves it unbound and `rag.query` fails honest at execution."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = get_default_tool_registry(provider=get_model_provider())
    return _tool_registry


def get_orchestrator() -> Orchestrator:
    """The orchestrator, composed ONCE (recomposed only via configure)."""
    global _orchestrator
    if _orchestrator is None:
        agent_registry = get_agent_registry()
        tool_registry = get_tool_registry()
        provider = get_model_provider()
        _orchestrator = Orchestrator(
            Planner(provider, agent_registry, tool_registry),
            PlanValidator(agent_registry, tool_registry),
            Aggregator(),
            agent_registry,
            tool_registry=tool_registry,
        )
    return _orchestrator


def get_run_manager() -> Any:
    """Process-wide RunManager — runs must be addressable across requests."""
    global _run_manager
    if _run_manager is None:
        from rip_maf.runs.manager import RunManager

        _run_manager = RunManager(
            provider=get_model_provider(), orchestrator=get_orchestrator()
        )
    return _run_manager
