"""In-memory tool registry — the Planner's Tier 2 menu.

Mirrors AgentRegistry: hold tool instances by tool_id, expose read-only
manifest metadata. The Planner renders capability menus from here; the
Validator checks referenced ids against it.

RIP port: no plugin system — static composition via
get_default_tool_registry (ADR-017).
"""

from __future__ import annotations

from rip_maf.providers.base import ModelProvider
from rip_maf.tools.base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.tool_id in self._tools:
            raise ValueError(f"Tool already registered: {tool.tool_id}")
        self._tools[tool.tool_id] = tool

    def get(self, tool_id: str) -> Tool:
        if tool_id not in self._tools:
            raise KeyError(f"Unknown tool: {tool_id}")
        return self._tools[tool_id]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, tool_id: object) -> bool:
        return tool_id in self._tools

    def manifest(self) -> list[dict]:
        result: list[dict] = []
        for tool in self._tools.values():
            result.append(
                {
                    "tool_id": tool.tool_id,
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                    "output_schema": tool.output_schema,
                    "effect_class": tool.effect_class,
                    "requires_approval": tool.requires_approval,
                    "cost_class": tool.cost_class,
                }
            )
        return result


def get_default_tool_registry(
    rag: object | None = None,
    provider: ModelProvider | None = None,
) -> ToolRegistry:
    """Build the fixed tool set (no plugin system — static composition).

    `rag` is bound to rag.query (the lifespan VectorRAG singleton). Unbound
    tools fail honest at execution.
    """
    # Local imports: keeps `rip_maf.tools.registry` importable without pulling
    # tool modules (and their settings/deps) until the factory runs.
    from rip_maf.tools.corpus_inspect import CorpusInspectTool
    from rip_maf.tools.doc_convert import DocConvertTool
    from rip_maf.tools.doc_generate import DocGenerateTool
    from rip_maf.tools.plot_chart import PlotChartTool
    from rip_maf.tools.rag_query import RagQueryTool

    registry = ToolRegistry()
    registry.register(RagQueryTool(rag=rag, provider=provider))
    registry.register(PlotChartTool())
    registry.register(DocGenerateTool())
    registry.register(DocConvertTool())
    registry.register(CorpusInspectTool())
    return registry
