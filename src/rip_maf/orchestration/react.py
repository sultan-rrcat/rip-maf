"""L3 ReAct — compatibility shim over `react_engine`.

The implementation lives on `ReActEngine` in `react_engine.py`. This
module re-exports the public names so existing callers (orchestrator,
tests) are unaffected.

Tool input rules are NOT re-exported: they live with the tools
(`Tool.validate_input` / `Tool.explain_invalid`) and are reached through the
registry. This module used to re-export a copy of them, which drifted.
"""

from rip_maf.orchestration.react_engine import (
    _ANSWER_FALLBACK_FIELDS,
    _OVERVIEW_HINTS,
    _TOOL_OUTPUT_TYPES,
    MAX_REACT_ITERATIONS,
    REACT_SCHEMA,
    ReActEngine,
    ReactResult,
    _action_signature,
    _chart_observation,
    _default_react_mode,
    _fallback_answer_text,
    _normalize_react_input,
    _output_type,
    _plot_data_key,
    _plot_rules,
    _route_block,
    _synthesis_evidence_line,
    _tool_examples,
    _validate_react_input,
    run_react,
)

__all__ = [
    "MAX_REACT_ITERATIONS",
    "REACT_SCHEMA",
    "_ANSWER_FALLBACK_FIELDS",
    "_OVERVIEW_HINTS",
    "_TOOL_OUTPUT_TYPES",
    "ReActEngine",
    "ReactResult",
    "_action_signature",
    "_chart_observation",
    "_default_react_mode",
    "_fallback_answer_text",
    "_normalize_react_input",
    "_output_type",
    "_plot_data_key",
    "_plot_rules",
    "_route_block",
    "_synthesis_evidence_line",
    "_tool_examples",
    "_validate_react_input",
    "run_react",
]
