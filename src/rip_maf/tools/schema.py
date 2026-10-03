"""Minimal JSON-Schema subset checker — the declarative half of a tool contract.

A tool's input contract has three layers and exactly one home each:

1. `Tool.input_schema` — the declarative half (this module): field names,
   types, required fields, enums, and whether the key set is closed.
2. `Tool.validate_input` — the imperative half: cross-field rules a schema
   cannot express (exactly-one-of, "values must be a flat array", length
   agreement). Only the tool itself writes these.
3. `Tool.input_example` — one copyable call, so an LLM that gets the shape
   wrong is shown the shape rather than told the shape is wrong.

Callers reach all three through `Tool.validate_input` / `Tool.explain_invalid`
(never by copying rules — that is what produced four divergent plot.chart
contracts in `react_engine.py`).

Deliberately a subset, no `jsonschema` dependency (offline-first): `type`,
`properties`, `required`, `enum`, `minItems`, `additionalProperties: false`,
and `items`. Anything a plan needs that JSON Schema cannot say is
cross-step dataflow (placeholder wiring, "numbers must come from an
upstream step") and stays in `orchestration/validator.py`, which is
placeholder-aware in a way this deliberately is not.
"""

from __future__ import annotations

from typing import Any

#: Keys the ENGINE owns, not the model: `corpus_id` is run-scoped truth
#: (the plan validator already exempts it from required-field checks, and
#: ADR-027 forbids the model emitting it) and `expected_output_type` is
#: step metadata that `run_plan_graph` attaches to the resolved input. Both
#: are injected after planning, so a closed key set must tolerate them —
#: exempt for exactly the same reason the validator does.
ENGINE_INJECTED_KEYS = frozenset({"corpus_id", "expected_output_type"})

_TYPE_NAMES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(value: Any) -> str:
    # bool before int: isinstance(True, int) is True in Python.
    for py_type, name in _TYPE_NAMES.items():
        if py_type is bool and not isinstance(value, bool):
            continue
        if isinstance(value, py_type):
            return name
    return type(value).__name__


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True  # unknown constraint: never invent a rejection


def _check_field(tool_id: str, path: str, value: Any, spec: dict) -> str | None:
    enum = spec.get("enum")
    if enum is not None and value not in enum:
        label = f"field {path!r}" if path else "input"
        return f"{tool_id} {label} must be one of {list(enum)} (got {value!r})"
    expected = spec.get("type")
    if isinstance(expected, str) and not _type_ok(value, expected):
        label = f"field {path!r}" if path else "input"
        return (
            f"{tool_id} {label} must be {expected} "
            f"(got {_json_type(value)})"
        )
    if isinstance(value, list):
        min_items = spec.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            label = f"field {path!r}" if path else "input"
            return f"{tool_id} {label} needs at least {min_items} item(s)"
        item_spec = spec.get("items")
        if isinstance(item_spec, dict):
            for i, item in enumerate(value):
                child = f"{path}[{i}]" if path else f"[{i}]"
                if isinstance(item_spec.get("properties"), dict) and isinstance(
                    item, dict
                ):
                    inner = check_input(
                        {"type": "object", "properties": item_spec["properties"]},
                        item,
                        tool_id,
                        path_prefix=child,
                    )
                    if inner is not None:
                        return inner
                else:
                    err = _check_field(tool_id, child, item, item_spec)
                    if err is not None:
                        return err
    return None


def check_input(
    schema: dict,
    value: Any,
    tool_id: str,
    *,
    path_prefix: str = "",
) -> str | None:
    """Return None when `value` satisfies `schema`, else the reason.

    Only the JSON-Schema subset listed in the module docstring is honoured;
    unknown keywords are ignored (forward-compatible, never a false
    rejection).
    """
    if not isinstance(value, dict):
        return f"{tool_id} input must be a JSON object (got {_json_type(value)})"
    props: dict = schema.get("properties") or {}
    required: list = schema.get("required") or []

    if schema.get("additionalProperties") is False:
        extra = sorted(set(value) - set(props) - ENGINE_INJECTED_KEYS)
        if extra:
            return (
                f"{tool_id} does not accept {extra!r} — its fields are "
                f"{sorted(props)}"
            )

    for field in required:
        if field in ENGINE_INJECTED_KEYS:
            continue  # injected by the engine at execution time
        if field not in value or value[field] is None:
            return (
                f"{tool_id} requires {field!r} "
                f"(required fields: {sorted(required)})"
            )
        if isinstance(value[field], str) and not value[field].strip():
            return f"{tool_id} requires a non-empty {field!r}"
        if isinstance(value[field], list) and not value[field]:
            return f"{tool_id} requires a non-empty {field!r} array"

    for field, spec in props.items():
        if field not in value or value[field] is None or not isinstance(spec, dict):
            continue
        path = f"{path_prefix}.{field}" if path_prefix else field
        err = _check_field(tool_id, path, value[field], spec)
        if err is not None:
            return err
    return None
