"""plot.chart tool — deterministic stdlib SVG charts.

Deliberately dependency-free: bars and lines render as inline SVG from the
standard library, so no new runtime dependency is introduced. File
persistence arrives with artifact delivery (Q34); today the SVG travels
inline in the tool response.

Effect class: sandboxed — bounded compute producing an artifact.

RIP port: no plugin system — direct Tool subclass (ADR-017).
"""

from __future__ import annotations

from html import escape
from typing import ClassVar

from rip_maf.tools.base import Tool, ToolRequest, ToolResponse

_MAX_POINTS = 50
_WIDTH, _HEIGHT = 640, 360
_PAD_LEFT, _PAD_RIGHT, _PAD_TOP, _PAD_BOTTOM = 56, 16, 36, 44

#: Series palette for multi-series charts (line strokes / bar fills).
_PALETTE = ("#4a90d9", "#e94f37", "#44af69", "#f2a541", "#7b6fd0")


def _scale(values: list[float], height: float) -> list[float]:
    peak = max(values) if values else 0.0
    base = min(0.0, min(values) if values else 0.0)
    span = peak - base or 1.0
    return [(v - base) / span * height for v in values]


def render_svg(
    chart_type: str, labels: list[str], values: list[float], *, title: str = "",
    series: list[tuple[str, list[float]]] | None = None,
) -> str:
    """Render a bar/line chart as an SVG document string.

    Single-series (``series=None``): legacy labels+values behavior.
    Multi-series: ``labels`` are the shared x-axis, ``series`` holds
    ``(name, values)`` pairs drawn in palette order with a legend.
    """
    multi = list(series) if series else []
    plot_w = _WIDTH - _PAD_LEFT - _PAD_RIGHT
    plot_h = _HEIGHT - _PAD_TOP - _PAD_BOTTOM
    all_values = [v for _, vals in multi for v in vals] if multi else list(values)
    heights_all = _scale(all_values, plot_h)
    per_series: list[tuple[str, list[float]]] = []
    if multi:
        offset = 0
        for name, vals in multi:
            per_series.append((name, heights_all[offset:offset + len(vals)]))
            offset += len(vals)
    else:
        per_series = [("", heights_all)]
    n = len(labels)
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH}" height="{_HEIGHT}" role="img">',
    ]
    if title:
        parts.append(
            f"<text x='{_WIDTH // 2}' y='24' text-anchor='middle' "
            f"font-size='16' font-family='sans-serif'>{escape(title)}</text>"
        )
    parts.append(
        f"<rect x='{_PAD_LEFT}' y='{_PAD_TOP}' width='{plot_w}' "
        f"height='{plot_h}' fill='none' stroke='#888'/>"
    )
    if chart_type == "bar":
        gap = 6.0
        if multi:
            group_w = plot_w / n if n else 0
            bar_w = (group_w - gap * (len(multi) + 1)) / len(multi) if n else 0
            for i in range(n):
                for j, (name, vals) in enumerate(multi):
                    h = per_series[j][1][i]
                    x = _PAD_LEFT + i * group_w + gap + j * (bar_w + gap)
                    y = _PAD_TOP + plot_h - h
                    color = _PALETTE[j % len(_PALETTE)]
                    parts.append(
                        f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' "
                        f"height='{h:.1f}' fill='{color}'>"
                        f"<title>{escape(name)} {escape(labels[i])}: "
                        f"{vals[i]}</title></rect>"
                    )
        else:
            heights = per_series[0][1]
            bar_w = (plot_w - gap * (n + 1)) / n if n else 0
            for i, (label, h) in enumerate(zip(labels, heights, strict=True)):
                x = _PAD_LEFT + gap + i * (bar_w + gap)
                y = _PAD_TOP + plot_h - h
                parts.append(
                    f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' "
                    f"height='{h:.1f}' fill='#4a90d9'>"
                    f"<title>{escape(label)}: {values[i]}</title></rect>"
                )
    else:  # line
        step = plot_w / (n - 1) if n > 1 else 0
        series_vals: list[tuple[str, list[float]]] = multi if multi else [("", values)]
        for j, ((name, vals), (_pname, hts)) in enumerate(
            zip(series_vals, per_series)
        ):
            color = _PALETTE[j % len(_PALETTE)] if multi else "#4a90d9"
            points = " ".join(
                f"{_PAD_LEFT + i * step:.1f},{_PAD_TOP + plot_h - h:.1f}"
                for i, h in enumerate(hts)
            )
            parts.append(
                f"<polyline points='{points}' fill='none' stroke='{color}' "
                f"stroke-width='2'/>"
            )
            for i, (label, h) in enumerate(zip(labels, hts, strict=True)):
                x = _PAD_LEFT + i * step
                y = _PAD_TOP + plot_h - h
                tip = f"{escape(name + ' ' if name else '')}{escape(label)}: {vals[i]}"
                parts.append(
                    f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3' fill='{color}'>"
                    f"<title>{tip}</title></circle>"
                )
        if multi:
            lx = _WIDTH - _PAD_RIGHT - 8
            for j, (name, _hts) in enumerate(per_series):
                color = _PALETTE[j % len(_PALETTE)]
                y = _PAD_TOP + 8 + j * 18
                parts.append(
                    f"<rect x='{lx - 130:.1f}' y='{y:.1f}' width='12' height='12' "
                    f"fill='{color}'/>"
                    f"<text x='{lx - 114:.1f}' y='{y + 10:.1f}' font-size='11' "
                    f"font-family='sans-serif'>{escape(name)}</text>"
                )
    # x labels: first, middle, last (keeps small SVGs readable)
    for i in sorted({0, n // 2, n - 1}):
        x = _PAD_LEFT + (i + 0.5) * (plot_w / n) if chart_type == "bar" else _PAD_LEFT + i * step
        parts.append(
            f"<text x='{x:.1f}' y='{_HEIGHT - 12}' text-anchor='middle' "
            f"font-size='11' font-family='sans-serif'>{escape(labels[i])}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _default_title(
    labels: list[str], multi: list[tuple[str, list[float]]]
) -> str:
    """Fallback chart title when the caller omits one.

    The ReAct prompt and pre-flight validation both require a short
    title, but direct calls (L2 builders, tests, older clients) may
    still omit it — an untitled chart renders with no heading. Derive
    something descriptive from the data: series names scoped by the
    x-axis labels (e.g. "YOLOv26s, YOLOv26n by FASDD_CV vs AgniNetra").
    """
    base = " vs ".join(labels) if len(labels) <= 5 else f"{len(labels)} data points"
    if multi:
        names = ", ".join(name for name, _ in multi)
        return f"{names} by {base}"[:160]
    return base[:160]


class PlotChartTool(Tool):
    tool_id = "plot.chart"
    name = "Plot Chart"
    description = (
        "Render a bar or line chart as inline SVG from labels + numeric values. "
        "Single series: labels + values. Multi-series (comparisons): shared "
        "labels + series: [{label, values}] (at most 5 series, drawn with a legend)."
    )
    input_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "chart_type": {"type": "string", "enum": ["bar", "line"]},
            # Untyped items on purpose: `labels` are str()-coerced, and
            # `values` legitimately arrive as numeric strings — a resolved
            # "{{id}}" placeholder lands as ONE CSV element ("0.82, 0.88")
            # that `_to_numbers` splits (ADR-027). Type-checking them here
            # would reject the very shape the placeholder path produces.
            "labels": {"type": "array", "minItems": 1},
            "values": {"type": "array"},
            "series": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "values": {"type": "array"},
                    },
                    "required": ["label", "values"],
                },
            },
            "title": {"type": "string"},
        },
        "required": ["chart_type", "labels"],
        # Closed key set: an invented per-entity array (`india_values`,
        # `china_values`) or label array (`series_labels`) would otherwise
        # pass as a single-series call and silently drop a series — the
        # ADR-027 hallucination class wearing a new costume (trace c9b02eef).
        "additionalProperties": False,
    }
    #: The shape a 3B model could not derive from prose. Trace c9b02eef
    #: proposed `series_labels` + a parallel `china_values` array twice; the
    #: prose rule ("grouped comparisons use series:[{label, values}]") was
    #: not copyable, so it needs saying as a literal call.
    input_example: ClassVar[str] = (
        "Comparison of two or more entities = ONE 'series' array over shared "
        "'labels': "
        '{"chart_type": "line", "labels": ["2015", "2016", "2017"], "series": ['
        '{"label": "India", "values": [7.2, 7.1, 7.0]}, '
        '{"label": "China", "values": [6.9, 6.5, 6.0]}], '
        '"title": "GDP growth %: India vs China"}. '
        "There is no 'series_labels' key and no per-entity array like "
        "'india_values'/'china_values' — put every entity in 'series'. "
        "ONE entity = 'labels' + 'values' only, never 'series'."
    )
    output_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "svg": {"type": "string"},
            "chart_type": {"type": "string"},
            "point_count": {"type": "integer"},
            "series_count": {"type": "integer"},
        },
    }
    effect_class = "sandboxed"  # type: ignore[assignment]
    cost_class = "low"

    #: At most this many series per chart (legend space + readability).
    _MAX_SERIES = 5

    def validate_input(self, request_input: dict) -> str | None:
        """Full input contract, cross-field rules included.

        The declarative schema cannot say "exactly one of values/series",
        "values are flat numbers", or "series values agree with labels", so
        those live here — beside the renderer, not in a caller.
        """
        err = super().validate_input(request_input)
        if err is not None:
            return err
        labels = request_input.get("labels")
        values = request_input.get("values")
        series_in = request_input.get("series")
        if not isinstance(labels, list) or not labels:
            return "'labels' must be a non-empty array"
        if series_in is not None:
            if values is not None:
                return (
                    "pass either 'values' (single series) or 'series', never both"
                )
            if not isinstance(series_in, list) or not series_in:
                return "'series' must be a non-empty array of {label, values}"
            if len(series_in) > self._MAX_SERIES:
                return f"at most {self._MAX_SERIES} series per chart"
            for entry in series_in:
                if not isinstance(entry, dict):
                    return (
                        "'series' entries must be {label, values} objects"
                    )
                name = entry.get("label")
                if not isinstance(name, str) or not name.strip():
                    return (
                        "'series' entries need a non-empty string 'label'"
                    )
                numbers = self._to_numbers(entry.get("values"))
                if numbers is None:
                    return f"series {name!r} 'values' must all be numbers"
                if len(numbers) != len(labels):
                    return (
                        f"series {name!r} has {len(numbers)} values "
                        f"but {len(labels)} labels"
                    )
            if len(labels) > _MAX_POINTS:
                return f"at most {_MAX_POINTS} points per chart"
            return None
        if not isinstance(values, list) or not values:
            return (
                "'values' must be a non-empty array "
                "(or pass 'series' for multi-series)"
            )
        numbers = self._to_numbers(values)
        if numbers is None:
            return (
                "'values' must all be numbers (a flat array, one per label); "
                "for a comparison pass series: [{label, values}] with shared "
                "'labels' instead of nesting arrays inside 'values'"
            )
        if len(labels) != len(numbers):
            return "'labels' and 'values' must have the same length"
        if len(labels) > _MAX_POINTS:
            return f"at most {_MAX_POINTS} points per chart"
        return None

    def required_for_model(self, request_input: dict) -> str | None:
        """A model-authored chart must name its metric.

        Direct callers may omit it — `_default_title` derives a heading from
        the data (trace affdbbd4: ReAct omitted titles on 3 of 4 charts).
        """
        if self.validate_input(request_input) is not None:
            return None  # report the real problem, not the missing title
        if not str(request_input.get("title", "")).strip():
            return (
                "plot.chart needs a short 'title' naming the metric and "
                "comparison (e.g. 'mAP@50-95: FASDD_CV vs AgniNetra')"
            )
        return None

    def execute(self, request: ToolRequest) -> ToolResponse:
        # The contract is validated before anything is rendered, so this
        # method can assume chart_type/labels/values|series are coherent.
        invalid = self.invalid_response(request.input)
        if invalid.error is not None:
            return invalid
        chart_type = request.input.get("chart_type")
        labels = request.input.get("labels")
        values = request.input.get("values")
        series_in = request.input.get("series")
        title = str(request.input.get("title", "") or "")
        str_labels = [str(label) for label in labels]
        if series_in is not None:
            multi = [
                (str(entry["label"]), self._to_numbers(entry["values"]))
                for entry in series_in
            ]
            svg = render_svg(
                chart_type, str_labels, [],
                title=title or _default_title(str_labels, multi),
                series=multi,
            )
            return ToolResponse(
                tool_id=self.tool_id,
                ok=True,
                output=svg,
                data={
                    "svg": svg,
                    "chart_type": chart_type,
                    "point_count": len(str_labels),
                    "series_count": len(multi),
                },
            )
        numbers = self._to_numbers(values)
        svg = render_svg(
            chart_type, str_labels, numbers,
            title=title or _default_title(str_labels, []),
        )
        return ToolResponse(
            tool_id=self.tool_id,
            ok=True,
            output=svg,
            data={
                "svg": svg,
                "chart_type": chart_type,
                "point_count": len(numbers),
                "series_count": 1,
            },
        )

    @staticmethod
    def _to_numbers(values: object) -> list[float] | None:
        """Coerce a values array to floats, splitting placeholder CSV strings.

        The engine resolves a whole upstream text output (e.g. a numbers
        step returning "0.82, 0.88") into ONE string element. Split
        comma-separated strings back into points so one placeholder can
        fill a whole series; empties are dropped (covers stray
        leading/trailing commas from templates). Returns None when any
        element is not numeric.
        """
        if not isinstance(values, list):
            return None
        flat: list[object] = []
        for v in values:
            if isinstance(v, str) and "," in v:
                flat.extend(part.strip() for part in v.split(","))
                continue
            flat.append(v)
        flat = [v for v in flat if not (isinstance(v, str) and v == "")]
        try:
            return [float(v) for v in flat]  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
