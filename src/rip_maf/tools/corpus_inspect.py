"""corpus.inspect tool — list files in a corpus (dynamic discovery).

Read-only inventory for the planner: returns `{file_id, file_name,
file_size, file_status}` rows for the run's corpus so the planner can
resolve "which document" without guessing. `corpus_id` is injected by
the orchestrator from `Run.corpus_id`, never LLM-generated (same
contract as rag.query).

Effect class: read-only.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from rip_maf.tools.base import Tool, ToolRequest, ToolResponse

logger = logging.getLogger("tools.corpus_inspect")


def _format_files(files: list[dict]) -> str:
    if not files:
        return "(no files in this corpus)"
    parts = [
        f"{f.get('file_name', '?')} ({f.get('file_status', '?')})"
        for f in files
    ]
    return f"{len(files)} file(s): " + ", ".join(parts)


class CorpusInspectTool(Tool):
    tool_id = "corpus.inspect"
    name = "Corpus Inspect"
    description = (
        "List files in this corpus with id, name, size and status "
        "(ready/processing/error). Call first when the request refers to "
        "'this document', 'convert', 'which file', or when the corpus "
        "snapshot may be stale."
    )
    input_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            # Injected by the engine; the model must never emit it, and the
            # schema checker exempts it from `required` for that reason.
            "corpus_id": {"type": "string"},
        },
        "required": ["corpus_id"],
        "additionalProperties": False,
    }
    input_example: ClassVar[str] = "corpus.inspect {} (takes no input)"
    output_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "files": {"type": "array"},
        },
    }
    effect_class = "read-only"  # type: ignore[assignment]
    cost_class = "low"

    def execute(self, request: ToolRequest) -> ToolResponse:
        corpus_id = request.input.get("corpus_id")
        if not corpus_id:
            return ToolResponse(
                tool_id=self.tool_id,
                ok=False,
                output=None,
                error="'corpus_id' is required in input (injected by the orchestrator, never the LLM)",
            )
        try:
            from rip_maf.core.db import pg_connection

            with pg_connection() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT file_id, file_name, file_size, file_status
                    FROM files
                    WHERE corpus_id = %s
                    ORDER BY created_at ASC
                    """,
                    (str(corpus_id),),
                )
                rows = cur.fetchall()
        except Exception as e:
            logger.exception("corpus.inspect query failed")
            return ToolResponse(
                tool_id=self.tool_id,
                ok=False,
                output=None,
                error=f"inspect failed: {e}",
            )
        files: list[dict[str, Any]] = [
            {
                "file_id": str(r[0]),
                "file_name": r[1],
                "file_size": r[2],
                "file_status": r[3],
            }
            for r in rows
        ]
        return ToolResponse(
            tool_id=self.tool_id,
            ok=True,
            output=_format_files(files),
            data={"files": files, "corpus_id": str(corpus_id)},
        )
