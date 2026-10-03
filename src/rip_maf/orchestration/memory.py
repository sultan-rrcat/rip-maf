"""Conversation context for orchestration.

Open WebUI owns conversation history and forwards it per request, so this
module does no folding or summarization: it renders the newest window of the
caller-supplied turns verbatim for injection into planner/agent prompts.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

WINDOW_SIZE = 10


class MemoryContext(BaseModel):
    recent: list[dict] = Field(default_factory=list)  # [{role, content}], newest last

    def as_prompt(self) -> str:
        """Compact text form for injection into planner / agent prompts."""
        if not self.recent:
            return ""
        lines = "\n".join(f"{m['role']}: {m['content']}" for m in self.recent)
        return f"Recent conversation:\n{lines}"


def estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def build_history_context(messages: list[dict]) -> str | None:
    """Render caller-supplied conversation history (newest window, verbatim).

    Returns None when there is nothing to render.
    """
    if not messages:
        return None
    recent = messages[-WINDOW_SIZE:]
    rendered = [
        {"role": m.get("role", "user"), "content": str(m.get("content", ""))}
        for m in recent
        if str(m.get("content", "")).strip()
    ]
    if not rendered:
        return None
    return MemoryContext(recent=rendered).as_prompt() or None


__all__ = ["WINDOW_SIZE", "MemoryContext", "build_history_context", "estimate_tokens"]
