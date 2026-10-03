"""Shared text post-processing for streaming-capable providers.

Ollama serves reasoning models whose <think> blocks can land in raw content.
The non-streaming path strips them after the fact; the streaming path needs
the same semantics INCREMENTALLY, because a tag can be split across chunk
boundaries ("<thi" + "nk>...").

ThinkFilter is a tiny state machine with that one job:

- closed <think>...</think> bodies are suppressed,
- text after an UNClOSED <think> is dropped (same rule as strip_think: the
  generation budget was likely exhausted inside reasoning),
- at most a tag-length tail is ever held back, so visible text flows with
  minimal latency.

Feed every raw chunk to feed(); call close() at end-of-stream to flush.
feed() may return "" (nothing visible yet) — callers must not yield empties,
per the generate_stream() non-empty-chunk contract.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger("providers.streaming")

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_OPEN = "<think>"
_CLOSE = "</think>"
# Longest tail that could still become a tag: "</think>" is the longer tag.
_HOLDBACK = len(_CLOSE) - 1


def strip_think(content: str) -> str:
    """Remove closed <think> blocks; drop everything after an unclosed one."""
    content = THINK_RE.sub("", content).strip()
    if _OPEN in content:
        logger.warning(
            "unclosed <think> block dropped (generation budget likely "
            "exhausted inside reasoning)"
        )
        content = content.split(_OPEN, 1)[0].strip()
    return content


def _holdback_len(buf: str, tag: str) -> int:
    """Length of the longest buf tail that is a strict prefix of tag."""
    for n in range(min(len(buf), len(tag) - 1), 0, -1):
        if buf.endswith(tag[:n]):
            return n
    return 0


class ThinkFilter:
    """Incremental <think>-block suppressor (see module docstring)."""

    def __init__(self) -> None:
        self._buf = ""
        self._thinking = False
        self._warned = False

    def feed(self, chunk: str) -> str:
        """Consume one raw chunk; return the newly visible text (maybe "")."""
        self._buf += chunk
        out: list[str] = []
        while self._buf:
            if self._thinking:
                end = self._buf.find(_CLOSE)
                if end == -1:
                    # No close tag yet; keep a tail in case it arrives split.
                    keep = self._buf[-_HOLDBACK:] if len(self._buf) > _HOLDBACK else self._buf
                    self._buf = keep
                    break
                self._buf = self._buf[end + len(_CLOSE):]
                self._thinking = False
                continue
            start = self._buf.find(_OPEN)
            if start == -1:
                # No open tag; hold a tail that could still become one.
                hold = _holdback_len(self._buf, _OPEN)
                if hold:
                    out.append(self._buf[:-hold])
                    self._buf = self._buf[-hold:]
                else:
                    out.append(self._buf)
                    self._buf = ""
                break
            out.append(self._buf[:start])
            self._buf = self._buf[start + len(_OPEN):]
            self._thinking = True
        return "".join(out)

    def close(self) -> str:
        """End-of-stream flush: emit held-back text, drop an unclosed block."""
        if self._thinking:
            if not self._warned:
                self._warned = True
                logger.warning(
                    "unclosed <think> block dropped (generation budget likely "
                    "exhausted inside reasoning)"
                )
            self._buf = ""
            self._thinking = False
            return ""
        tail, self._buf = self._buf, ""
        return tail
