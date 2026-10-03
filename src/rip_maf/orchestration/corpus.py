"""Shared corpus-state seam for L2 builders and L3 ReAct.

Both the deterministic builders and the ReAct fallback need to tri-state
the corpus file snapshot (is retrieval provably useless?). This module
owns the snapshot parsing so neither consumer leaks into the other's
internals — previously `react.py` imported the private `_corpus_state`
from `builders.py` across the seam.
"""

from __future__ import annotations

import re

#: Snapshot lines look like "report.pdf [ready] id=abc123" (see
#: runs/manager._load_file_snapshot). Only ready files convert.
_SNAPSHOT_FILE = re.compile(
    r"(.+?)\s*\[(ready|processing|uploading|error)\]\s*id=(\S+)"
)

#: The snapshot's "N file(s): " prefix. The regex above is anchored at the
#: line start, so without this strip the first file's name carries the
#: prefix — harmless for id matching, wrong in anything user-facing (the
#: convert counter-question would read "2 file(s): a.pdf").
_SNAPSHOT_COUNT_PREFIX = re.compile(r"^\d+\s+files?\(s\):\s*")


def _snapshot_files(corpus_context: str | None) -> list[tuple[str, str, str]]:
    """Parse (name, status, file_id) triples out of the snapshot string."""
    if not corpus_context:
        return []
    cleaned = []
    for name, status, fid in _SNAPSHOT_FILE.findall(corpus_context):
        fid_clean = fid.strip().rstrip(";,")
        if fid_clean:
            cleaned.append(
                (_SNAPSHOT_COUNT_PREFIX.sub("", name.strip()), status.strip().lower(), fid_clean)
            )
    return cleaned


def _ready_files(corpus_context: str | None) -> list[tuple[str, str]]:
    """Return (name, file_id) for ALL ready files in snapshot order."""
    return [
        (name, fid)
        for name, status, fid in _snapshot_files(corpus_context)
        if status == "ready" and fid
    ]


def get_corpus_state(corpus_context: str | None) -> str:
    """Tri-state the corpus file snapshot for empty-corpus guards.

    Returns one of:
    - "unknown": snapshot is None (DB failure) — caller should still try
      rag.query; the database is ground truth, not the snapshot.
    - "ready": at least one ready file — document-grounded retrieval path.
    - "processing": files exist but none is ready yet (uploading /
      processing / error) — nothing retrievable right now.
    - "empty": known to hold zero files — retrieval would provably return
      "(no chunks retrieved)".
    """
    if corpus_context is None:
        return "unknown"
    files = _snapshot_files(corpus_context)
    if not files:
        return "empty"
    if any(status == "ready" and fid for _, status, fid in files):
        return "ready"
    return "processing"
