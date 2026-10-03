"""Idle-turn guard for the L3 ReAct loop.

The ReAct loop can spin on turns that produce no observation (provider
errors, unknown executors, malformed inputs, repeats of failed actions).
This module concentrates that guard in one place: each non-progress turn
calls `record_idle()`, each executed step or final answer calls
`record_progress()`. When consecutive idle turns reach the limit,
`record_idle()` returns True and the caller breaks out of the loop.

Previously this pattern was inlined at 10 guard sites inside `run_react`
(`idle_turns += 1; if idle_turns >= 2`), with the threshold hardcoded
everywhere and the reset in two places — a new guard had to remember
both halves.

`record_blocked()` is the separate budget for turns the ENVIRONMENT
refuses (rag.query on an empty corpus, corpus.inspect as a mandated
first step): those are not model malformations, so charging them to the
idle budget could end a run that had made real progress. Trace c9b02eef
killed a run at iteration 2 with zero executed steps that way — one
shape hint (idle) plus one empty-corpus refusal (idle) tripped the
threshold. The block budget is looser and never resets the idle counter.
"""

from __future__ import annotations

#: Consecutive non-progress turns before the loop fails fast instead of
#: burning the remaining iteration budget on identical errors.
MAX_IDLE_TURNS = 2

#: Environment-refused turns allowed before the loop gives up on the
#: blocked action entirely (the iteration budget still bounds the loop).
MAX_BLOCKED_TURNS = 3


class IdleGuard:
    """Counts consecutive idle turns; signals when the loop should break."""

    def __init__(
        self,
        max_idle_turns: int = MAX_IDLE_TURNS,
        max_blocked_turns: int = MAX_BLOCKED_TURNS,
    ):
        self._idle = 0
        self._max = max_idle_turns
        self._blocked = 0
        self._max_blocked = max_blocked_turns

    def record_idle(self) -> bool:
        """Record one non-progress turn. Returns True when the loop should break."""
        self._idle += 1
        return self._idle >= self._max

    def record_blocked(self) -> bool:
        """Record one environment-refused turn.

        Returns True only when the same kind of block has repeated past
        its budget. Never touches the idle counter: the model may have
        proposed a perfectly valid action that the corpus cannot serve.
        """
        self._blocked += 1
        return self._blocked >= self._max_blocked

    def record_progress(self) -> None:
        """Reset the counters — an executed step or final answer is progress."""
        self._idle = 0
        self._blocked = 0

    @property
    def idle_turns(self) -> int:
        """Current consecutive idle-turn count (for observability/tests)."""
        return self._idle

    @property
    def blocked_turns(self) -> int:
        """Current consecutive environment-refused turn count."""
        return self._blocked
