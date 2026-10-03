"""Observability for RIP: optional Langfuse tracing (Athena port).

Everything here is a no-op when tracing is disabled
(`langfuse_enabled=False`), so enabling/disabling tracing never changes run
behavior. When enabled, the run worker opens one trace per run (session =
corpus, mirroring Athena's session = conversation), the engine emits
router (sibling of plan) + plan/aggregate spans, plan_graph emits per-step
spans nested INSIDE the plan span (explicit trace_context parenting via
graph state/config, not contextvars alone), the ReAct fallback emits
react → react:iter-N spans with step:rN nested under their iteration,
and the TracingProvider tags every LLM call as a
generation — all nested under the run trace, even across LangGraph pool
threads and the plan graph's worker threads.
"""
