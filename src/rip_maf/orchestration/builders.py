"""L2 deterministic builders — code-built DAGs, no DAG-LLM.

For intents in DETERMINISTIC_INTENTS the plan shape is fixed; only slot
values (queries, request text) vary. Wiring (depends_on + {{id}}
placeholders) is set by construction, so the ecd93eb4 failure class
(prose mention of steps without placeholders) cannot occur.

Deliberately NOT built: the two REACT_ONLY intents (summarize_plot,
plot_standalone) — a chart needs model-derived `labels`/`series` names no
deterministic shape can know (inventing them would be the hallucinated-chart
class ADR-027 exists to prevent), so they go to L3 ReAct. Convert builders
resolve literal file ids from the corpus snapshot; anything unresolvable
returns None → L3 ReAct.

Bucket split (ADR-035): DOC intents emit rag.query/doc.* only after the
corpus tri-state proves retrieval can work; NON_DOC intents never emit
rag.query at all.
"""

from __future__ import annotations

import uuid

from rip_maf.orchestration.corpus import (
    _ready_files,
    _snapshot_files,
    get_corpus_state,
)
from rip_maf.orchestration.intents import Intent
from rip_maf.orchestration.plan import Plan, PlanStep
from rip_maf.orchestration.router import RouterResult

_PER_FILE_TOP_K = 4

#: Parallel file-scoped shard cap (ADR-030): more than this contends the
#: single Ollama server and the step budget for no retrieval gain.
_MAX_SHARDS = 5

#: doc.generate title is a literal (no placeholder), so it must be derived
#: from the request itself — keep it short enough to stay a heading.
_MAX_TITLE_CHARS = 90


def _corpus_state(corpus_context: str | None) -> str:
    """Backwards-compat wrapper over `corpus.get_corpus_state`.

    Kept so existing importers (`react.py` legacy, tests) keep working;
    new code should import from `app.orchestration.corpus` directly.
    """
    return get_corpus_state(corpus_context)


def build_knowledge_qa(request_text: str) -> Plan:
    """Answer from general knowledge — one reasoning step, request verbatim.

    The NON_DOC counterpart of qa_single: no rag.query, so nothing can
    return "(no chunks retrieved)" and force the grounded prompt to say
    "not in the documents". Also the landing shape for a qa_single routed
    onto an empty corpus (see `build_qa_no_docs`).
    """
    return Plan(
        plan_id=str(uuid.uuid4()),
        goal=request_text,
        steps=[
            PlanStep(
                step_id="1",
                agent_id="reasoning",
                input={"message": request_text},
                expected_output_type="answer",
            )
        ],
    )


def build_qa_no_docs(request_text: str) -> Plan:
    """Answer a factual question with no retrievable documents.

    Empty-corpus path: emitting rag.query would provably return
    "(no chunks retrieved)" and force the grounded prompt to answer
    "not in the documents" — useless for general-knowledge questions
    like "What is QLoRA?". Answer from general knowledge instead
    (verbatim request, no document-grounding wrapper).
    """
    return build_knowledge_qa(request_text)


def build_code(request_text: str) -> Plan:
    """One `coding` agent step — code never needs the corpus."""
    return Plan(
        plan_id=str(uuid.uuid4()),
        goal=request_text,
        steps=[
            PlanStep(
                step_id="1",
                agent_id="coding",
                input={"message": request_text},
                expected_output_type="answer",
            )
        ],
    )


def build_no_docs_clarification(request_text: str, *, processing: bool = False) -> Plan:
    """Ask the user to upload/wait — nothing exists to summarize/compare.

    expected_output_type="clarification" so the single terminal step is
    returned verbatim as the answer (no retrieval to ground anything else).
    """
    if processing:
        detail = (
            "The corpus's files are not ready yet (still uploading, "
            "processing, or errored). Ask the user to wait until "
            "processing finishes and then retry"
        )
    else:
        detail = (
            "There are no ready documents in this corpus. Ask the user "
            "to upload documents or clarify how to proceed without them"
        )
    return Plan(
        plan_id=str(uuid.uuid4()),
        goal=request_text,
        steps=[
            PlanStep(
                step_id="1",
                agent_id="reasoning",
                input={"message": f"{detail}. Request: {request_text}"},
                expected_output_type="clarification",
            )
        ],
    )


def build_chat(request_text: str) -> Plan:
    return Plan(
        plan_id=str(uuid.uuid4()),
        goal=request_text,
        steps=[
            PlanStep(
                step_id="1",
                agent_id="reasoning",
                input={"message": request_text},
                expected_output_type="text",
            )
        ],
    )


def build_qa_single(query: str, request_text: str) -> Plan:
    return Plan(
        plan_id=str(uuid.uuid4()),
        goal=request_text,
        steps=[
            PlanStep(
                step_id="1",
                tool_id="rag.query",
                input={
                    "query": query or request_text,
                    "top_k": _PER_FILE_TOP_K,
                    "mode": "specific",
                },
                expected_output_type="chunks",
            ),
            PlanStep(
                step_id="2",
                agent_id="reasoning",
                input={
                    "message": (
                        f"Answer the user's request using ONLY these retrieved "
                        f"chunks {{{{1}}}}. Say 'not in the documents' when the "
                        f"chunks are empty or read '(no chunks retrieved)'. "
                        f"Never mention chunk ids or placeholders. "
                        f"Request: {request_text}"
                    )
                },
                depends_on=["1"],
                expected_output_type="answer",
            ),
        ],
    )


def build_compare_multi(
    queries: list[str],
    request_text: str,
    corpus_context: str | None = None,
) -> Plan:
    """Per-file specific shards fanning into one reasoning step.

    File-aware (not query-angle) fan-out: one file-scoped rag.query
    (mode=specific, top_k=4) per ready snapshot file. Topical query uses
    the request text so complexity/signal sections rank, not intro
    keywords. Falls back to legacy query-angle steps when no snapshot.
    Grounding is structural: every rag step id appears in both depends_on
    and as a {{id}} placeholder in the reasoning message.
    """
    ready = _ready_files(corpus_context)[:_MAX_SHARDS]
    steps: list[PlanStep] = []
    if ready:
        for i, (_name, fid) in enumerate(ready, start=1):
            steps.append(
                PlanStep(
                    step_id=str(i),
                    tool_id="rag.query",
                    input={
                        "query": request_text,
                        "top_k": _PER_FILE_TOP_K,
                        "file_id": fid,
                        "mode": "specific",
                    },
                    expected_output_type="chunks",
                )
            )
        # Single ready file still yields one shard + reduce (honest
        # single-doc answer rather than a padded duplicate query).
        if len(steps) == 1 and len(_snapshot_files(corpus_context)) <= 1:
            pass
    else:
        sources = [q for q in (queries or []) if q.strip()]
        while len(sources) < 2:
            sources.append(request_text)
        sources = sources[:_MAX_SHARDS]  # parallelism budget: ≤5 siblings
        for i, q in enumerate(sources, start=1):
            steps.append(
                PlanStep(
                    step_id=str(i),
                    tool_id="rag.query",
                    input={"query": q, "top_k": _PER_FILE_TOP_K, "mode": "specific"},
                    expected_output_type="chunks",
                )
            )
    dep_ids = [s.step_id for s in steps]
    refs = " ".join(f"{{{{{sid}}}}}" for sid in dep_ids)
    steps.append(
        PlanStep(
            step_id=str(len(steps) + 1),
            agent_id="reasoning",
            input={
                "message": (
                    f"Using ONLY these retrieved chunks ({refs}), address the "
                    f"request. Say 'not in the documents' for anything the "
                    f"chunks do not cover, including when they read "
                    f"'(no chunks retrieved)'. Never mention chunk ids or "
                    f"placeholders. Request: {request_text}"
                )
            },
            depends_on=dep_ids,
            expected_output_type="answer",
        )
    )
    return Plan(plan_id=str(uuid.uuid4()), goal=request_text, steps=steps)


def build_summarize(request_text: str, corpus_context: str | None = None) -> Plan:
    """Per-file overview shards fanning into one reduce step.

    Each shard uses mode=overview (stratified one-per-H1 sample, top_k=4)
    so every file contributes its overall idea. Single reduce step keeps
    the Ollama budget flat.
    """
    ready = _ready_files(corpus_context)[:_MAX_SHARDS]
    steps: list[PlanStep] = []
    if ready:
        for i, (_name, fid) in enumerate(ready, start=1):
            steps.append(
                PlanStep(
                    step_id=str(i),
                    tool_id="rag.query",
                    input={
                        "query": request_text,
                        "top_k": _PER_FILE_TOP_K,
                        "file_id": fid,
                        "mode": "overview",
                    },
                    expected_output_type="chunks",
                )
            )
    else:
        steps.append(
            PlanStep(
                step_id="1",
                tool_id="rag.query",
                input={
                    "query": request_text,
                    "top_k": _PER_FILE_TOP_K,
                    "mode": "overview",
                },
                expected_output_type="chunks",
            )
        )
    dep_ids = [s.step_id for s in steps]
    refs = " ".join(f"{{{{{sid}}}}}" for sid in dep_ids)
    steps.append(
        PlanStep(
            step_id=str(len(steps) + 1),
            agent_id="reasoning",
            input={
                "message": (
                    f"Using ONLY these retrieved chunks ({refs}), write the "
                    f"requested summary. Say 'not in the documents' when the "
                    f"chunks are empty or read '(no chunks retrieved)'. "
                    f"Never mention chunk ids or placeholders. "
                    f"Request: {request_text}"
                )
            },
            depends_on=dep_ids,
            expected_output_type="answer",
        )
    )
    return Plan(plan_id=str(uuid.uuid4()), goal=request_text, steps=steps)


def build_quiz(
    query: str,
    request_text: str,
    corpus_context: str | None = None,
) -> Plan:
    """Per-file overview shards, then ONE writer step.

    Single writer (never fan-out) per planner Rule 9: splitting by
    difficulty costs more and contends the single Ollama server.
    Overview mode gives breadth for question coverage.
    """
    ready = _ready_files(corpus_context)[:_MAX_SHARDS]
    steps: list[PlanStep] = []
    if ready and len(ready) > 1:
        for i, (_name, fid) in enumerate(ready, start=1):
            steps.append(
                PlanStep(
                    step_id=str(i),
                    tool_id="rag.query",
                    input={
                        "query": query or request_text,
                        "top_k": _PER_FILE_TOP_K,
                        "file_id": fid,
                        "mode": "overview",
                    },
                    expected_output_type="chunks",
                )
            )
        dep_ids = [s.step_id for s in steps]
        refs = " ".join(f"{{{{{sid}}}}}" for sid in dep_ids)
        steps.append(
            PlanStep(
                step_id=str(len(steps) + 1),
                agent_id="reasoning",
                input={
                    "message": (
                        f"Using ONLY these retrieved chunks ({refs}), "
                        f"write the requested questions/quiz. Say 'not in "
                        f"the documents' when the chunks are empty or read "
                        f"'(no chunks retrieved)'. Never mention chunk ids "
                        f"or placeholders. "
                        f"Request: {request_text}"
                    )
                },
                depends_on=dep_ids,
                expected_output_type="answer",
            )
        )
        return Plan(plan_id=str(uuid.uuid4()), goal=request_text, steps=steps)
    return Plan(
        plan_id=str(uuid.uuid4()),
        goal=request_text,
        steps=[
            PlanStep(
                step_id="1",
                tool_id="rag.query",
                input={
                    "query": query or request_text,
                    "top_k": _PER_FILE_TOP_K,
                    "mode": "overview",
                },
                expected_output_type="chunks",
            ),
            PlanStep(
                step_id="2",
                agent_id="reasoning",
                input={
                    "message": (
                        f"Using ONLY these retrieved chunks {{{{1}}}}, "
                        f"write the requested questions/quiz. Say 'not in "
                        f"the documents' when the chunks are empty or read "
                        f"'(no chunks retrieved)'. Never mention chunk ids "
                        f"or placeholders. "
                        f"Request: {request_text}"
                    )
                },
                depends_on=["1"],
                expected_output_type="answer",
            ),
        ],
    )


def build_convert_all(target_format: str, request_text: str) -> Plan:
    return Plan(
        plan_id=str(uuid.uuid4()),
        goal=request_text,
        steps=[
            PlanStep(
                step_id="1",
                tool_id="doc.convert",
                input={"file_id": "*", "target_format": target_format},
                expected_output_type="document",
            )
        ],
    )


def build_convert_ambiguous(request_text: str, corpus_context: str | None) -> Plan:
    """Ask the counter-question instead of guessing a conversion.

    "convert it to docx" names a format but no file; "convert this
    document" names neither. Guessing either produces a wrong artifact,
    so one `clarification` step states exactly what is missing. Costs a
    single reasoning call where ReAct previously burned up to six
    iterations to reach the same question.
    """
    snapshot = _snapshot_files(corpus_context)
    ready = [n for n, s, _f in snapshot if s == "ready"]
    if len(ready) == 1:
        known = f"The corpus has one ready file ({ready[0]})."
    elif ready:
        known = (
            f"The corpus has {len(ready)} ready files: {', '.join(ready[:5])}."
        )
    elif snapshot:
        known = (
            "The corpus has files but none is ready yet (still uploading or "
            "processing) — there is nothing to convert right now."
        )
    else:
        known = "The corpus's file inventory is unavailable."
    return Plan(
        plan_id=str(uuid.uuid4()),
        goal=request_text,
        steps=[
            PlanStep(
                step_id="1",
                agent_id="reasoning",
                input={
                    "message": (
                        "The user asked to convert a document but did not say "
                        "which file and/or which target format (md, docx or "
                        "pdf). Ask the counter-question in one short reply — "
                        "name the available files and the three formats, and "
                        "do not convert anything yet. "
                        f"{known} Request: {request_text}"
                    )
                },
                expected_output_type="clarification",
            )
        ],
    )


def _report_title(request_text: str) -> str:
    """Literal doc.generate title derived from the request (no placeholder).

    doc.generate's `title` must be present at plan time (validator rejects
    a missing required field), so it is derived from the request text
    rather than from an upstream step.
    """
    flat = " ".join((request_text or "").split()).strip(" .:-")
    return (flat[:_MAX_TITLE_CHARS] or "Report")


def build_report(request_text: str, corpus_context: str | None = None) -> Plan:
    """Per-file overview shards → one writer step → one doc.generate.

    The writer step is what makes the report grounded (ADR-027): it
    carries every shard's chunks as {{id}} placeholders, and doc.generate
    consumes that writer's prose, so the validator's report-grounding rule
    is satisfied structurally rather than by prompt discipline. doc.generate
    renders md/docx/pdf and the file travels via the artifacts event.
    """
    ready = _ready_files(corpus_context)[:_MAX_SHARDS]
    steps: list[PlanStep] = []
    if ready:
        for i, (_name, fid) in enumerate(ready, start=1):
            steps.append(
                PlanStep(
                    step_id=str(i),
                    tool_id="rag.query",
                    input={
                        "query": request_text,
                        "top_k": _PER_FILE_TOP_K,
                        "file_id": fid,
                        "mode": "overview",
                    },
                    expected_output_type="chunks",
                )
            )
    else:
        steps.append(
            PlanStep(
                step_id="1",
                tool_id="rag.query",
                input={
                    "query": request_text,
                    "top_k": _PER_FILE_TOP_K,
                    "mode": "overview",
                },
                expected_output_type="chunks",
            )
        )
    dep_ids = [s.step_id for s in steps]
    refs = " ".join(f"{{{{{sid}}}}}" for sid in dep_ids)
    writer_id = str(len(steps) + 1)
    # Built by concatenation, not an f-string: "{{{{id}}}}" inside an
    # f-string literal is a brace-parsing trap.
    writer_ref = "{{" + writer_id + "}}"
    steps.append(
        PlanStep(
            step_id=writer_id,
            agent_id="reasoning",
            input={
                "message": (
                    f"Using ONLY these retrieved chunks ({refs}), write the "
                    f"requested report as self-contained Markdown prose with "
                    f"short section headings. Say 'not in the documents' for "
                    f"anything the chunks do not cover, including when they "
                    f"read '(no chunks retrieved)'. Never mention chunk ids or "
                    f"placeholders. Request: {request_text}"
                )
            },
            depends_on=dep_ids,
            expected_output_type="answer",
        )
    )
    steps.append(
        PlanStep(
            step_id=str(len(steps) + 1),
            tool_id="doc.generate",
            input={
                "title": _report_title(request_text),
                "sections": [{"heading": "Report", "body": writer_ref}],
            },
            depends_on=[writer_id],
            expected_output_type="document",
        )
    )
    return Plan(plan_id=str(uuid.uuid4()), goal=request_text, steps=steps)


def build_convert_one(file_id: str, target_format: str, request_text: str) -> Plan:
    return Plan(
        plan_id=str(uuid.uuid4()),
        goal=request_text,
        steps=[
            PlanStep(
                step_id="1",
                tool_id="doc.convert",
                input={"file_id": file_id, "target_format": target_format},
                expected_output_type="document",
            )
        ],
    )


def _resolve_convert_file_id(
    file_hint: str, corpus_context: str | None
) -> str | None:
    """Resolve a router file_hint to a literal snapshot file_id.

    Returns None when unresolvable (no hint, no snapshot, no/ambiguous
    match, or match not ready) — caller falls through to L3 ReAct, which
    asks the counter-question. Never invents an id.
    """
    hint = (file_hint or "").strip()
    if not hint or hint == "*":
        return None
    candidates = [
        (name, fid)
        for name, status, fid in _snapshot_files(corpus_context)
        if status == "ready" and hint.lower() in name.lower()
    ]
    if len(candidates) != 1:
        return None
    return candidates[0][1]


def build(
    request_text: str,
    route: RouterResult,
    corpus_context: str | None = None,
) -> Plan | None:
    """Dispatch router result to a deterministic builder.

    DOC intents consult the corpus tri-state first — retrieval on an empty
    or still-processing corpus provably returns nothing. NON_DOC intents
    never reach that code: they cannot emit rag.query at all (ADR-035).

    Returns None for intents without a fixed shape — caller falls through
    to L3 ReAct.
    """
    # -- NON_DOC: never touch the corpus ---------------------------------
    if route.intent is Intent.CHAT:
        return build_chat(request_text)
    if route.intent is Intent.KNOWLEDGE_QA:
        return build_knowledge_qa(request_text)
    if route.intent is Intent.CODE:
        return build_code(request_text)
    # -- DOC: retrieval or byte-level file ops ------------------------------
    if route.intent is Intent.QA_SINGLE:
        state = _corpus_state(corpus_context)
        if state == "empty":
            # Routed as a document question, but there are no documents:
            # answer generally rather than claim "not in the documents".
            return build_knowledge_qa(request_text)
        if state == "processing":
            return build_no_docs_clarification(request_text, processing=True)
        return build_qa_single(request_text, request_text)
    if route.intent is Intent.COMPARE_MULTI:
        state = _corpus_state(corpus_context)
        if state in ("empty", "processing"):
            return build_no_docs_clarification(
                request_text, processing=(state == "processing")
            )
        ready = _ready_files(corpus_context)
        if ready and len(ready) > _MAX_SHARDS:
            return None  # too many files: fall through to L3 ReAct
        return build_compare_multi([], request_text, corpus_context)
    if route.intent is Intent.SUMMARIZE:
        state = _corpus_state(corpus_context)
        if state in ("empty", "processing"):
            return build_no_docs_clarification(
                request_text, processing=(state == "processing")
            )
        ready = _ready_files(corpus_context)
        if ready and len(ready) > _MAX_SHARDS:
            return None
        return build_summarize(request_text, corpus_context)
    if route.intent is Intent.QUIZ:
        state = _corpus_state(corpus_context)
        if state in ("empty", "processing"):
            return build_no_docs_clarification(
                request_text, processing=(state == "processing")
            )
        ready = _ready_files(corpus_context)
        if ready and len(ready) > _MAX_SHARDS:
            return None
        return build_quiz(request_text, request_text, corpus_context)
    if route.intent is Intent.REPORT:
        state = _corpus_state(corpus_context)
        if state in ("empty", "processing"):
            return build_no_docs_clarification(
                request_text, processing=(state == "processing")
            )
        ready = _ready_files(corpus_context)
        if ready and len(ready) > _MAX_SHARDS:
            return None
        return build_report(request_text, corpus_context)
    if route.intent is Intent.CONVERT_AMBIGUOUS:
        # No file and/or no format named: one counter-question, no guess.
        return build_convert_ambiguous(request_text, corpus_context)
    if route.intent is Intent.CONVERT_ALL:
        # Unlike convert_one (unresolvable hints fall through to the
        # counter-question), "*" with zero ready files builds a doomed
        # doc.convert that fails after retries — clarify instead.
        state = _corpus_state(corpus_context)
        if state in ("empty", "processing"):
            return build_no_docs_clarification(
                request_text, processing=(state == "processing")
            )
        if not route.target_format:
            return build_convert_ambiguous(request_text, corpus_context)
        return build_convert_all(route.target_format, request_text)
    if route.intent is Intent.CONVERT_ONE:
        # No corpus guard here on purpose: an unresolvable hint (empty
        # corpus, unready match, ambiguous match) already falls through
        # to the counter-question below, which names the ready files — a
        # better message than the generic upload/wait clarification.
        if not route.target_format:
            return build_convert_ambiguous(request_text, corpus_context)
        file_id = _resolve_convert_file_id(route.file_hint, corpus_context)
        if file_id is None:
            # Missing / ambiguous / not-ready match: ask rather than convert
            # the wrong file (the counter-question names the ready files).
            return build_convert_ambiguous(request_text, corpus_context)
        return build_convert_one(file_id, route.target_format, request_text)
    return None
