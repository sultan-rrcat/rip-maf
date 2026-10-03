"""L1 intent router — one tiny structured LLM call, the sole dispatcher.

Classifies the user request into exactly one intent plus slot values
(file_hint, target_format). Returns Intent + confidence; callers
route < threshold to UNKNOWN (→ L3 ReAct). Every request — including
greetings — goes through the LLM; there is no deterministic fast-path.
Query generation is performed inside rag.query, not by the router.

The prompt states the corpus's corpus state and splits the menu into
DOC-BASED vs NON-DOC intents (ADR-035). Without the corpus the router
cannot tell "answer from the documents" from "answer from your own
knowledge", so a parametric request ("Plot India vs China GDP growth")
routed as a document intent and paid a retrieval round trip for a
guaranteed "(no chunks retrieved)".
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel

from rip_maf.core.config import settings
from rip_maf.orchestration.corpus import _snapshot_files, get_corpus_state
from rip_maf.orchestration.intents import (
    DOC_INTENTS,
    INTENT_DESCRIPTIONS,
    NON_DOC_INTENTS,
    ROUTER_CONFIDENCE_THRESHOLD,
    Intent,
)
from rip_maf.providers.base import ModelProvider

logger = logging.getLogger("orchestration.router")

ROUTER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "confidence": {"type": "number"},
        "file_hint": {"type": "string"},
        "target_format": {"type": "string"},
    },
    "required": ["intent", "confidence"],
}

#: Formats doc.convert accepts; anything else means "format unstated".
_CONVERT_FORMATS = frozenset({"md", "docx", "pdf"})

#: "all/every ... documents/files" → convert-all file_hint.
_ALL_DOCS_RE = re.compile(
    r"\b(all|every|each)\b.{0,20}\b(documents?|files?|docs?)\b", re.IGNORECASE
)

#: Format words in free text ("markdown" reads as md). Matched only after
#: snapshot filenames are masked out, so "Report.pdf" never reads as pdf.
_FORMAT_WORD_RE = re.compile(r"\b(docx|pdf|markdown|md)\b", re.IGNORECASE)
_FORMAT_WORD_MAP = {"markdown": "md", "md": "md", "docx": "docx", "pdf": "pdf"}


def _recover_convert_slots(
    request_text: str,
    intent: Intent,
    file_hint: str,
    target_format: str,
    corpus_context: str | None,
) -> tuple[str, str]:
    """Fill convert slots the LLM left empty, deterministically.

    Small models reliably return the intent with blank slots (live probe:
    granite-3B filled neither slot on any convert phrasing), which demotes
    every conversion to a counter-question. The request text plus the
    snapshot already hold both answers: a named snapshot file (or an
    all/every-documents phrase) and an md|docx|pdf word. The intent itself
    is never changed — only empty slots are filled.
    """
    if intent not in (
        Intent.CONVERT_ONE, Intent.CONVERT_ALL, Intent.CONVERT_AMBIGUOUS,
    ):
        return file_hint, target_format
    if file_hint and target_format:
        return file_hint, target_format
    text = request_text or ""
    lowered = text.lower()
    names = [
        name for name, _status, _fid in _snapshot_files(corpus_context)
    ]
    hint = file_hint
    if not hint:
        if _ALL_DOCS_RE.search(text):
            hint = "*"
        else:
            for name in names:
                if name.lower() in lowered:
                    hint = name
                    break
            else:
                for name in names:
                    stem = name.rsplit(".", 1)[0]
                    if stem and stem.lower() in lowered:
                        hint = name
                        break
    fmt = target_format
    if not fmt:
        masked = lowered
        for name in names:
            masked = masked.replace(name.lower(), " ")
        match = _FORMAT_WORD_RE.search(masked)
        if match:
            fmt = _FORMAT_WORD_MAP[match.group(1).lower()]
    return hint, fmt


class RouterResult(BaseModel):
    intent: Intent = Intent.UNKNOWN
    confidence: float = 0.0
    routed_by: str = "llm"
    # Convert slots: file_hint names one file (or "*" for all) and
    # target_format is md|docx|pdf. Empty slots are first recovered
    # deterministically from the request text + snapshot
    # (_recover_convert_slots); what stays unstated yields a
    # counter-question instead of a guessed conversion.
    file_hint: str = ""
    target_format: str = ""

    @property
    def is_doc_intent(self) -> bool:
        """Whether the routed intent claims the corpus's documents."""
        return self.intent in DOC_INTENTS


#: Corpus-state → one line the router must reason about. The doc/non-doc
#: split is only decidable with this in view, so it is prompt input, not
#: a post-hoc correction.
_CORPUS_LINES = {
    "ready": "the corpus HAS uploaded documents — DOC-BASED intents are answerable",
    "empty": (
        "the corpus is EMPTY (nothing uploaded) — every DOC-BASED intent can "
        "only answer 'upload documents first', so prefer a NON-DOC intent "
        "whenever the request is answerable from your own knowledge"
    ),
    "processing": (
        "the corpus's files are still uploading/processing — retrieval has "
        "nothing to search yet, so prefer a NON-DOC intent when possible"
    ),
    "unknown": (
        "the file inventory is unavailable — assume the corpus MAY have "
        "documents and classify on the wording alone"
    ),
}


def build_router_prompt(corpus_context: str | None = None) -> str:
    """Assemble the router system prompt (pure — no provider needed).

    Split from `Router.route` so the wording is unit-testable without a
    live Ollama, and so the corpus line has one home.
    """
    state = get_corpus_state(corpus_context)
    snapshot = corpus_context if corpus_context else "(inventory unavailable)"
    doc_lines = "\n".join(
        f"- {i.value}: {INTENT_DESCRIPTIONS[i]}" for i in Intent if i in DOC_INTENTS
    )
    non_doc_lines = "\n".join(
        f"- {i.value}: {INTENT_DESCRIPTIONS[i]}"
        for i in Intent
        if i in NON_DOC_INTENTS
    )
    return (
        "You are an intent router. Classify the user request into exactly "
        "one intent. Query generation for document retrieval is performed "
        "inside rag.query, not by you.\n"
        f"Corpus documents:\n{snapshot}\nCorpus: {_CORPUS_LINES[state]}\n"
        "DOC-BASED intents (answered from the corpus's documents/files):\n"
        f"{doc_lines}\n"
        "NON-DOC intents (answered from your own knowledge — never search "
        f"documents):\n{non_doc_lines}\n"
        f"- {Intent.UNKNOWN.value}: {INTENT_DESCRIPTIONS[Intent.UNKNOWN]}\n"
        "Rule 1 (decisive): if the request can be answered without the "
        "corpus, choose a NON-DOC intent. Choose a DOC-BASED intent only "
        "when the request refers to the corpus's uploaded documents or "
        "files.\n"
        "Rule 2 (charts): a chart from numbers in the message or from your own "
        f"knowledge is {Intent.PLOT_STANDALONE.value}; a chart of the "
        f"corpus's document data is {Intent.SUMMARIZE_PLOT.value}, even when "
        "the request also says compare or summarize.\n"
        "Convert intents only: file_hint is the named file (or \"*\" when "
        "the request says all/every documents, else \"\"), target_format "
        "is md|docx|pdf when stated (else \"\").\n"
        "Return intent as the exact value string and confidence as 0.0-1.0."
    )


class Router:
    def __init__(
        self,
        provider: ModelProvider,
        model: str | None = None,
    ):
        self._provider = provider
        self._model = model or settings.ollama_default_model

    def route(
        self,
        request_text: str,
        context: str | None = None,
        corpus_context: str | None = None,
    ) -> RouterResult:
        system_prompt = build_router_prompt(corpus_context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request_text},
        ]
        if context:
            # Follow-up fragments ("in a table format", "now as bullets") are
            # unclassifiable alone (trace 35e8fbd9: bare fragment -> unknown).
            # Recent turns let the router classify the combined intent.
            messages.insert(
                1,
                {"role": "system", "content": (
                    "Conversation context (recent turns, oldest first). The "
                    "request may be a follow-up to it — classify the combined "
                    "intent:\n" + context
                )},
            )
        try:
            raw = self._provider.generate_structured(
                model=self._model,
                messages=messages,
                schema=ROUTER_SCHEMA,
                temperature=0,
            )
        except Exception as e:  # noqa: BLE001 - fail-open to L3 ReAct
            logger.warning("router LLM failed, falling back to unknown: %s", e)
            return RouterResult(intent=Intent.UNKNOWN, routed_by="llm")
        try:
            intent = Intent(str(raw.get("intent", "unknown")).strip().lower())
        except ValueError:
            intent = Intent.UNKNOWN
        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))
        file_hint = raw.get("file_hint", "") or ""
        file_hint = file_hint.strip() if isinstance(file_hint, str) else ""
        target_format = raw.get("target_format", "") or ""
        target_format = (
            target_format.strip().lower() if isinstance(target_format, str) else ""
        )
        if intent in (
            Intent.CONVERT_ONE, Intent.CONVERT_ALL, Intent.CONVERT_AMBIGUOUS,
        ) and (not file_hint or not target_format):
            file_hint, target_format = _recover_convert_slots(
                request_text, intent, file_hint, target_format, corpus_context
            )
        if target_format not in _CONVERT_FORMATS:
            target_format = ""
        if confidence < ROUTER_CONFIDENCE_THRESHOLD:
            intent = Intent.UNKNOWN
        return RouterResult(
            intent=intent,
            confidence=confidence,
            file_hint=file_hint,
            target_format=target_format,
        )
