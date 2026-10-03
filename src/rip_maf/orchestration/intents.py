"""Intent taxonomy for layered planning.

The taxonomy is split into two buckets so the doc/non-doc line is data,
not prose:

- DOC_INTENTS: the answer lives in the corpus's uploaded documents
  (retrieval via rag.query, or byte-level file ops via doc.convert /
  doc.generate).
- NON_DOC_INTENTS: the answer lives in the model's own knowledge or in
  pure computation — these must never emit rag.query.

L1 router classifies into these intents; L2 builders handle every intent
except REACT_ONLY_INTENTS (and UNKNOWN / low confidence), which go to
L3 ReAct.

Pure data. NO LLM here.
"""

from __future__ import annotations

from enum import Enum


class Intent(str, Enum):
    # -- doc-based: answered from the corpus's documents ---------------
    QA_SINGLE = "qa_single"
    COMPARE_MULTI = "compare_multi"
    SUMMARIZE = "summarize"
    SUMMARIZE_PLOT = "summarize_plot"
    REPORT = "report"
    CONVERT_ONE = "convert_one"
    CONVERT_ALL = "convert_all"
    CONVERT_AMBIGUOUS = "convert_ambiguous"
    QUIZ = "quiz"
    # -- non-doc: answered from general knowledge / computation -----------
    CHAT = "chat"
    KNOWLEDGE_QA = "knowledge_qa"
    PLOT_STANDALONE = "plot_standalone"
    CODE = "code"
    UNKNOWN = "unknown"


#: Router confidence below this routes to UNKNOWN (→ L3 ReAct).
ROUTER_CONFIDENCE_THRESHOLD = 0.6

#: One line per intent for the tiny router prompt (kept here so prompts stay small).
INTENT_DESCRIPTIONS: dict[Intent, str] = {
    Intent.CHAT: "greeting, thanks, or small talk",
    Intent.KNOWLEDGE_QA: "a general question answerable from your own knowledge, NOT from the corpus",
    Intent.CODE: "write, explain, review, or debug code",
    Intent.PLOT_STANDALONE: "draw a chart from numbers given in the message or recalled from your own knowledge (never from the corpus)",
    Intent.QA_SINGLE: "one factual question answered from the corpus's documents",
    Intent.COMPARE_MULTI: "compare, contrast, or rank two or more documents/topics (no chart requested)",
    Intent.SUMMARIZE: "summarize documents without chart or report file",
    Intent.SUMMARIZE_PLOT: "summarize/compare the corpus's documents AND draw/plot/chart the numbers — any plot/draw/chart/show-as-graph ask about document data belongs here, even when the request also says compare",
    Intent.REPORT: "write the corpus's answer as a titled report file (doc.generate)",
    Intent.CONVERT_ONE: "convert one named corpus file to md/docx/pdf",
    Intent.CONVERT_ALL: "convert all/plural corpus documents to md/docx/pdf",
    Intent.CONVERT_AMBIGUOUS: "convert it/the document without naming file or format",
    Intent.QUIZ: "generate questions, quiz, or MCQs from the corpus's documents",
    Intent.UNKNOWN: "anything else or unclear",
}

#: Intents whose answer must come from the corpus's documents (RAG or
#: byte-level file tools). Anything here is pointless on an empty corpus.
DOC_INTENTS = frozenset(
    {
        Intent.QA_SINGLE,
        Intent.COMPARE_MULTI,
        Intent.SUMMARIZE,
        Intent.SUMMARIZE_PLOT,
        Intent.REPORT,
        Intent.CONVERT_ONE,
        Intent.CONVERT_ALL,
        Intent.CONVERT_AMBIGUOUS,
        Intent.QUIZ,
    }
)

#: Intents answered from general knowledge or computation. These must
#: never emit rag.query — retrieval would only burn iterations on a
#: provably irrelevant corpus (trace c9b02eef).
NON_DOC_INTENTS = frozenset(
    {
        Intent.CHAT,
        Intent.KNOWLEDGE_QA,
        Intent.PLOT_STANDALONE,
        Intent.CODE,
    }
)

#: Intents with no fixed DAG shape, served by L3 ReAct. Both plot intents
#: need model-derived chart `labels`/`series` names that no deterministic
#: shape can wire by construction (placeholders carry whole text, not
#: split-able label lists) — inventing them would be the hallucinated-chart
#: class ADR-027 prevents.
REACT_ONLY_INTENTS = frozenset({Intent.SUMMARIZE_PLOT, Intent.PLOT_STANDALONE})

#: Intents served by deterministic code-built DAGs (no DAG-LLM needed).
#: Derived from the buckets above so the taxonomy cannot drift: a new
#: intent is deterministic by default and must be added to
#: REACT_ONLY_INTENTS explicitly to earn a ReAct hop.
DETERMINISTIC_INTENTS = (DOC_INTENTS | NON_DOC_INTENTS) - REACT_ONLY_INTENTS
