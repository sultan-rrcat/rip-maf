# ADR — Architecture Decisions (rip-maf)

Backend-only agentic-RAG service. Open WebUI is the frontend, chat history,
and identity; this service owns retrieval, tools, and orchestration.

## ADR-001: Backend only — Open WebUI is the frontend

- **Status:** Accepted
- **Context:** The previous project (RIP) had a bespoke React frontend, cookie
  auth, and notebook/conversation management. Maintaining a second chat UI
  duplicates what Open WebUI does better (auth, history, sharing, file preview,
  community plugins).
- **Decision:** rip-maf serves a single client: the Open WebUI Pipe
  (`integrations/openwebui/rip_pipe.py`), exposed as a selectable model. There
  is no browser-facing UI, no cookies, no CORS. A companion Filter
  (`rip_filter.py`) suppresses Open WebUI's own RAG for the RIP model so this
  backend is the only retriever.
- **Consequences:** The Pipe owns the full request/response mapping (SSE →
  deltas/sources/artifacts). All other API evolution must keep the Pipe
  contract stable.

## ADR-002: Corpus replaces notebook as the scope unit

- **Status:** Accepted
- **Context:** Retrieval, run persistence, and tool scoping need a unit, but the
  notebook (document collection *plus* conversation) is gone: Open WebUI owns
  conversations.
- **Decision:** The unit is the **corpus**: one per `(owui_user, owui_chat)`,
  stored in `corpora(corpus_id, corpus_ref, owner_ref)` with
  `UNIQUE(owner_ref, corpus_ref)`. Renamed end-to-end (`notebook_id` →
  `corpus_id`, `notebook.inspect` → `corpus.inspect`); tables
  `users/sessions/messages/notebooks` were not ported.
- **Consequences:** Uploads in a chat are scoped to that chat and owner. Chat
  deletion does not reap corpora yet (future work).

## ADR-003: Service-key-only auth, no local users

- **Status:** Accepted
- **Context:** The Pipe runs server-side inside Open WebUI and already knows
  the OWUI identity. Local accounts/sessions add nothing.
- **Decision:** Every request needs `X-RIP-Service-Key` (matches
  `RIP_SERVICE_KEY`) plus `X-RIP-User-Id` and `X-RIP-User-Name`, resolved by
  `auth.py::require_principal` into a lightweight `Principal{owner_ref, name}`.
  There is no login, no cookie, no `users` table. Trust boundary = internal
  network + shared secret (ports are loopback-bound in compose).
- **Consequences:** `owner_ref`-scoped 404s (never 403 leaks). Key rotation =
  change both sides.

## ADR-004: Stateless over conversations

- **Status:** Accepted
- **Context:** Two histories (Open WebUI's chat + a local rolling summary)
  would drift and duplicate writes.
- **Decision:** The Pipe forwards prior turns on every run
  (`history: [{role, content}]`); the worker renders only the newest window
  verbatim (`orchestration/memory.py::build_history_context`). No rolling
  summary, no `messages` table, no summary-fold LLM call. Run/analysis of past
  turns is the Pipe's job.
- **Consequences:** Long conversations depend on Open WebUI's context
  management (context compaction). The `OLLAMA_CONTEXT_WINDOW` budget must be
  sized accordingly.

## ADR-005: Keep the full orchestrator

- **Status:** Accepted
- **Context:** "RAG and tools" could have meant dropping the engine and letting
  Open WebUI's native agentic loop drive the tools. The engine — L1 router →
  L2 deterministic plan DAG → L3 ReAct fallback, validators, deterministic
  aggregation — is the project's actual value over Open WebUI's single tool
  loop.
- **Decision:** Port the orchestrator, builders, validator, plan graph,
  aggregator, the two agents (`reasoning`, `coding`), and five tools
  (`rag.query`, `corpus.inspect`, `plot.chart`, `doc.generate`,
  `doc.convert`) intact, renamed to corpus scope. Dropped: `image.generate`
  (no image model configured), `code.sandbox` (needed host Docker), `vision`.
- **Consequences:** The router/builders/ReAct contract and the SSE vocabulary
  are kept wire-stable for the Pipe. Tool-executor failure semantics are
  fail-honest (no approval gate).

## ADR-006: No health endpoint

- **Status:** Accepted
- **Context:** No browser client polls this service, and the previous `/health`
  + `/v1/admin/health` existed for a frontend/compose-gating pair that no
  longer exists.
- **Decision:** No `/health`, no admin stub. Liveness = container state +
  log output. Postgres keeps its own `pg_isready` healthcheck for startup
  ordering (`depends_on: service_healthy`).
- **Consequences:** `GET /v1/runs/{id}` is the functional health signal.
  Monitoring must look at run success/errors, not a status route.
