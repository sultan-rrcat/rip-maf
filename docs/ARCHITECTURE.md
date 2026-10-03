# RIP-MAF Architecture

Backend-only agentic-RAG service. Open WebUI is the UI, chat history, and
identity; this service owns retrieval, tools, and orchestration.

## Topology

```
┌────────────────────────── Open WebUI (separate) ─────────────────────────┐
│  Chat UI · authn · history · file preview   (rip_pipe.py + rip_filter.py) │
└────────────────────────────────┬──────────────────────────────────────────┘
      X-RIP-Service-Key + X-RIP-User-Id/Name (rip-maf-net / 127.0.0.1)
┌────────────────────────────────▼──────────────────────────────────────────┐
│                              rip-maf backend                              │
│  POST /v1/corpus  →  POST /v1/corpus/{id}/files  →  POST /v1/runs         │
│  GET /v1/runs/{id}/events (SSE) · cancel · artifacts                      │
│  L1 router → L2 deterministic plan DAG → L3 ReAct fallback                │
│  Agents: reasoning, coding   Tools: rag.query, corpus.inspect, plot.chart,│
│  doc.generate, doc.convert   Providers: Ollama (direct)                   │
└────────────────────────────────┬──────────────────────────────────────────┘
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
   PostgreSQL + pgvector    Ollama (host)        Langfuse (opt-in)
   corpora/files/embed-     qwen2.5:14b or       per-run traces
   dings/runs/run_events    override
```

## Data model (`schema.sql`, 5 tables)

| Table | Key columns |
|---|---|
| `corpora` | `corpus_id`, `name`, `corpus_ref` (Open WebUI chat id), `owner_ref` (Open WebUI user id), `UNIQUE(owner_ref, corpus_ref)` |
| `files` | `file_id`, `corpus_id`, `file_name`, `file_size`, `file_status` (`processing/ready/error`) |
| `embeddings` | `file_id`, `chunk_text`, `embedding vector(1024)` + HNSW, `chunk_index` (overview stratification), `metadata`, `text_search tsvector` + GIN |
| `runs` | `id`, `corpus_id`, `status` (`pending/running/completed/failed/cancelled`), `goal`, `plan`, `result` |
| `run_events` | `run_id`, `seq` (gap-free, structural only; `delta` live-only, never persisted), `event_type`, `payload` |

No `users`, `sessions`, `messages`, or notebook tables. Conversation history is
never stored: the Pipe forwards prior turns on each run and the worker renders
them verbatim (`orchestration/memory.py::build_history_context`, newest window).

## Request lifecycle

1. Pipe ensures the corpus: `POST /v1/corpus {chat_id}` → `{corpus_id}` (one
   corpus per `(owui_user, owui_chat)`, owner-scoped, idempotent).
2. Pipe uploads attached files: `POST /v1/corpus/{id}/files` (multipart) —
   runs the ingest pipeline **inline**: `services/ingest.py::run_rag_pipeline`
   (Docling/OpenDataLoader → header chunks → BGE-M3 embeddings → store).
   Returns when the file is `ready` (or `error`).
3. Pipe creates the run: `POST /v1/runs {corpus_id, message, history}` →
   `202 {run_id}` (404 unknown corpus, 422 empty message, 429 when 4 workers busy).
4. Worker (`runs/manager.py`, thread-per-run): renders history + file snapshot
   → `orchestrator.run(...)`.
5. **Router** (L1, sole dispatcher, one cheap `generate_structured` call) classifies
   intent + slots against the corpus state. **Builders** (L2) emit fixed DAGs for
   deterministic intents; builder misses run **ReAct** (L3, ≤6 iterations).
   **Engine** executes steps (corpus id injected server-side); **Aggregator**
   assembles the answer deterministically.
6. Worker emits `sources` per completed `rag.query`, `artifacts` for file outputs,
   `summary` (final answer), then `run_completed`. DB row is written before the
   terminal event; a late cancel never overwrites a final state.
7. Pipe consumes `GET /v1/runs/{id}/events` (SSE), maps frames to chat, fetches
   artifacts with the service key (charts inlined as data URIs), cancels on Stop.

SSE vocabulary (11 types, emit order): `run_started · plan · step_started ·
delta (live-only, fractional seq, never persisted) · step_completed · sources ·
artifacts · summary · run_completed · error · cancelled`.

## Retrieval (RAG)

Ingest (`services/ingest.py`): PDF→Markdown (Docling default, OpenDataLoader
fallback) → `MarkdownHeaderTextSplitter` (H1/H2/H3) → BGE-M3 embeddings →
`embeddings`. Re-ingest replaces the file's rows in one transaction.

Retrieve (`rag/vector_rag.py`): pgvector cosine + full-text rank → RRF (k=60)
→ BGE CrossEncoder rerank → top_k. `file_id`/`file_name` scope both paths;
`mode=overview` stratifies one chunk per H1. Whole-file early return: if the
whole scope fits `rag_whole_file_pct` (15%) of the window, all chunks return in
document order with `rerank_score=None`, skipping embed/retrieve/rerank.

## Module map (`src/rip_maf/`)

| Path | Role |
|---|---|
| `main.py` | FastAPI entrypoint: lifespan (VectorRAG + runtime composition), route mounts behind `require_principal` |
| `auth.py` | Service-key principal: `Principal{owner_ref, name}` from `X-RIP-Service-Key` + user headers; 401 otherwise |
| `core/config.py` | `Settings` (DB, Ollama, BGE paths, `rip_service_key`, sandbox-free) |
| `core/db.py`, `logging.py` | Threaded pg pool; console + file logging |
| `api/corpus.py` | Corpus ensure + file ingest |
| `api/runs.py` | Run lifecycle: create/get/cancel/events/artifacts |
| `api/deps.py` | Process-wide runtime composition (provider, registries, orchestrator, run manager) |
| `rag/` | Ingest pipeline + `VectorRAG` (singleton bound in lifespan) |
| `orchestration/` | L1 router → L2 builders → L3 ReAct → validator → plan graph → aggregator |
| `agents/` | `reasoning`, `coding` + registry |
| `tools/` | Contract (`base`, `schema`, `registry`, `executor`) + the five tools |
| `providers/` | `ModelProvider` contract + Ollama + `<think>` streaming filter + tracing |
| `runs/` | Thread-per-run worker + Postgres CRUD for runs/events |
| `envelope.py`, `artifacts.py` | SSE event vocabulary; tool outputs → disk + download refs |
| `services/`, `observability/` | RAG context helpers + ingest worker; opt-in Langfuse |

## Observability (opt-in Langfuse)

One trace per run worker (`session_id = corpus_id`, `trace_name = run`): router →
plan (+ step spans) → aggregate; ReAct iterations when used. Disabled path is
behavior-identical.

## Design decisions

See `docs/ADR.md`. The load-bearing ones: full orchestrator kept (the asset);
service-key-only auth with no local users; stateless conversation history;
`(owui_user, owui_chat)` corpus unit; rip-maf stays internal-only.
