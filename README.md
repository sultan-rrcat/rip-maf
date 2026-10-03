# RIP-MAF — Research Intelligence Platform, Multi-Agent Framework

Backend-only agentic-RAG service that powers the **RIP** model in Open WebUI.
No frontend, no notebooks: Open WebUI is the UI, chat history, and auth; this
service owns document RAG, the tool set, and the multi-agent orchestrator.

## What it is

- **RAG**: uploads are parsed (Docling), chunked, embedded (BGE-M3), stored in
  Postgres + pgvector, and retrieved with vector + full-text fusion + BGE rerank.
- **Orchestration**: L1 router → L2 deterministic plan DAG → L3 ReAct fallback,
  executed by a LangGraph plan graph and assembled deterministically.
- **Agents**: `reasoning`, `coding`.
- **Tools**: `rag.query`, `corpus.inspect`, `plot.chart`, `doc.generate`,
  `doc.convert`.
- **Stateless over conversations**: Open WebUI sends the history; this service
  stores none.
- **LLM**: Ollama only. **Offline**: no cloud dependencies.

## API (all require `X-RIP-Service-Key`)

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/corpus` | Ensure the corpus for an OWUI chat (`{chat_id, corpus_name?}`) |
| POST | `/v1/corpus/{corpus_id}/files` | Ingest one uploaded file (multipart) |
| POST | `/v1/runs` | Start a run (`{corpus_id, message, history}`) → `202 {run_id}` |
| GET | `/v1/runs/{id}` | Run detail |
| POST | `/v1/runs/{id}/cancel` | Cooperative cancel |
| GET | `/v1/runs/{id}/events` | SSE event stream |
| GET | `/v1/runs/{id}/artifacts/{aid}` | Download an artifact |

Headers: `X-RIP-Service-Key` (shared secret), `X-RIP-User-Id`,
`X-RIP-User-Name` (Open WebUI identity → corpus owner).

## Run

```powershell
copy .env.example .env    # set RIP_SERVICE_KEY, OLLAMA_BASE_URL, BGE paths
docker compose up -d --build
# point Open WebUI at it:
docker network connect rip-maf-net open-webui
```

## Run Open WebUI (standalone)

```powershell
docker run -d -p 3000:8080 `
  --add-host=host.docker.internal:host-gateway `
  -v open-webui:/app/backend/data `
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 `
  -e OFFLINE_MODE=true `
  -e HF_HUB_OFFLINE=1 `
  -e RAG_EMBEDDING_ENGINE=ollama `
  -e RAG_OLLAMA_BASE_URL=http://host.docker.internal:11434 `
  -e RAG_EMBEDDING_MODEL=nomic-embed-text `
  -e ENABLE_OPENAI_API=False `
  -e CORS_ALLOW_ORIGIN=http://localhost:3000 `
  -e WEBUI_SECRET_KEY=$env:WEBUI_SECRET_KEY `
  -e RIP_BASE_URL=http://backend:8000 `
  -e RIP_SERVICE_KEY=$env:RIP_SERVICE_KEY `
  --name open-webui --restart always `
  ghcr.io/open-webui/open-webui:main
docker network connect rip-maf-net open-webui   # so RIP_BASE_URL resolves
```

Then install the two Open WebUI functions from `integrations/openwebui/`
(see that folder's README) and select the **RIP** model.

## Layout

```
src/rip_maf/
  main.py config.py db.py logging.py auth.py
  api/        runs.py corpus.py deps.py
  rag/        pipeline.py vector_rag.py
  tools/      base schema registry executor + the five tools
  agents/     base registry reasoning coding
  providers/  base ollama streaming tracing
  orchestration/  orchestrator engine planner plan plan_graph react_engine
                  router builders intents validator aggregator results
                  idle_guard react memory
  runs/       manager.py store.py
  envelope.py artifacts.py observability/ services/
integrations/openwebui/  rip_pipe.py rip_filter.py
tests/
```

## Tests & lint

```powershell
pip install -e .[dev]
ruff check src
pytest
```
