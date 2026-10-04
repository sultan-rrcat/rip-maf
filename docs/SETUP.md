# Setup Guide: rip-maf

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python | 3.11+ |
| Docker | Desktop (Windows/macOS) or Engine (Linux) |
| PostgreSQL | `pgvector/pgvector:pg16` via compose (extension `vector` enabled) |
| Ollama | Reachable at `OLLAMA_BASE_URL`, chat model pulled (`OLLAMA_DEFAULT_MODEL`) |
| BGE weights | `bge-m3` dir + `bge_reranker_v2_m3` dir on the host |

## 2. Environment

```powershell
copy .env.example .env
```

| Key | Purpose |
|---|---|
| `HOST_BACKEND_PORT` / `HOST_PG_PORT` | Host-side published ports (loopback-bound) |
| `DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD` | Postgres (compose backend uses `DB_HOST=postgres`) |
| `OLLAMA_BASE_URL` / `OLLAMA_DEFAULT_MODEL` / `OLLAMA_TIMEOUT_MS` | LLM |
| `BGE_MODELS_DIR` | Host dir mounted at `/app/models` in compose; override `BGE_M3_MODEL_PATH` / `BGE_RERANKER_V2_M3` for host-local runs |
| `RIP_SERVICE_KEY` | Shared secret the Open WebUI Pipe sends as `X-RIP-Service-Key`. Set the same value in the Pipe's Valves |
| `RAG_PDF_LOADER` | `docling` (default) or `opendataloader` |
| `OLLAMA_CONTEXT_WINDOW` | Must equal the Ollama server's `OLLAMA_CONTEXT_LENGTH` (`num_ctx` is ignored on `/v1/chat/completions`) |
| `RAG_WHOLE_FILE_PCT` | Whole-file RAG budget share (`0` disables) |
| `LANGFUSE_*` | Opt-in tracing (`LANGFUSE_ENABLED=false` default) |

## 3. Run

```powershell
docker compose up -d --build        # postgres + backend
# join a standalone Open WebUI container to the private network:
docker network connect rip-maf_rip-maf-net open-webui
```

Health is inferred from container state (no health endpoint by design) and
from `docker compose ps`. Fresh bootstrap replays `schema.sql` on an empty
`pgdata` volume: `corpora/files/embeddings/runs/run_events`.

Host-local dev (needs local Postgres + Ollama + weights):

```powershell
pip install -e .[dev]
$env:PYTHONPATH = "src"      # if not installed
uvicorn rip_maf.main:app --host 0.0.0.0 --port 8000
```

## 4. Smoke test (needs the service key)

```powershell
$h = @{
  "X-RIP-Service-Key" = "<RIP_SERVICE_KEY>"
  "X-RIP-User-Id"     = "smoke-user"
  "X-RIP-User-Name"   = "Smoke"
}
$c = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8010/v1/corpus" `
     -Headers $h -ContentType "application/json" -Body '{"chat_id":"smoke-1"}'
$c.corpus_id
# upload a file, then:
$r = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8010/v1/runs" `
     -Headers $h -ContentType "application/json" `
     -Body (@{corpus_id=$c.corpus_id; message="summarize the document"; history=@()} | ConvertTo-Json -Depth 5)
$r.run_id
# stream events:
Invoke-WebRequest -Uri "http://127.0.0.1:8010/v1/runs/$($r.run_id)/events" -Headers $h -TimeoutSec 600
```

### 4b. Open WebUI (standalone)

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
docker network connect rip-maf_rip-maf-net open-webui   # so RIP_BASE_URL resolves
```

Then create the `rip` (Pipe) and `rip_scope` (Filter) functions from
`integrations/openwebui/` (see that folder's README) and select **RIP**.

## 5. Tests & lint

```powershell
pip install -e .[dev]
ruff check src tests
pytest                 # DB tests run when Postgres is reachable; Ollama-live tests skip when down
```

## Troubleshooting

- **401 on every call** → `RIP_SERVICE_KEY` mismatch between `.env` and the Pipe;
  also required: `X-RIP-User-Id` on every request.
- **503 on ingest** → BGE weights missing/unreadable (check `BGE_MODELS_DIR` / paths).
- **Empty/irrelevant answers** → Ollama model missing or wrong context length
  (`OLLAMA_CONTEXT_WINDOW` must equal the server window).
- **Slow first turn on big PDFs** → ingest runs inline in the Pipe before the run.
- **"no deterministic builder for intent unknown — L3 ReAct required"** → the router
  LLM call failed (most often `OLLAMA_DEFAULT_MODEL` not pulled: check
  `ollama list`); L3 ReAct then fails on the same missing model and the honest
  routing error surfaces.
- **500 on file ingest / "VectorRAG singleton is not bound"** → BGE weights not
  found; point `BGE_MODELS_DIR` at the host dir containing `bge-m3/` and
  `reranker/` and recreate the backend.
- **No traces in Langfuse** → `LANGFUSE_HOST` must be reachable *from inside the
  backend container*: use `http://langfuse-langfuse-web-1:3000` and
  `docker network connect langfuse_default rip-maf-backend-1` (plain
  `localhost:3002` gets connection-refused and spans are dropped).
- **"service key is not configured" in Open WebUI** → `RIP_SERVICE_KEY` empty in
  the Open WebUI container env (shell variable was unset at `docker run`);
  recreate it with the same value as the backend, or set the Pipe's valve.
- **Network name** → compose names it `rip-maf_rip-maf-net` (project prefix),
  not `rip-maf-net`.
