# RIP-MAF ↔ Open WebUI

Open WebUI is the frontend; rip-maf is the agentic-RAG backend, exposed as a
selectable model. Uploaded files are parsed and embedded by rip-maf (Docling
+ BGE-M3 + pgvector), not by Open WebUI's knowledge base.

## How it works

```
Open WebUI chat
  └─ Filter  rip_filter.py   clears OWUI's own RAG sources for the RIP model
  └─ Pipe    rip_pipe.py     ensure corpus → ingest files → POST /v1/runs → SSE
        │  X-RIP-Service-Key + X-RIP-User-Id
        ▼
rip-maf      /v1/corpus (ensure + ingest), /v1/runs (answer)
        Router → Builders → ReAct · RAG · tools · Ollama
```

- Corpus = one per `(Open WebUI user, chat)`.
- rip-maf is stateless over conversations: Open WebUI sends the history.
- rip-maf stays internal-only; the Pipe inlines charts as data URIs and
  names documents.

## Install

The Open WebUI container is configured with `RIP_BASE_URL` and
`RIP_SERVICE_KEY` in its environment, so the Pipe reads them into its Valves
automatically — no manual URL/key entry. It reaches the backend at
`http://backend:8000` because it is attached to the `rip-maf-net` network
(see below).

In Open WebUI:

1. **Admin → Functions → Create** twice (paste the file content; with
   `OFFLINE_MODE=true` the "Import from Link" GitHub fetch is blocked):
   - Name `rip`, paste `rip_pipe.py`.
   - Name `rip_scope`, paste `rip_filter.py`.
   Leaving each as `Action`/auto-detected is fine — the class name
   (`Pipe` / `Filter`) determines the type.
2. **Workspace → Models**: the `RIP` model appears. Edit it and attach the
   `rip_scope` **Filter**.
3. Recommended for the RIP model: turn **File Context** capability **off** and
   do not attach Open WebUI Knowledge bases, so RIP is the only retriever.

### Standalone Open WebUI (docker run)

The backend is published on `127.0.0.1:${HOST_BACKEND_PORT}` for host tools,
but a standalone OWUI container reaches it over `rip-maf-net`. After starting
the rip-maf stack:

```powershell
docker network connect rip-maf-net open-webui
```

and start OWUI with the two RIP env vars plus `WEBUI_SECRET_KEY`:

```powershell
docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway `
  -v open-webui:/app/backend/data `
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 `
  -e OFFLINE_MODE=true -e HF_HUB_OFFLINE=1 `
  -e RAG_EMBEDDING_ENGINE=ollama `
  -e RAG_OLLAMA_BASE_URL=http://host.docker.internal:11434 `
  -e RAG_EMBEDDING_MODEL=nomic-embed-text `
  -e ENABLE_OPENAI_API=False -e CORS_ALLOW_ORIGIN=http://localhost:3000 `
  -e WEBUI_SECRET_KEY=$env:WEBUI_SECRET_KEY `
  -e RIP_BASE_URL=http://backend:8000 `
  -e RIP_SERVICE_KEY=$env:RIP_SERVICE_KEY `
  --name open-webui --restart always ghcr.io/open-webui/open-webui:main
```

## Run

```powershell
copy .env.example .env      # set RIP_SERVICE_KEY, OLLAMA_BASE_URL, BGE paths
docker compose up -d --build    # postgres + backend (OWUI runs separately)
docker network connect rip-maf-net open-webui    # if using standalone docker run
```

Open `http://localhost:${HOST_OWUI_PORT}` (default 3000), create the admin
account, select the **RIP** model, drop a PDF into the chat, and ask about it.

## Demo script

1. Open WebUI → select **RIP**.
2. New chat → upload a PDF → ask a question answerable only from it.
3. Show the streamed grounded answer and the **Sources** block.
4. Ask something that triggers `plot.chart` (e.g. "chart the figures") → show
   the inline chart.
5. Turn on the `rip` Pipe's `SHOW_REASONING` valve to reveal the plan/steps.

## Limits (current)

- Ingest supports OWUI **local** file storage; S3/GCS paths are skipped.
- Files are ingested synchronously in the Pipe before the run (first turn on a
  large PDF is slower).
- Document artifacts are surfaced by name only (the browser cannot reach RIP on
  the internal network); charts and images are inlined.
- Corpus lifecycle on chat deletion is not implemented yet.
