# rip_maf/main.py — standalone agentic-RAG backend for the Open WebUI Pipe.
#
# Loads VectorRAG once in the lifespan, composes the runtime (provider →
# agent/tool registries → orchestrator → run manager) and installs it into
# `rip_maf.api.deps`. Served as `uvicorn rip_maf.main:app`.

from contextlib import asynccontextmanager

from dotenv import find_dotenv, load_dotenv
from fastapi import Depends, FastAPI

# Load FIRST, before any rip_maf import: the settings singleton is built at
# first config import, so a late load_dotenv() silently leaves defaults in
# force. find_dotenv searches CWD upward, so rip-maf/.env loads from any CWD.
load_dotenv(find_dotenv(usecwd=True))

from rip_maf.api import corpus as corpus_api
from rip_maf.api import deps, runs
from rip_maf.auth import require_principal
from rip_maf.core.logging import setup_logging
from rip_maf.observability.langfuse import flush as langfuse_flush
from rip_maf.observability.langfuse import init_langfuse

logger = setup_logging()
init_langfuse()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Deferred: vector_rag pulls torch. The singleton is constructed once here
    # so rag.query reuses it instead of reloading models per query.
    from rip_maf.tools.rag_query import bind_rag_singleton

    # Upload reaper: mark files stuck in 'processing' for >30 min as 'error'.
    try:
        from rip_maf.core.db import pg_connection

        with pg_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE files SET file_status = 'error'
                WHERE file_status = 'processing'
                AND created_at < NOW() - INTERVAL '30 minutes'
            """
            )
            if cur.rowcount > 0:
                logger.warning(
                    "upload reaper marked %d stale processing file(s) as error",
                    cur.rowcount,
                )
    except Exception:
        logger.warning("upload reaper failed, continuing", exc_info=True)

    logger.info("Initiating ML models...")
    from rip_maf.core.config import settings as _settings

    logger.info(
        "BGE m3=%s reranker=%s",
        _settings.bge_m3_model_path,
        _settings.bge_reranker_v2_m3,
    )
    try:
        from rip_maf.rag.vector_rag import VectorRAG

        app.state.rag = VectorRAG()
        bind_rag_singleton(app.state.rag)
        logger.info("ML models loaded successfully.")
    except Exception:
        # Degraded boot, not a crash: /v1/* fails honest per request.
        app.state.rag = None
        logger.warning(
            "ML models NOT loaded (BGE weights missing?) — rag.query fails honest",
            exc_info=True,
        )

    try:
        from rip_maf.agents.registry import get_default_agent_registry
        from rip_maf.orchestration.aggregator import Aggregator
        from rip_maf.orchestration.orchestrator import Orchestrator
        from rip_maf.orchestration.planner import Planner
        from rip_maf.orchestration.validator import PlanValidator
        from rip_maf.providers.ollama import OllamaProvider
        from rip_maf.providers.tracing import wrap_provider
        from rip_maf.runs.manager import RunManager
        from rip_maf.tools.registry import get_default_tool_registry

        provider = wrap_provider(OllamaProvider())
        agent_registry = get_default_agent_registry(provider)
        tool_registry = get_default_tool_registry(
            rag=app.state.rag, provider=provider
        )
        orchestrator = Orchestrator(
            Planner(provider, agent_registry, tool_registry),
            PlanValidator(agent_registry, tool_registry),
            Aggregator(),
            agent_registry,
            tool_registry=tool_registry,
        )
        run_manager = RunManager(provider=provider, orchestrator=orchestrator)
        deps.configure(
            provider=provider,
            agent_registry=agent_registry,
            tool_registry=tool_registry,
            orchestrator=orchestrator,
            run_manager=run_manager,
        )
        logger.info("/v1 runtime configured (rag-bound tool registry)")
    except Exception:
        logger.warning(
            "/v1 runtime NOT configured (Ollama down?) — /v1/runs fails honest",
            exc_info=True,
        )

    yield

    logger.info("Shutting down and clearing models...")
    langfuse_flush()
    app.state.rag = None
    deps.reset()
    try:
        from rip_maf.core.db import close_pool

        close_pool()
    except Exception:
        logger.warning("DB pool cleanup failed", exc_info=True)


logger.info("Logging has been successfully set up.")

app = FastAPI(lifespan=lifespan)

# All routes require the Open WebUI service-key principal (no browser client).
_auth = [Depends(require_principal)]
app.include_router(corpus_api.router, dependencies=_auth)
app.include_router(runs.router, dependencies=_auth)
