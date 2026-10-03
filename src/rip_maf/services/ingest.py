# file_processor.py 

import asyncio
import os
from typing import Any

from rip_maf.core.config import settings
from rip_maf.core.db import pg_connection
from rip_maf.core.logging import setup_logging

logger = setup_logging()

async def run_rag_pipeline(file_id: str, rag: Any):
    try:
        # ─── Fetch metadata ───
        with pg_connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT corpus_id, file_name FROM files WHERE file_id=%s",
                (file_id,),
            )
            result = cur.fetchone()

        if not result:
            raise RuntimeError(f"File not found: {file_id}")

        corpus_id = str(result[0])
        file_name = result[1]
        ext = os.path.splitext(file_name or "")[1] or ".pdf"
        file_path = os.path.join(settings.upload_dir, corpus_id, f"{file_id}{ext}")

        logger.info(f"Starting pipeline: {file_name}")

        # ─── Blocking steps → offload to thread ───
        try:
            documents = await asyncio.to_thread(rag.document_loader, file_path)
            logger.info("Document loaded")
        except Exception:
            logger.exception("Document loading failed")
            raise

        try:
            chunks = await asyncio.to_thread(rag.chunk_documents, documents, file_name)
            logger.info(f"Chunks created: {len(chunks)}")
        except Exception:
            logger.exception("Chunking failed")
            raise

        try:
            embeddings = await asyncio.to_thread(rag.generate_embeddings, chunks)
            logger.info("Embeddings generated")
        except Exception:
            logger.exception("Embedding failed")
            raise

        try:
            await asyncio.to_thread(
                rag.store_chunks_and_embeddings, file_id, chunks, embeddings
            )
            logger.info("Embeddings stored")
        except Exception:
            logger.exception("Embedding storage failed")
            raise

        # ─── Mark ready ───
        with pg_connection() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE files SET file_status = 'ready' WHERE file_id = %s",
                (file_id,),
            )
        logger.info(f"File ready: {file_id}")

    except Exception:
        logger.exception(f"Pipeline failed: {file_id}")
        try:
            with pg_connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE files SET file_status = 'error' WHERE file_id = %s",
                    (file_id,),
                )
        except Exception:
            logger.exception(f"Failed to mark error status: {file_id}")