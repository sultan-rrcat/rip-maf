
# pipeline.py 


import json

from docling.document_converter import DocumentConverter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from psycopg2.extras import Json
from sentence_transformers import CrossEncoder

from rip_maf.core.config import settings
from rip_maf.core.db import pg_connection
from rip_maf.core.logging import setup_logging

logger = setup_logging()


class RagPipeline:
    def __init__(self):
        from pathlib import Path

        for label, model_path in (
            ("BGE_M3_MODEL_PATH", settings.bge_m3_model_path),
            ("BGE_RERANKER_V2_M3", settings.bge_reranker_v2_m3),
        ):
            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"BGE model missing at {model_path} ({label}) — "
                    "set an absolute path in .env"
                )
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=settings.bge_m3_model_path
        )
        self.reranker_model = CrossEncoder(settings.bge_reranker_v2_m3)

    # =========================
    # 📄 DOCUMENT LOADER
    # =========================
    def _load_with_docling(self, file):
        # Primary: Docling (re-enabled P0.1; fallback below needs Java)
        converter = DocumentConverter()
        result = converter.convert(file)

        md_text = result.document.export_to_markdown()

        logger.info("✅ Loaded with Docling")

        return [
            Document(
                page_content=md_text, metadata={"source": file, "loader": "docling"}
            )
        ]

    def _load_with_opendataloader(self, file):
        loader = OpenDataLoaderPDFLoader(file, format="markdown")
        documents = loader.load_and_split()

        logger.info(f"✅ Loaded with OpenDataLoader: {len(documents)} pages")

        return documents  # already Document objects

    def document_loader(self, file):
        # Order from settings (RAG_PDF_LOADER); the other loader is fallback.
        primary = (settings.rag_pdf_loader or "docling").strip().lower()
        loaders = (
            (self._load_with_opendataloader, self._load_with_docling)
            if primary == "opendataloader"
            else (self._load_with_docling, self._load_with_opendataloader)
        )
        names = (
            ("OpenDataLoader", "Docling")
            if primary == "opendataloader"
            else ("Docling", "OpenDataLoader")
        )
        try:
            return loaders[0](file)
        except Exception as e:  # noqa: BLE001 - loader fallback must catch anything
            logger.warning(f"⚠️ {names[0]} failed, fallback to {names[1]}: {e}")

            try:
                return loaders[1](file)
            except Exception as e:
                logger.info(f"❌ Both loaders failed: {e}")
                raise

    # =========================
    # ✂️ CHUNKING
    # =========================
    def chunk_documents(self, documents, file_name):
        try:
            # STEP 1: Markdown structure
            headers_to_split_on = [
                ("#", "H1"),
                ("##", "H2"),
                ("###", "H3"),
            ]

            md_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on
            )

            final_chunks = []

            # Join pages BEFORE splitting. Loaders return one Document per
            # page, and a Markdown heading that lands at the end of a page
            # carries no body until the next page. Splitting per page made
            # MarkdownHeaderTextSplitter emit no chunk for such a trailing
            # heading, silently discarding it — its body then reappeared on
            # the next page with NO header metadata at all (observed live:
            # 18-page lab report whose "Lab 4: Network and Information Lab"
            # heading sat at the end of page 10; Lab 4's content was stored
            # but unlabelled, so "Lab 4" existed nowhere in `embeddings` and
            # a whole-file rag.query answer had to omit it). Joining lets the
            # heading attach to the body it introduces.
            joined_markdown = "\n\n".join(doc.page_content for doc in documents)

            for chunk in md_splitter.split_text(joined_markdown):
                # 🔥 Extract headers safely
                h1 = chunk.metadata.get("H1")
                h2 = chunk.metadata.get("H2")
                h3 = chunk.metadata.get("H3")

                # 🔥 Build clean metadata
                clean_metadata = {
                    "source": file_name,  # only filename, not full path
                }

                # Only include headers if they exist
                if h1:
                    clean_metadata["H1"] = h1.strip()
                if h2:
                    clean_metadata["H2"] = h2.strip()
                if h3:
                    clean_metadata["H3"] = h3.strip()

                final_chunks.append(
                    Document(
                        page_content=chunk.page_content.strip(),
                        metadata=clean_metadata,
                    )
                )

            logger.info(
                f"[Step 1] Final chunks (Markdown Splitter): {len(final_chunks)}"
            )
            return final_chunks

        except Exception:
            logger.exception("Error splitting")
            raise

    # =========================
    # 🧠 EMBEDDINGS
    # =========================
    def generate_embeddings(self, chunks):
        try:
            texts = [doc.page_content for doc in chunks]
            embeddings = self.embedding_model.embed_documents(texts)
            return embeddings

        except Exception:
            logger.exception("Error while generating embeddings")
            raise

    # =========================
    # 💾 STORE
    # =========================

    def store_chunks_and_embeddings(self, file_id, chunks, embeddings) -> list[str]:

        def parse_metadata(metadata):
            if isinstance(metadata, str):
                try:
                    return json.loads(metadata)
                except ValueError:
                    return {"raw": metadata}
            return metadata

        embedding_ids = []
        deleted = 0

        try:
            with pg_connection() as conn:
                with conn.cursor() as cur:
                    # Replace, never append. Re-processing a file is the
                    # documented recovery path (POST /api/files/{id}/process,
                    # ADR-005 / PLAN B3), and INSERT-only left the previous
                    # chunks in place — so every retry duplicated content,
                    # corrupted `chunk_index`, and inflated the whole-file
                    # size probe until ADR-033's shortcut silently stopped
                    # firing for the file. Same transaction as the inserts
                    # below, so a failure mid-write rolls back to the old set.
                    cur.execute(
                        "DELETE FROM embeddings WHERE file_id = %s",
                        (file_id,),
                    )
                    deleted = cur.rowcount

                    for i, (doc, embedding) in enumerate(zip(chunks, embeddings)):
                        cur.execute(
                            """
                            INSERT INTO embeddings
                                (file_id, chunk_index, chunk_text, embedding, metadata)
                            VALUES (%s, %s, %s, %s, %s)
                            RETURNING embedding_id
                            """,
                            (
                                file_id,
                                i,  # explicit index
                                doc.page_content,
                                embedding,
                                Json(parse_metadata(doc.metadata)),
                            ),
                        )
                        row = cur.fetchone()
                        embedding_ids.append(str(row[0]))

                conn.commit()

            logger.info(
                f"✅ Replaced {len(embedding_ids)} embeddings "
                f"(deleted {deleted} previous for this file)."
            )
            return embedding_ids

        except Exception:
            logger.exception("Error storing embeddings")
            raise
