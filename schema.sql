CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ─── corpora ────────────────────────────────────────────────────────────────
-- One corpus per (owner, chat). Retrieval and runs are scoped by corpus_id.

CREATE TABLE IF NOT EXISTS public.corpora
(
    corpus_id   uuid NOT NULL DEFAULT gen_random_uuid(),
    name        text COLLATE pg_catalog."default" NOT NULL,
    corpus_ref  text COLLATE pg_catalog."default" NOT NULL,   -- Open WebUI chat id
    owner_ref   text COLLATE pg_catalog."default" NOT NULL,   -- Open WebUI user id
    created_at  timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT corpora_pkey PRIMARY KEY (corpus_id),
    CONSTRAINT corpora_owner_ref_corpus_ref_key UNIQUE (owner_ref, corpus_ref)
) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_corpora_owner
    ON public.corpora (owner_ref);

-- ─── files ──────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS public.files
(
    file_id     uuid NOT NULL DEFAULT gen_random_uuid(),
    corpus_id   uuid,
    file_name   text COLLATE pg_catalog."default" NOT NULL,
    file_size   integer,
    file_status character varying(20) COLLATE pg_catalog."default",
    created_at  timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT files_pkey PRIMARY KEY (file_id),
    CONSTRAINT files_corpus_id_fkey FOREIGN KEY (corpus_id)
        REFERENCES public.corpora (corpus_id)
        ON UPDATE NO ACTION
        ON DELETE CASCADE
) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_files_corpus
    ON public.files (corpus_id);

-- ─── embeddings ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS public.embeddings
(
    embedding_id uuid NOT NULL DEFAULT gen_random_uuid(),
    file_id      uuid,
    chunk_text   text COLLATE pg_catalog."default",
    embedding    vector(1024),
    chunk_index  integer,
    metadata     jsonb,
    text_search  tsvector GENERATED ALWAYS AS (to_tsvector('english'::regconfig, chunk_text)) STORED,
    created_at   timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT embeddings_pkey PRIMARY KEY (embedding_id),
    CONSTRAINT embeddings_file_id_fkey FOREIGN KEY (file_id)
        REFERENCES public.files (file_id)
        ON UPDATE NO ACTION
        ON DELETE CASCADE
) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS embeddings_embedding_idx
    ON public.embeddings USING hnsw (embedding vector_cosine_ops)
    TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS text_search_idx
    ON public.embeddings USING gin (text_search)
    TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_embeddings_file
    ON public.embeddings (file_id);

-- ─── runs ───────────────────────────────────────────────────────────────────
-- Ephemeral orchestration units; persisted to survive reconnects.

CREATE TABLE IF NOT EXISTS public.runs
(
    id          uuid NOT NULL DEFAULT gen_random_uuid(),
    corpus_id   uuid NOT NULL,
    status      character varying(20) COLLATE pg_catalog."default" NOT NULL DEFAULT 'pending',
    goal        text,
    plan        jsonb,
    result      jsonb,
    created_at  timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at  timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT runs_pkey PRIMARY KEY (id),
    CONSTRAINT runs_corpus_id_fkey FOREIGN KEY (corpus_id)
        REFERENCES public.corpora (corpus_id)
        ON UPDATE NO ACTION
        ON DELETE CASCADE,
    CONSTRAINT runs_status_check
        CHECK (status::text = ANY (ARRAY['pending','running','completed','failed','cancelled']::text[]))
) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_runs_corpus
    ON public.runs (corpus_id);

-- ─── run_events ─────────────────────────────────────────────────────────────
-- SSE event log per run (structural replay); deltas are live-only.

CREATE TABLE IF NOT EXISTS public.run_events
(
    id         serial PRIMARY KEY,
    run_id     uuid NOT NULL,
    seq        integer NOT NULL,
    event_type character varying(30) COLLATE pg_catalog."default" NOT NULL,
    payload    jsonb,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT run_events_run_id_fkey FOREIGN KEY (run_id)
        REFERENCES public.runs (id)
        ON UPDATE NO ACTION
        ON DELETE CASCADE
) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_run_events_run_seq
    ON public.run_events (run_id, seq);
