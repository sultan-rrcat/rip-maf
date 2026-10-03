"""
title: RIP (Research Intelligence Platform)
author: rip-maf
version: 1.0.0
required_open_webui_version: 0.6.0
requirements: httpx
description: >
  Drives the RIP agentic-RAG backend as an Open WebUI model. Selecting "RIP"
  routes the chat turn to RIP's multi-agent orchestrator (router -> plan DAG
  -> ReAct fallback, 3 agents, 7 tools) over RIP's own document corpus.
  Files uploaded in the chat are ingested by RIP (Docling + BGE-M3 + pgvector),
  not by Open WebUI's knowledge base. Conversation history is supplied by
  Open WebUI; RIP is stateless over conversations.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, AsyncGenerator

import httpx
from pydantic import BaseModel, Field


def _content_to_text(content: Any) -> str:
    """Flatten an Open WebUI message content (str or multimodal parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                out.append(part.get("text", ""))
            elif isinstance(part, str):
                out.append(part)
        return "\n".join(p for p in out if p)
    return str(content or "")


class Pipe:
    class Valves(BaseModel):
        # Defaults read the container environment at import time, so passing
        # -e RIP_BASE_URL / -e RIP_SERVICE_KEY to the Open WebUI container
        # configures these valves with no UI editing.
        RIP_BASE_URL: str = Field(
            default=os.getenv("RIP_BASE_URL", "http://backend:8000"),
            description="RIP backend base URL reachable from the Open WebUI container.",
        )
        RIP_SERVICE_KEY: str = Field(
            default=os.getenv("RIP_SERVICE_KEY", ""),
            description="Shared secret sent as X-RIP-Service-Key (must match RIP_SERVICE_KEY on the backend).",
        )
        SHOW_REASONING: bool = Field(
            default=False,
            description="Show RIP's plan/steps (collapsed details) instead of streaming only the final answer.",
        )
        SHOW_STATUS: bool = Field(
            default=True,
            description="Show a live status pill while RIP works.",
        )
        MAX_HISTORY: int = Field(
            default=20,
            description="Max prior turns forwarded to RIP as context.",
        )

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self) -> list[dict]:
        return [{"id": "rip", "name": "RIP"}]

    # -- helpers -----------------------------------------------------------

    def _base_url(self) -> str:
        # Env fallback at call time: Open WebUI persists Valve values in its DB,
        # so a function imported before the env was set can hold empty valves.
        return (
            self.valves.RIP_BASE_URL or os.getenv("RIP_BASE_URL", "http://backend:8000")
        ).rstrip("/")

    def _service_key(self) -> str:
        return (self.valves.RIP_SERVICE_KEY or os.getenv("RIP_SERVICE_KEY", "")).strip()

    def _headers(self, __user__: dict | None) -> dict:
        user = __user__ or {}
        return {
            "X-RIP-Service-Key": self._service_key(),
            "X-RIP-User-Id": str(user.get("id") or user.get("email") or "anonymous"),
            "X-RIP-User-Name": str(user.get("name") or user.get("username") or ""),
        }

    @staticmethod
    def _sse_payload(line: str) -> dict | None:
        line = line.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            return None
        raw = line[len("data:") :].strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return None

    async def _ensure_corpus(self, client: httpx.AsyncClient, chat_id: str) -> str:
        resp = await client.post(
            f"{self._base_url()}/v1/corpus",
            json={"chat_id": chat_id},
        )
        resp.raise_for_status()
        return resp.json()["corpus_id"]

    async def _ingest_files(
        self, client: httpx.AsyncClient, corpus_id: str, files: list[dict]
    ) -> None:
        for item in files or []:
            meta = item.get("file") or {}
            path = meta.get("path")
            filename = meta.get("filename") or item.get("name") or "upload"
            if not path or str(path).startswith(("http://", "https://", "s3://")):
                # Non-local storage is not supported by the local-path ingest
                # path (demo uses local storage). Skip rather than fail the turn.
                continue
            if not os.path.exists(path):
                continue
            with open(path, "rb") as fh:
                data = fh.read()
            resp = await client.post(
                f"{self._base_url()}/v1/corpus/{corpus_id}/files",
                files={"file": (filename, data)},
            )
            resp.raise_for_status()

    async def _fetch_artifact(
        self, client: httpx.AsyncClient, url: str
    ) -> tuple[bytes, str] | None:
        try:
            resp = await client.get(f"{self._base_url()}{url}")
            resp.raise_for_status()
            return resp.content, resp.headers.get("content-type", "")
        except Exception:
            return None

    @staticmethod
    def _render_artifact(kind: str, mime: str, filename: str, data: bytes) -> str | None:
        if kind == "chart" or mime.startswith("image/"):
            b64 = base64.b64encode(data).decode("ascii")
            return f"![{filename}](data:{mime or 'image/png'};base64,{b64})"
        # Documents/images that the browser cannot reach on the internal RIP
        # network are surfaced by name only.
        return f"- **{filename}** (`{kind}`)"

    # -- main --------------------------------------------------------------

    async def pipe(
        self,
        body: dict,
        __user__: dict | None = None,
        __metadata__: dict | None = None,
        __files__: list | None = None,
        __event_emitter__=None,
    ) -> AsyncGenerator[str, None]:
        metadata = __metadata__ or {}
        headers = self._headers(__user__)
        base = self._base_url()

        if not headers["X-RIP-Service-Key"]:
            yield (
                "RIP: service key is not configured. Set the `rip` function's "
                "RIP_SERVICE_KEY valve (or the Open WebUI container's "
                "RIP_SERVICE_KEY env) to match the backend."
            )
            return

        user_prompt = (metadata.get("user_prompt") or "").strip()
        if not user_prompt:
            messages = body.get("messages") or []
            if messages:
                user_prompt = _content_to_text(messages[-1].get("content"))
        if not user_prompt:
            yield "No user message."
            return

        chat_id = metadata.get("chat_id") or body.get("chat_id") or "default"

        # Prior turns (Open WebUI owns history); drop the final user turn,
        # which travels separately as `message`.
        normalized = []
        for m in body.get("messages") or []:
            role = m.get("role")
            if role not in ("user", "assistant", "system"):
                continue
            normalized.append({"role": role, "content": _content_to_text(m.get("content"))})
        if normalized and normalized[-1]["role"] == "user":
            normalized = normalized[:-1]
        history = normalized[-self.valves.MAX_HISTORY :]

        async def emit_status(description: str, done: bool = False) -> None:
            if self.valves.SHOW_STATUS and __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": description, "done": done}}
                )

        streaming = bool(body.get("stream", True))

        async with httpx.AsyncClient(timeout=None, headers=headers) as client:
            try:
                corpus_id = await self._ensure_corpus(client, chat_id)
            except Exception as e:
                yield f"RIP: could not resolve corpus: {e}"
                return

            try:
                if __files__:
                    await emit_status("Ingesting documents into RIP…")
                    await self._ingest_files(client, corpus_id, __files__)
            except Exception as e:
                yield f"RIP: file ingestion failed: {e}"
                return

            try:
                resp = await client.post(
                    f"{base}/v1/runs",
                    json={
                        "corpus_id": corpus_id,
                        "message": user_prompt,
                        "history": history,
                    },
                )
                resp.raise_for_status()
                run_id = resp.json()["run_id"]
            except Exception as e:
                yield f"RIP: could not start run: {e}"
                return

            await emit_status("RIP is working…")
            chunks: list[str] = []
            final = ""
            sources: list[dict] = []
            artifacts: list[dict] = []
            error_text = ""
            step_lines: list[str] = []

            try:
                async with client.stream(
                    "GET", f"{base}/v1/runs/{run_id}/events"
                ) as stream:
                    stream.raise_for_status()
                    async for line in stream.aiter_lines():
                        ev = self._sse_payload(line)
                        if not ev:
                            continue
                        etype = ev.get("type")
                        if etype == "delta":
                            text = ev.get("content", "")
                            if not text:
                                continue
                            if self.valves.SHOW_REASONING:
                                chunks.append(text)
                                if streaming:
                                    yield text
                        elif etype == "plan":
                            for s in ev.get("steps", []):
                                step_lines.append(
                                    f"| {s.get('step_id')} | {s.get('executor')} "
                                    f"| {s.get('expected_output_type')} |"
                                )
                        elif etype == "step_completed":
                            step_lines.append(
                                f"| {ev.get('step_id')} | {ev.get('status')} |"
                            )
                        elif etype == "sources":
                            sources.extend(ev.get("sources") or [])
                        elif etype == "artifacts":
                            artifacts.extend(ev.get("artifacts") or [])
                        elif etype == "summary":
                            final = ev.get("content", "") or final
                        elif etype == "error":
                            error_text = ev.get("message", "unknown error")
                        elif etype in ("run_completed", "cancelled"):
                            pass
            except httpx.HTTPError as e:
                error_text = error_text or f"stream error: {e}"
            finally:
                # Cooperative cancel on early generator close (user pressed Stop).
                try:
                    await client.post(f"{base}/v1/runs/{run_id}/cancel")
                except Exception:
                    pass

            await emit_status("Done", done=True)

            if error_text and not final:
                yield f"RIP error: {error_text}"
                return

            # Assemble trailing blocks (sources, artifacts, optional steps).
            appendix: list[str] = []
            if sources:
                seen = set()
                lines = []
                for s in sources:
                    key = (s.get("source"), s.get("section"))
                    if key in seen:
                        continue
                    seen.add(key)
                    section = f" — {s['section']}" if s.get("section") else ""
                    lines.append(f"- `{s.get('source', '?')}`{section}")
                appendix.append("**Sources**\n" + "\n".join(lines))
            for a in artifacts:
                fetched = await self._fetch_artifact(client, a.get("url", ""))
                if not fetched:
                    appendix.append(f"- **{a.get('filename', 'artifact')}** ({a.get('kind')})")
                    continue
                rendered = self._render_artifact(
                    a.get("kind", ""), fetched[1] or a.get("mime", ""),
                    a.get("filename", "artifact"), fetched[0],
                )
                if rendered:
                    appendix.append(rendered)
            if self.valves.SHOW_REASONING and step_lines:
                appendix.append(
                    "<details><summary>Steps</summary>\n\n"
                    "| step | provider/status | type |\n|---|---|---|\n"
                    + "\n".join(step_lines)
                    + "\n</details>"
                )

            tail = ("\n\n" + "\n\n".join(appendix)) if appendix else ""

            if streaming and not self.valves.SHOW_REASONING:
                # deltas were suppressed; emit the final answer now.
                answer = (final or "").strip()
                if answer:
                    yield answer
                if tail:
                    yield tail
                return

            if streaming:
                if tail:
                    yield tail
                return

            # non-streaming: single concatenated string
            body_text = "".join(chunks) if self.valves.SHOW_REASONING else (final or "")
            yield body_text + tail
