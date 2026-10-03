"""
title: RIP Scope (documents owned by RIP)
author: rip-maf
version: 1.0.0
required_open_webui_version: 0.6.0
description: >
  Prevents Open WebUI from injecting its own document retrieval into turns
  handled by the RIP model, so RIP is the only retriever. It clears the
  resolved RAG sources Open WebUI would wrap into the prompt; the uploaded
  files themselves are left in place because the RIP Pipe needs them to
  ingest into RIP's corpus.

  Recommended companion setup: set the RIP model's "File Context" capability
  OFF and do not attach Open WebUI Knowledge bases. This filter is the
  belt-and-braces guard.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Filter:
    class Valves(BaseModel):
        TARGET_MODEL_MATCH: str = Field(
            default="rip",
            description="Substring matched against the model id; matching turns get their OWUI RAG sources cleared.",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def inlet(
        self,
        body: dict,
        __user__: dict | None = None,
        __metadata__: dict | None = None,
    ) -> dict:
        model = str(body.get("model") or "").lower()
        if self.valves.TARGET_MODEL_MATCH.lower() not in model:
            return body

        metadata = body.get("metadata")
        if isinstance(metadata, dict):
            metadata["sources"] = []
        if isinstance(__metadata__, dict):
            __metadata__["sources"] = []

        # Keep body["files"] intact: the RIP Pipe ingests those into RIP.
        # Dropping OWUI's resolved sources stops the citation wrap from
        # duplicating OWUI retrieval in the prompt RIP receives.
        return body
