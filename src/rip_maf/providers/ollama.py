"""Ollama provider: multi-model local serving over the OpenAI-compatible API.

Implements the ModelProvider contract (Charter NFR-7) against Ollama's
OpenAI-compat endpoint (ollama.com/blog/openai-compatibility, verified; the
compat layer maps response_format onto the native `format` param for
structured output — ollama.com/blog/structured-outputs).

Deltas, all contract-driven:

- MULTI-MODEL: Ollama serves any pulled model, chosen per request. Requested
  names the server actually has are honored verbatim; everything else
  resolves to `ollama_default_model`, logged once and reflected in
  served_model() so traces record what will really run.
- REMOTE HOST + PROXY TRAP: the server lives on the LAN and developer shells
  carry proxy env vars (curl needs --noproxy "*"). The httpx client therefore
  sets trust_env=False — default httpx honors HTTP(S)_PROXY, which would
  intercept this traffic.
- INIT: /api/tags is both the fail-honest health check and the source of the
  available-model set used for resolution above.

Lifecycle contract: the server is external (started by the user); this
provider owns no process and fails honestly at init if unreachable.

RIP port: no plugin system — direct ModelProvider subclass (ADR-017).
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import httpx

from rip_maf.core.config import settings

from .base import ModelProvider
from .streaming import ThinkFilter, strip_think

logger = logging.getLogger("providers.ollama")


class OllamaProvider(ModelProvider):
    """Ollama provider (no plugin system — direct ModelProvider subclass)."""

    def __init__(self) -> None:
        # trust_env=False is LOAD-BEARING: default httpx honors HTTP(S)_PROXY,
        # which would route this LAN traffic through the environment proxy.
        self._client = httpx.Client(
            base_url=settings.ollama_base_url,
            timeout=settings.ollama_timeout_ms / 1000.0,
            trust_env=False,
        )
        self._last_usage: dict[str, int] | None = None
        self._available: set[str] = set()
        self._warned: set[str] = set()
        self._reachable = False

        # Degraded boot: a 5s probe only — never raise here. Per-request paths
        # re-probe and fail honest when Ollama is still down.
        self._probe_once(timeout=5.0)

    def _probe_once(self, *, timeout: float) -> bool:
        """Single /api/tags probe; returns reachability, never raises."""
        try:
            response = self._client.get("/api/tags", timeout=timeout)
        except httpx.RequestError as e:
            logger.warning(
                "Ollama unreachable at %s: %s (degraded — per-request fail-honest)",
                settings.ollama_base_url,
                e,
            )
            self._reachable = False
            self._available = set()
            return False
        if response.status_code != 200:
            logger.warning(
                "Ollama health check status %s (degraded — per-request fail-honest)",
                response.status_code,
            )
            self._reachable = False
            self._available = set()
            return False
        try:
            self._available = {
                m["name"]
                for m in response.json().get("models", [])
                if m.get("name")
            }
        except (json.JSONDecodeError, TypeError, KeyError, AttributeError) as e:
            logger.warning(
                "Ollama /api/tags unreadable (%s); per-request model routing "
                "disabled — everything serves the configured default",
                e,
            )
            self._available = set()
        self._reachable = True
        logger.info(
            "Ollama reachable at %s (%d model(s): %s)",
            settings.ollama_base_url,
            len(self._available),
            ", ".join(sorted(self._available)) or "none listed",
        )
        return True

    def ensure_ready(self) -> None:
        """Fail-honest per-request gate; re-probes once when degraded."""
        if self._reachable:
            return
        if not self._probe_once(timeout=5.0):
            raise ValueError(
                f"Could not connect to Ollama at {settings.ollama_base_url}. "
                "Start it or fix OLLAMA_BASE_URL (see .env.example)."
            )

    @property
    def last_usage(self) -> dict[str, int] | None:
        """Token usage from the most recent generation."""
        return self._last_usage

    def _resolve_model(self, requested: str) -> str:
        """Honor per-request names the server actually has; everything else
        resolves to the configured default — logged ONCE per foreign name,
        never silent, and         mirrored by served_model() for trace honesty."""
        if not self._reachable:
            return settings.ollama_default_model
        if requested in self._available:
            return requested
        if requested not in self._warned:
            self._warned.add(requested)
            logger.warning(
                "Requested model %r is not pulled on Ollama; serving %r "
                "instead (pulled models: %s)",
                requested,
                settings.ollama_default_model,
                ", ".join(sorted(self._available)) or "none",
            )
        return settings.ollama_default_model

    def _chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.ensure_ready()
        payload: dict[str, Any] = {
            "model": self._resolve_model(model),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Thinking models (e.g. lfm2.5) otherwise spend the token
            # budget on chain-of-thought and return empty content
            # (observed: done_reason=length, 64/64 tokens thinking).
            # Pre-thinking-era servers that reject the unknown field get
            # one retry without it below.
            "think": False,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        def _post(body: dict[str, Any]) -> httpx.Response:
            try:
                return self._client.post("/v1/chat/completions", json=body)
            except httpx.RequestError as e:
                logger.error("Ollama request failed: %s", e)
                raise RuntimeError(f"Ollama connection failure: {e}") from e

        response = _post(payload)
        if response.status_code != 200 and "think" in response.text.lower():
            logger.warning("Ollama rejected think flag; retrying without it")
            payload = {k: v for k, v in payload.items() if k != "think"}
            response = _post(payload)
        if response.status_code != 200:
            logger.error(
                "Ollama non-200 status=%s body=%s",
                response.status_code,
                response.text,
            )
            raise RuntimeError(
                f"Ollama error [status {response.status_code}]: {response.text}"
            )
        return response.json()

    def _stream_payload(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._resolve_model(model),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            # think=False: thinking models must not burn the token budget
            # on chain-of-thought (see _chat). stream_options asks for a
            # trailing usage chunk so last_usage stays honest. Servers that
            # reject either unknown field hit _chat_stream's fallback.
            "think": False,
            "stream_options": {"include_usage": True},
        }
        if response_format is not None:
            payload["response_format"] = response_format
        return payload

    def _post_stream(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Yield parsed SSE data events for one streaming POST.

        Raises the same honest RuntimeErrors as _chat (connection failure,
        non-200 with status + body). A `data:` line that is not JSON is
        transport noise: logged and skipped, never fatal to the stream.
        """
        self.ensure_ready()
        try:
            with self._client.stream(
                "POST", "/v1/chat/completions", json=payload
            ) as response:
                if response.status_code != 200:
                    response.read()
                    logger.error(
                        "Ollama non-200 status=%s body=%s",
                        response.status_code,
                        response.text,
                    )
                    raise RuntimeError(
                        f"Ollama error [status {response.status_code}]: {response.text}"
                    )
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError as e:
                        logger.warning("skipping malformed SSE chunk: %s", e)
        except httpx.RequestError as e:
            logger.error("Ollama request failed: %s", e)
            raise RuntimeError(f"Ollama connection failure: {e}") from e

    def _chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        payload = self._stream_payload(
            model, messages,
            temperature=temperature, max_tokens=max_tokens, response_format=response_format,
        )
        try:
            yield from self._post_stream(payload)
        except RuntimeError as e:
            # Older servers reject unknown fields instead of ignoring them:
            # retry ONCE with each named field stripped (usage then stays
            # None when stream_options goes — logged, never fabricated).
            err = str(e).lower()
            if "think" in err and "think" in payload:
                logger.warning("Ollama rejected think flag; retrying without it")
                payload = {k: v for k, v in payload.items() if k != "think"}
            if "stream_options" in err and "stream_options" in payload:
                logger.warning(
                    "Ollama rejected stream_options; retrying without usage request "
                    "(token usage will be missing for this generation)"
                )
                payload = {k: v for k, v in payload.items() if k != "stream_options"}
            if "think" in payload and "stream_options" in payload:
                raise
            yield from self._post_stream(payload)

    @staticmethod
    def _delta_content(event: dict[str, Any]) -> str:
        """Text of one stream event, or "" for control/usage-only events."""
        try:
            return event["choices"][0]["delta"].get("content") or ""
        except (KeyError, IndexError, AttributeError, TypeError):
            return ""

    @staticmethod
    def _content(data: dict[str, Any]) -> str:
        try:
            content = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError) as e:
            logger.error("malformed Ollama response: %s", e)
            raise RuntimeError(
                f"Malformed response structure from Ollama: {e}"
            ) from e
        # Thinking models can emit <think> blocks in content; strip closed
        # blocks, and drop everything after an unclosed one (the generation
        # budget was likely exhausted inside reasoning).
        return strip_think(content)

    def _record_usage(self, data: dict[str, Any]) -> None:
        usage = data.get("usage")
        if usage is None:
            self._last_usage = None
            return
        details: dict[str, int] = {
            "input": usage.get("prompt_tokens", 0),
            "output": usage.get("completion_tokens", 0),
        }
        # Langfuse usage buckets are mutually exclusive: `input` excludes any
        # input_* sub-bucket, so cached prompt tokens must be split out.
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        if cached:
            details["input"] = max(details["input"] - cached, 0)
            details["input_cached_tokens"] = cached
        details["total"] = usage.get("total_tokens", 0)
        self._last_usage = details

    def served_model(self, requested_model: str) -> str:
        # Multi-model server: available names pass through; foreign names
        # resolve exactly like _chat resolves them, so the trace records the
        # model that will really run (never a foreign name for Ollama output).
        return self._resolve_model(requested_model)

    def generate(
        self,
        model,
        messages,
        *,
        temperature=settings.default_temperature,
        max_tokens: int | None = settings.default_max_tokens,
    ) -> str:
        data = self._chat(
            model,
            messages,
            temperature=temperature,
            # Never omit: an omitted max_tokens lets server defaults rule.
            max_tokens=max_tokens if max_tokens is not None else settings.default_max_tokens,
        )
        self._record_usage(data)
        return self._content(data)

    def generate_stream(
        self,
        model,
        messages,
        *,
        temperature=settings.default_temperature,
        max_tokens: int | None = settings.default_max_tokens,
    ) -> Iterator[str]:
        # Reset first: usage is only meaningful if this stream reports it
        # (a stale value from a previous call would lie).
        self._last_usage = None
        think = ThinkFilter()
        for event in self._chat_stream(
            model,
            messages,
            temperature=temperature,
            # Never omit: an omitted max_tokens lets server defaults rule.
            max_tokens=max_tokens if max_tokens is not None else settings.default_max_tokens,
        ):
            if event.get("usage") is not None:
                self._record_usage(event)
            visible = think.feed(self._delta_content(event))
            if visible:
                yield visible
        # A failed stream raises out of the loop above, so the tail flush
        # only runs on success (no trailing partial on failure).
        tail = think.close()
        if tail:
            yield tail

    def generate_structured(
        self, model, messages, schema, *, temperature: float = 0.0
    ) -> dict[str, Any]:
        # Native /api/chat with the RAW schema as `format` — NOT the /v1
        # OpenAI wrapper. The compat layer's response_format mapping proved
        # lossy on minicpm5-2b (malformed plans: stray step_ids, empty inputs),
        # while native format went 4/4 VALID on the real planner prompt
        # (2026-09-13 probes). think=False explicit: this tag rejects
        # think:true ("does not support thinking").
        self.ensure_ready()
        payload: dict[str, Any] = {
            "model": self._resolve_model(model),
            "messages": messages,
            "stream": False,
            "format": schema,
            "options": {
                "temperature": temperature,
                "num_predict": settings.default_max_tokens,
            },
            "think": False,
        }
        try:
            response = self._client.post("/api/chat", json=payload)
        except httpx.RequestError as e:
            logger.error("Ollama request failed: %s", e)
            raise RuntimeError(f"Ollama connection failure: {e}") from e
        if response.status_code != 200:
            logger.error(
                "Ollama non-200 status=%s body=%s",
                response.status_code,
                response.text,
            )
            raise RuntimeError(
                f"Ollama error [status {response.status_code}]: {response.text}"
            )
        try:
            data = response.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Malformed Ollama response (not JSON): {e}") from e
        # Native usage counters differ from the OpenAI shape: map honestly,
        # never fabricate.
        prompt_count = data.get("prompt_eval_count")
        eval_count = data.get("eval_count")
        if prompt_count is not None or eval_count is not None:
            prompt_count = prompt_count or 0
            eval_count = eval_count or 0
            self._last_usage = {
                "input": prompt_count,
                "output": eval_count,
                "total": prompt_count + eval_count,
            }
        else:
            self._last_usage = None
        try:
            content = data["message"]["content"].strip()
        except (KeyError, AttributeError, TypeError) as e:
            logger.error("malformed Ollama response: %s", e)
            raise RuntimeError(
                f"Malformed response structure from Ollama: {e}"
            ) from e
        content = strip_think(content)
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Failed to parse structured output: {e}") from e

    def embed(self, model: str, text: str) -> list[float]:
        raise NotImplementedError(
            "Ollama provider is text-generation only; embeddings are not "
            "supported yet (Ollama can serve them via /api/embed if/when the "
            "RAG agent needs it — deliberately deferred)."
        )

    def generate_image(self, prompt: str) -> tuple[str, bytes]:
        """Image generation via the OpenAI-compatible surface (ADR-024).

        Attempts POST /v1/images/generations with OLLAMA_IMAGE_MODEL — the
        same compat surface every other method here speaks. An empty model id
        or a server without the endpoint fails honestly (never fabricated).
        """
        import base64

        model = settings.ollama_image_model.strip()
        if not model:
            raise NotImplementedError(
                "Ollama has no image model configured (OLLAMA_IMAGE_MODEL is empty)"
            )
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "response_format": "b64_json",
        }
        try:
            response = self._client.post("/v1/images/generations", json=payload)
        except httpx.RequestError as e:
            raise RuntimeError(f"Ollama image request failed: {e}") from e
        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama image error [status {response.status_code}]: {response.text}"
            )
        try:
            b64 = response.json()["data"][0]["b64_json"]
        except (KeyError, IndexError, AttributeError, TypeError, ValueError) as e:
            raise RuntimeError(f"Malformed Ollama image response: {e}") from e
        if not isinstance(b64, str) or not b64:
            raise RuntimeError("Malformed Ollama image response: empty b64_json")
        try:
            return "image/png", base64.b64decode(b64)
        except Exception as e:
            raise RuntimeError(f"Malformed Ollama image payload: {e}") from e

    def list_available_models(self) -> list[dict]:
        fallback = [
            {
                "id": settings.ollama_default_model,
                "display_name": settings.ollama_default_model,
            }
        ]
        try:
            response = self._client.get("/api/tags")
            if response.status_code != 200:
                logger.error(
                    "model listing non-200 status=%s body=%s",
                    response.status_code,
                    response.text,
                )
                return fallback
            data = response.json()
        except (httpx.RequestError, json.JSONDecodeError) as e:
            logger.error(
                "model listing failed, falling back to configured model: %s", e
            )
            return fallback
        models = [
            {"id": m.get("name"), "display_name": m.get("name")}
            for m in data.get("models", [])
            if m.get("name")
        ]
        if not models:
            models.append(fallback[0])
        return models
