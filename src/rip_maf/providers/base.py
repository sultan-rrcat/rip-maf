"""ModelProvider interface.

This is the portability boundary (Charter NFR-7): everything above this layer
(Router, Builders, ReAct, Execution Engine, agents) talks ONLY to this interface.
Ollama is the sole backend (ADR-003 superseded); future providers implement it
behind this contract.

Operations needed across the system:
- generate:          plain text completion (Reasoning, Coding, Vision chat)
- generate_structured: JSON output matching a schema (Router intent, ReAct steps, rag query decomposition)
- embed:             vector embedding (RAG / retrieval, added at PM-5)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any


class ModelProvider(ABC):
    @abstractmethod
    def generate(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        """Return the model's text response to `messages` (OpenAI-style message list)."""

    def generate_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield the model's text response as incremental text chunks.

        Chunks are non-empty text fragments; their concatenation is the same
        answer `generate()` would return. Providers with native token
        streaming override this; the default is honest (one chunk), so every
        existing subclass keeps working unchanged.

        Concrete (not abstract) ON PURPOSE: test fakes and future providers
        that only implement generate() must keep instantiating.
        """
        yield self.generate(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )

    @abstractmethod
    def generate_structured(
        self,
        model: str,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        *,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Return a JSON object (as a Python dict) conforming to `schema`."""

    @abstractmethod
    def embed(self, model: str, text: str) -> list[float]:
        """Return the embedding vector for `text`."""

    @abstractmethod
    def list_available_models(self) -> list[dict]:
        """Every provider must answer 'Which models can you generate with?'"""

    def generate_image(self, prompt: str) -> tuple[str, bytes]:
        """Generate one image for `prompt`; return (mime_type, image_bytes).

        OPTIONAL capability (ADR-024): providers without an image model keep
        this default, which fails honestly. The image.generate tool binds the
        active provider and calls this — core code never names a provider.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no image-generation model configured"
        )

    def served_model(self, requested_model: str) -> str:
        """The model that will actually serve this request.

        Defaults to the requested model. Backends that ignore per-capability
        routing (single-model servers) override this so traces record what
        really ran, not what was asked for.
        """
        return requested_model
