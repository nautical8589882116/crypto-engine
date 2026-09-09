"""Unified LLM facade over OpenAI and Anthropic (same pattern as the parent)."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Protocol

from quant_crypto.config import Settings


class LLMClient(Protocol):
    def complete(self, system: str, messages: list[dict[str, str]]) -> str: ...

    def complete_json(self, system: str, messages: list[dict[str, str]]) -> dict[str, Any]: ...


class _BaseLLM(ABC):
    def __init__(self, settings: Settings, max_retries: int = 3, timeout: float = 30.0, http_client: Any = None):
        self.settings = settings
        self.max_retries = max_retries
        self.timeout = timeout
        self._http_client = http_client

    @abstractmethod
    def _raw_complete(self, system: str, messages: list[dict[str, str]]) -> str:
        """Provider-specific completion call."""

    def complete(self, system: str, messages: list[dict[str, str]]) -> str:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._raw_complete(system, messages)
            except ValueError:
                # Configuration errors (e.g. missing API key) are not transient.
                raise
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2**attempt)
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts") from last_err

    def complete_json(self, system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        return _parse_json(self.complete(system, messages))


def _parse_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from an LLM response, tolerating code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json")
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    return json.loads(cleaned[start : end + 1])


class OpenAIClient(_BaseLLM):
    def _raw_complete(self, system: str, messages: list[dict[str, str]]) -> str:
        from openai import OpenAI

        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set")
        client = OpenAI(api_key=self.settings.openai_api_key, timeout=self.timeout, http_client=self._http_client)
        msgs: list[dict[str, str]] = [{"role": "system", "content": system}, *messages]
        resp = client.chat.completions.create(model=self.settings.openai_model, messages=msgs)  # type: ignore[arg-type]
        return resp.choices[0].message.content or ""


class AnthropicClient(_BaseLLM):
    def _raw_complete(self, system: str, messages: list[dict[str, str]]) -> str:
        from anthropic import Anthropic

        if not self.settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        client = Anthropic(
            api_key=self.settings.anthropic_api_key,
            timeout=self.timeout,
            http_client=self._http_client,
            default_headers=(
                {"anthropic-workspace-id": self.settings.anthropic_workspace_id}
                if self.settings.anthropic_workspace_id
                else None
            ),
        )
        msgs = [m for m in messages if m.get("role") != "system"]
        resp = client.messages.create(model=self.settings.anthropic_model, system=system, messages=msgs, max_tokens=1024)
        return "".join(b.text for b in resp.content if hasattr(b, "text"))


def get_client(settings: Settings, provider: str | None = None) -> LLMClient:
    provider = provider or "openai"
    if provider == "anthropic":
        return AnthropicClient(settings)
    return OpenAIClient(settings)
