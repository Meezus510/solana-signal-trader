"""
Thin JSON-response adapters for different LLM providers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class OpenAICompatibleJSONProvider:
    api_key_env: str
    base_url_env: str | None = None
    default_base_url: str | None = None

    def generate_json(self, prompt: str, *, model: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is not installed") from exc

        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            raise ValueError(f"{self.api_key_env} not set")

        kwargs = {"api_key": api_key}
        base_url = os.getenv(self.base_url_env, "").strip() if self.base_url_env else ""
        if not base_url:
            base_url = self.default_base_url or ""
        if base_url:
            kwargs["base_url"] = base_url

        client = OpenAI(**kwargs)
        response = client.responses.create(
            model=model,
            input=prompt,
            text={"format": {"type": "json_object"}},
        )
        return response.output_text.strip()


@dataclass(frozen=True)
class AnthropicJSONProvider:
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_tokens: int = 800

    def generate_json(self, prompt: str, *, model: str) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("anthropic package is not installed") from exc

        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            raise ValueError(f"{self.api_key_env} not set")

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()


class OpenAIJSONProvider(OpenAICompatibleJSONProvider):
    def __init__(self) -> None:
        super().__init__(api_key_env="OPENAI_API_KEY")


class DeepSeekJSONProvider(OpenAICompatibleJSONProvider):
    def __init__(self) -> None:
        super().__init__(
            api_key_env="DEEPSEEK_API_KEY",
            base_url_env="DEEPSEEK_BASE_URL",
            default_base_url="https://api.deepseek.com",
        )
