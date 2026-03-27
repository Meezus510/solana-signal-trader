"""
Thin JSON-response adapters for different LLM providers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()

_ENV_FALLBACKS = {
    "OPENAI_API_KEY": ("OPEN_AI_API_KEY",),
}


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
            for alias in _ENV_FALLBACKS.get(self.api_key_env, ()):
                api_key = os.getenv(alias, "").strip()
                if api_key:
                    break
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
    max_tokens: int = 1500

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
            system="You are a trading strategy configuration assistant. Respond with valid JSON only — no explanation, no markdown, no preamble. Start your response with { and end with }.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text.strip()
        # Extract JSON if model added any preamble
        start = text.find("{")
        end = text.rfind("}") + 1
        return text[start:end] if start != -1 and end > start else text


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
        # DeepSeek doesn't support responses.create(), use chat.completions
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a trading strategy configuration assistant. Respond with valid JSON only — no explanation, no markdown, no preamble. Start your response with { and end with }."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1500,
        )
        text = response.choices[0].message.content.strip()
        # Extract JSON if model added any preamble
        start = text.find("{")
        end = text.rfind("}") + 1
        return text[start:end] if start != -1 and end > start else text
