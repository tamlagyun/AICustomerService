from dataclasses import dataclass
import json
from typing import Protocol

import httpx

from app.agent.decision import AgentDecision, parse_agent_decision
from app.config import get_settings
from app.llm_usage import LLMTokenUsage, parse_llm_token_usage


@dataclass(frozen=True)
class LLMResponse:
    content: str
    usage: LLMTokenUsage | None = None


@dataclass(frozen=True)
class LLMProviderConfig:
    provider: str
    base_url: str
    api_key: str
    model: str


class LLMClientProtocol(Protocol):
    async def decide_action(self, messages: list[dict[str, str]]) -> AgentDecision:
        raise NotImplementedError

    async def generate_reply(self, messages: list[dict[str, str]]) -> LLMResponse:
        raise NotImplementedError

    async def stream_reply(self, messages: list[dict[str, str]]):
        raise NotImplementedError


class OpenAICompatibleLLMClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 20,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.last_usage: LLMTokenUsage | None = None

    async def decide_action(self, messages: list[dict[str, str]]) -> AgentDecision:
        response = await self._chat(messages)
        return parse_agent_decision(response.content)

    async def generate_reply(self, messages: list[dict[str, str]]) -> LLMResponse:
        return await self._chat(messages)

    async def stream_reply(self, messages: list[dict[str, str]]):
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.2,
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    delta = extract_stream_delta(line)
                    if delta:
                        yield delta

    async def _chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.2,
                },
            )
            response.raise_for_status()
            payload = response.json()

        content = payload["choices"][0]["message"]["content"]
        self.last_usage = parse_llm_token_usage(payload.get("usage"))
        return LLMResponse(content=content, usage=self.last_usage)


def extract_stream_delta(line: str) -> str | None:
    if not line.startswith("data:"):
        return None

    raw_data = line.removeprefix("data:").strip()
    if not raw_data or raw_data == "[DONE]":
        return None

    try:
        payload = json.loads(raw_data)
    except json.JSONDecodeError:
        return None

    choices = payload.get("choices", [])
    if not choices:
        return None

    delta = choices[0].get("delta", {})
    content = delta.get("content")
    if isinstance(content, str):
        return content
    return None


def build_llm_client(model_provider: str | None = None) -> LLMClientProtocol | None:
    settings = get_settings()
    if not settings.llm_enabled:
        return None

    provider_config = _provider_config(settings, _select_provider(settings, model_provider))
    if not provider_config.api_key:
        return None

    return OpenAICompatibleLLMClient(
        base_url=provider_config.base_url,
        api_key=provider_config.api_key,
        model=provider_config.model,
        timeout_seconds=settings.llm_timeout_seconds,
    )


def _select_provider(settings, requested_provider: str | None) -> str:
    allowed_providers = _split_csv(settings.llm_allowed_providers)
    default_provider = settings.llm_default_provider.strip().lower() or settings.llm_provider
    default_provider = default_provider.strip().lower()
    requested = (requested_provider or default_provider).strip().lower()

    if requested in allowed_providers:
        return requested
    if default_provider in allowed_providers:
        return default_provider
    return allowed_providers[0] if allowed_providers else "deepseek"


def _provider_config(settings, provider: str) -> LLMProviderConfig:
    if provider == "qwen":
        return LLMProviderConfig(
            provider="qwen",
            base_url=settings.qwen_base_url,
            api_key=settings.qwen_api_key,
            model=settings.qwen_model,
        )

    legacy_matches_deepseek = settings.llm_provider.strip().lower() == "deepseek"
    return LLMProviderConfig(
        provider="deepseek",
        base_url=settings.deepseek_base_url or (settings.llm_base_url if legacy_matches_deepseek else ""),
        api_key=settings.deepseek_api_key or (settings.llm_api_key if legacy_matches_deepseek else ""),
        model=settings.deepseek_model or (settings.llm_model if legacy_matches_deepseek else ""),
    )


def _split_csv(raw_value: str) -> list[str]:
    return [item.strip().lower() for item in raw_value.split(",") if item.strip()]
