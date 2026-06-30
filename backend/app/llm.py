from dataclasses import dataclass
import json
from typing import Protocol

import httpx

from app.agent.decision import AgentDecision, parse_agent_decision
from app.config import get_settings


@dataclass(frozen=True)
class LLMResponse:
    content: str


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
        return LLMResponse(content=content)


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


def build_llm_client() -> LLMClientProtocol | None:
    settings = get_settings()
    if not settings.llm_enabled or not settings.llm_api_key:
        return None

    return OpenAICompatibleLLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )
