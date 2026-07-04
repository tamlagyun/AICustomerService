from __future__ import annotations

import asyncio
import logging

from app.agent.decision import AgentDecision
from app.llm import LLMClientProtocol, LLMResponse

logger = logging.getLogger(__name__)


class StreamingLLMClient:
    def __init__(self, wrapped: LLMClientProtocol, token_queue: asyncio.Queue[str]) -> None:
        self.wrapped = wrapped
        self.token_queue = token_queue

    async def decide_action(self, messages: list[dict[str, str]]) -> AgentDecision:
        return await self.wrapped.decide_action(messages)

    async def generate_reply(self, messages: list[dict[str, str]]) -> LLMResponse:
        content = ""
        stream_reply = getattr(self.wrapped, "stream_reply", None)
        if stream_reply is None:
            return await self.wrapped.generate_reply(messages)

        try:
            async for token in stream_reply(messages):
                content += token
                await self.token_queue.put(token)
        except Exception:
            logger.exception("LLM streaming reply failed; falling back to non-streaming reply")
            return await self.wrapped.generate_reply(messages)

        return LLMResponse(content=content)

    async def stream_reply(self, messages: list[dict[str, str]]):
        async for token in self.wrapped.stream_reply(messages):
            yield token
