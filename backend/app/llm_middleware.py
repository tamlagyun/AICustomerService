from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import logging
from typing import TypeVar

from app.agent.decision import AgentDecision
from app.agent.trace import AgentTrace, TraceEventType
from app.llm import LLMClientProtocol, LLMResponse
from app.llm_usage import LLMTokenUsage, LLMUsageSummary

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class LLMCallContext:
    provider: str = ""
    model: str = ""
    session_id: str = ""
    agent_trace: AgentTrace | None = None
    usage_summary: LLMUsageSummary | None = None
    input_token_price_per_1k: float = 0.0
    output_token_price_per_1k: float = 0.0

    def merged(self, override: LLMCallContext) -> LLMCallContext:
        return LLMCallContext(
            provider=override.provider or self.provider,
            model=override.model or self.model,
            session_id=override.session_id or self.session_id,
            agent_trace=override.agent_trace or self.agent_trace,
            usage_summary=override.usage_summary or self.usage_summary,
            input_token_price_per_1k=(
                override.input_token_price_per_1k
                if override.input_token_price_per_1k
                else self.input_token_price_per_1k
            ),
            output_token_price_per_1k=(
                override.output_token_price_per_1k
                if override.output_token_price_per_1k
                else self.output_token_price_per_1k
            ),
        )


class LLMClientMiddleware:
    def __init__(
        self,
        wrapped: LLMClientProtocol,
        context: LLMCallContext | None = None,
    ) -> None:
        self.wrapped = wrapped
        self.context = context or LLMCallContext()

    async def decide_action(self, messages: list[dict[str, str]]) -> AgentDecision:
        return await self._call("decide_action", messages, lambda: self.wrapped.decide_action(messages))

    async def generate_reply(self, messages: list[dict[str, str]]) -> LLMResponse:
        return await self._call("generate_reply", messages, lambda: self.wrapped.generate_reply(messages))

    async def stream_reply(self, messages: list[dict[str, str]]):
        started_at = self._record_started("stream_reply", messages)
        chunk_count = 0
        try:
            stream_reply = getattr(self.wrapped, "stream_reply", None)
            if stream_reply is None:
                raise AttributeError("wrapped LLM client does not support stream_reply")
            async for token in stream_reply(messages):
                chunk_count += 1
                yield token
        except Exception as exc:
            self._record_failed("stream_reply", started_at, exc)
            raise
        self._record_finished("stream_reply", started_at, {"chunk_count": chunk_count})

    async def _call(
        self,
        operation: str,
        messages: list[dict[str, str]],
        call: Callable[[], Awaitable[T]],
    ) -> T:
        started_at = self._record_started(operation, messages)
        try:
            result = await call()
        except Exception as exc:
            self._record_failed(operation, started_at, exc)
            raise
        self._record_usage(operation, result)
        self._record_finished(operation, started_at)
        return result

    def _record_started(self, operation: str, messages: list[dict[str, str]]) -> float | None:
        if self.context.agent_trace is None:
            return None
        return self.context.agent_trace.record_started(
            TraceEventType.LLM_STARTED,
            f"llm.{operation}",
            self._metadata(operation, messages),
        )

    def _record_finished(
        self,
        operation: str,
        started_at: float | None,
        extra_metadata: dict[str, object] | None = None,
    ) -> None:
        if self.context.agent_trace is None:
            return
        metadata = self._metadata(operation)
        metadata.update(extra_metadata or {})
        self.context.agent_trace.record_finished(
            TraceEventType.LLM_FINISHED,
            f"llm.{operation}",
            started_at,
            metadata,
        )

    def _record_failed(self, operation: str, started_at: float | None, exc: Exception) -> None:
        error = f"{type(exc).__name__}: {exc}"
        logger.exception(
            "LLM call failed; operation=%s provider=%s model=%s session_id=%s",
            operation,
            self._provider_for_log(),
            self._model_for_log(),
            self.context.session_id,
        )
        if self.context.agent_trace is None:
            return
        self.context.agent_trace.record_finished(
            TraceEventType.LLM_FINISHED,
            f"llm.{operation}",
            started_at,
            self._metadata(operation),
            error=error,
        )

    def _metadata(
        self,
        operation: str,
        messages: list[dict[str, str]] | None = None,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "operation": operation,
            "provider": self._provider_for_log(),
            "model": self._model_for_log(),
        }
        if self.context.session_id:
            metadata["session_id"] = self.context.session_id
        if messages is not None:
            metadata["message_count"] = len(messages)
        return metadata

    def _record_usage(self, operation: str, result: object) -> None:
        if self.context.usage_summary is None:
            return
        usage = _usage_from_result_or_client(result, self.wrapped)
        if usage is None:
            return
        self.context.usage_summary.add(
            provider=self._provider_for_log(),
            model=self._model_for_log(),
            operation=operation,
            usage=usage,
            input_token_price_per_1k=self.context.input_token_price_per_1k,
            output_token_price_per_1k=self.context.output_token_price_per_1k,
        )

    def _provider_for_log(self) -> str:
        return self.context.provider or "unknown"

    def _model_for_log(self) -> str:
        return self.context.model or "unknown"


def wrap_llm_client(
    client: LLMClientProtocol,
    context: LLMCallContext | None = None,
) -> LLMClientMiddleware:
    next_context = context or LLMCallContext()
    if isinstance(client, LLMClientMiddleware):
        return LLMClientMiddleware(client.wrapped, client.context.merged(next_context))
    return LLMClientMiddleware(client, next_context)


def _usage_from_result_or_client(result: object, client: LLMClientProtocol) -> LLMTokenUsage | None:
    if isinstance(result, LLMResponse):
        return result.usage

    usage = getattr(client, "last_usage", None)
    if isinstance(usage, LLMTokenUsage):
        return usage

    wrapped = getattr(client, "wrapped", None)
    if wrapped is not None:
        nested_usage = getattr(wrapped, "last_usage", None)
        if isinstance(nested_usage, LLMTokenUsage):
            return nested_usage

    return None
