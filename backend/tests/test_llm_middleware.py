import logging

import pytest

from app.agent.decision import AgentAction, AgentDecision
from app.agent.trace import AgentTrace, TraceEventType
from app.llm import LLMResponse
from app.llm_middleware import LLMCallContext, LLMClientMiddleware, wrap_llm_client


class SuccessfulLLMClient:
    async def decide_action(self, messages: list[dict[str, str]]) -> AgentDecision:
        return AgentDecision(action=AgentAction.DIRECT_ANSWER, reason="test")

    async def generate_reply(self, messages: list[dict[str, str]]) -> LLMResponse:
        return LLMResponse(content="ok")


class FailingLLMClient:
    async def decide_action(self, messages: list[dict[str, str]]) -> AgentDecision:
        raise TimeoutError("decision timeout")

    async def generate_reply(self, messages: list[dict[str, str]]) -> LLMResponse:
        raise TimeoutError("reply timeout")


async def test_llm_middleware_records_trace_for_successful_generate_reply() -> None:
    trace = AgentTrace()
    client = LLMClientMiddleware(
        SuccessfulLLMClient(),
        LLMCallContext(provider="deepseek", model="deepseek-chat", agent_trace=trace),
    )

    response = await client.generate_reply([{"role": "user", "content": "你好"}])

    assert response.content == "ok"
    assert [event.event_type for event in trace.events] == [
        TraceEventType.LLM_STARTED,
        TraceEventType.LLM_FINISHED,
    ]
    assert [event.name for event in trace.events] == [
        "llm.generate_reply",
        "llm.generate_reply",
    ]
    assert trace.events[1].metadata["provider"] == "deepseek"
    assert trace.events[1].metadata["model"] == "deepseek-chat"


async def test_llm_middleware_logs_and_reraises_failures(caplog) -> None:
    trace = AgentTrace()
    client = LLMClientMiddleware(
        FailingLLMClient(),
        LLMCallContext(provider="qwen", model="qwen-plus", agent_trace=trace),
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(TimeoutError):
            await client.decide_action([{"role": "user", "content": "查询资料"}])

    assert "LLM call failed; operation=decide_action provider=qwen model=qwen-plus" in caplog.text
    assert [event.event_type for event in trace.events] == [
        TraceEventType.LLM_STARTED,
        TraceEventType.LLM_FINISHED,
    ]
    assert trace.events[1].error == "TimeoutError: decision timeout"


def test_wrap_llm_client_reuses_existing_middleware_with_new_context() -> None:
    trace = AgentTrace()
    original = LLMClientMiddleware(
        SuccessfulLLMClient(),
        LLMCallContext(provider="deepseek", model="deepseek-chat"),
    )

    wrapped = wrap_llm_client(
        original,
        LLMCallContext(session_id="session-1", agent_trace=trace),
    )

    assert isinstance(wrapped, LLMClientMiddleware)
    assert wrapped is not original
    assert wrapped.context.provider == "deepseek"
    assert wrapped.context.model == "deepseek-chat"
    assert wrapped.context.session_id == "session-1"
    assert wrapped.context.agent_trace is trace
