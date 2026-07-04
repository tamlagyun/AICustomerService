import json

import pytest

from app.agent.audit import write_chat_audit_event
from app.config import Settings
from app.llm_usage import (
    LLMTokenUsage,
    LLMUsageSummary,
    parse_llm_token_usage,
)
from app.schemas import ChatResponse


def test_parse_llm_token_usage_from_openai_compatible_payload() -> None:
    usage = parse_llm_token_usage(
        {
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "total_tokens": 20,
        }
    )

    assert usage == LLMTokenUsage(
        prompt_tokens=12,
        completion_tokens=8,
        total_tokens=20,
    )


def test_usage_summary_accumulates_tokens_and_estimated_cost() -> None:
    summary = LLMUsageSummary()

    summary.add(
        provider="deepseek",
        model="deepseek-chat",
        operation="generate_reply",
        usage=LLMTokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
        input_token_price_per_1k=0.001,
        output_token_price_per_1k=0.002,
    )
    summary.add(
        provider="deepseek",
        model="deepseek-chat",
        operation="decide_action",
        usage=LLMTokenUsage(prompt_tokens=2000, completion_tokens=100, total_tokens=2100),
        input_token_price_per_1k=0.001,
        output_token_price_per_1k=0.002,
    )

    payload = summary.to_audit_payload()
    assert payload["llm_prompt_tokens"] == 3000
    assert payload["llm_completion_tokens"] == 600
    assert payload["llm_total_tokens"] == 3600
    assert payload["llm_estimated_cost"] == pytest.approx(0.0042)


def test_chat_audit_event_includes_llm_usage_summary(tmp_path) -> None:
    summary = LLMUsageSummary()
    summary.add(
        provider="qwen",
        model="qwen-plus",
        operation="generate_reply",
        usage=LLMTokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        input_token_price_per_1k=0.01,
        output_token_price_per_1k=0.02,
    )

    write_chat_audit_event(
        settings=Settings(
            log_dir=str(tmp_path),
            agent_audit_log_enabled=True,
        ),
        session_id="usage-session",
        player_id=None,
        message="你好",
        response=ChatResponse(reply="你好"),
        final_state={"llm_usage_summary": summary},
    )

    audit_file = tmp_path / "agent_audit.jsonl"
    payload = json.loads(audit_file.read_text(encoding="utf-8").splitlines()[-1])
    assert payload["llm_prompt_tokens"] == 100
    assert payload["llm_completion_tokens"] == 50
    assert payload["llm_total_tokens"] == 150
    assert payload["llm_estimated_cost"] == pytest.approx(0.002)
