import json

from app.agent.audit import write_chat_audit_event
from app.agent.context_budget import (
    ContextBudget,
    ContextBudgetResult,
    apply_context_budget,
    estimate_context_tokens,
)
from app.schemas import ChatResponse
from app.config import Settings


def test_context_budget_keeps_messages_when_under_budget() -> None:
    messages = [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "当前玩家问题：你好"},
    ]

    result = apply_context_budget(
        messages,
        ContextBudget(max_tokens=100, reserved_reply_tokens=10),
        protected_text="你好",
    )

    assert result.messages == messages
    assert result.truncated is False
    assert result.estimated_tokens_before == estimate_context_tokens(messages)
    assert result.estimated_tokens_after == estimate_context_tokens(messages)


def test_context_budget_truncates_old_history_before_current_question() -> None:
    messages = [
        {"role": "system", "content": "系统提示"},
        {
            "role": "user",
            "content": (
                "历史对话（最近 10 条）：\n"
                + "玩家：很早以前的问题" * 20
                + "\n当前玩家问题：请查询我的资料\n"
                + "请输出 JSON。"
            ),
        },
    ]

    result = apply_context_budget(
        messages,
        ContextBudget(max_tokens=80, reserved_reply_tokens=10),
        protected_text="请查询我的资料",
    )

    assert result.truncated is True
    assert "请查询我的资料" in result.messages[-1]["content"]
    assert "很早以前的问题" not in result.messages[-1]["content"]
    assert result.estimated_tokens_after <= 70


def test_context_budget_preserves_tool_result_over_history() -> None:
    tool_result = "工具结果：玩家资料：喜欢研究机制"
    messages = [
        {"role": "system", "content": "系统提示"},
        {
            "role": "user",
            "content": (
                "历史对话（最近 10 条）：\n"
                + "玩家：旧对话" * 30
                + "\n玩家问题：根据资料分析个性\n"
                + "最终任务：总结玩家个性\n"
                + f"{tool_result}\n"
                + "请生成最终回复。"
            ),
        },
    ]

    result = apply_context_budget(
        messages,
        ContextBudget(max_tokens=95, reserved_reply_tokens=10),
        protected_text="根据资料分析个性",
        priority_markers=["工具结果："],
    )

    assert result.truncated is True
    assert "根据资料分析个性" in result.messages[-1]["content"]
    assert tool_result in result.messages[-1]["content"]
    assert "旧对话" not in result.messages[-1]["content"]


def test_chat_audit_event_includes_context_budget_summary(tmp_path) -> None:
    budget_result = ContextBudgetResult(
        messages=[{"role": "user", "content": "当前玩家问题：你好"}],
        estimated_tokens_before=120,
        estimated_tokens_after=80,
        max_tokens=100,
        reserved_reply_tokens=20,
        truncated=True,
    )

    write_chat_audit_event(
        Settings(log_dir=str(tmp_path), agent_audit_log_enabled=True),
        session_id="budget-session",
        player_id=None,
        message="你好",
        response=ChatResponse(reply="你好"),
        final_state={"context_budget_result": budget_result},
    )

    audit_file = tmp_path / "agent_audit.jsonl"
    payload = json.loads(audit_file.read_text(encoding="utf-8").splitlines()[-1])
    assert payload["context_budget_max_tokens"] == 100
    assert payload["context_estimated_tokens_before"] == 120
    assert payload["context_estimated_tokens_after"] == 80
    assert payload["context_truncated"] is True
