from __future__ import annotations

from dataclasses import dataclass


Message = dict[str, str]


@dataclass(frozen=True)
class ContextBudget:
    max_tokens: int = 8000
    reserved_reply_tokens: int = 1000

    @property
    def input_token_budget(self) -> int:
        return max(1, self.max_tokens - self.reserved_reply_tokens)


@dataclass(frozen=True)
class ContextBudgetResult:
    messages: list[Message]
    estimated_tokens_before: int
    estimated_tokens_after: int
    max_tokens: int
    reserved_reply_tokens: int
    truncated: bool

    def to_audit_payload(self) -> dict[str, object]:
        return {
            "context_budget_max_tokens": self.max_tokens,
            "context_estimated_tokens_before": self.estimated_tokens_before,
            "context_estimated_tokens_after": self.estimated_tokens_after,
            "context_truncated": self.truncated,
        }


def apply_context_budget(
    messages: list[Message],
    budget: ContextBudget,
    *,
    protected_text: str = "",
    priority_markers: list[str] | None = None,
) -> ContextBudgetResult:
    estimated_before = estimate_context_tokens(messages)
    if estimated_before <= budget.input_token_budget:
        copied_messages = _copy_messages(messages)
        return ContextBudgetResult(
            messages=copied_messages,
            estimated_tokens_before=estimated_before,
            estimated_tokens_after=estimated_before,
            max_tokens=budget.max_tokens,
            reserved_reply_tokens=budget.reserved_reply_tokens,
            truncated=False,
        )

    priority_markers = priority_markers or []
    trimmed_messages = _copy_messages(messages)
    _drop_history_blocks(trimmed_messages)

    if estimate_context_tokens(trimmed_messages) > budget.input_token_budget:
        _truncate_user_messages(
            trimmed_messages,
            budget.input_token_budget,
            protected_text=protected_text,
            priority_markers=priority_markers,
        )

    estimated_after = estimate_context_tokens(trimmed_messages)
    return ContextBudgetResult(
        messages=trimmed_messages,
        estimated_tokens_before=estimated_before,
        estimated_tokens_after=estimated_after,
        max_tokens=budget.max_tokens,
        reserved_reply_tokens=budget.reserved_reply_tokens,
        truncated=trimmed_messages != messages or estimated_after < estimated_before,
    )


def estimate_context_tokens(messages: list[Message]) -> int:
    return sum(len(message.get("content", "")) for message in messages)


def empty_context_budget_audit_payload() -> dict[str, object]:
    return {
        "context_budget_max_tokens": 0,
        "context_estimated_tokens_before": 0,
        "context_estimated_tokens_after": 0,
        "context_truncated": False,
    }


def _copy_messages(messages: list[Message]) -> list[Message]:
    return [dict(message) for message in messages]


def _drop_history_blocks(messages: list[Message]) -> None:
    for message in messages:
        content = message.get("content", "")
        if "历史对话" not in content:
            continue
        message["content"] = _drop_history_block(content)


def _drop_history_block(content: str) -> str:
    lines = content.splitlines()
    output: list[str] = []
    skipping_history = False
    for line in lines:
        if line.startswith("历史对话"):
            output.append(line)
            output.append("（因上下文预算限制，较早历史已省略）")
            skipping_history = True
            continue
        if skipping_history and not _starts_preserved_section(line):
            continue
        skipping_history = False
        output.append(line)
    return "\n".join(output)


def _starts_preserved_section(line: str) -> bool:
    preserved_prefixes = (
        "当前玩家问题：",
        "玩家问题：",
        "最终任务：",
        "工具结果：",
        "结构化工具数据：",
        "来源：",
        "请",
    )
    return line.startswith(preserved_prefixes)


def _truncate_user_messages(
    messages: list[Message],
    token_budget: int,
    *,
    protected_text: str,
    priority_markers: list[str],
) -> None:
    system_tokens = sum(
        len(message.get("content", ""))
        for message in messages
        if message.get("role") == "system"
    )
    user_budget = max(1, token_budget - system_tokens)
    user_messages = [message for message in messages if message.get("role") != "system"]
    if not user_messages:
        return

    per_message_budget = max(1, user_budget // len(user_messages))
    for message in user_messages:
        content = message.get("content", "")
        if len(content) <= per_message_budget:
            continue
        message["content"] = _truncate_content(
            content,
            per_message_budget,
            protected_text=protected_text,
            priority_markers=priority_markers,
        )


def _truncate_content(
    content: str,
    limit: int,
    *,
    protected_text: str,
    priority_markers: list[str],
) -> str:
    required_lines = _required_lines(content, protected_text, priority_markers)
    required_text = "\n".join(required_lines)
    if required_text and len(required_text) <= limit:
        return required_text

    if required_text:
        return required_text[:limit]

    if protected_text and protected_text in content:
        return _slice_around(content, protected_text, limit)

    return content[-limit:]


def _required_lines(
    content: str,
    protected_text: str,
    priority_markers: list[str],
) -> list[str]:
    required: list[str] = []
    for line in content.splitlines():
        if protected_text and protected_text in line:
            required.append(line)
            continue
        if any(marker in line for marker in priority_markers):
            required.append(line)
    return list(dict.fromkeys(required))


def _slice_around(content: str, protected_text: str, limit: int) -> str:
    index = content.find(protected_text)
    if index < 0:
        return content[-limit:]
    start = max(0, index - max(0, (limit - len(protected_text)) // 2))
    end = min(len(content), start + limit)
    return content[start:end]
