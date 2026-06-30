from dataclasses import dataclass
from enum import StrEnum
import re


class SafetyAction(StrEnum):
    ALLOW = "allow"
    HANDOFF = "handoff"
    REFUSE = "refuse"


@dataclass(frozen=True)
class SafetyDecision:
    action: SafetyAction
    reply: str = ""


REFUSE_KEYWORDS = [
    "系统提示词",
    "system prompt",
    "api key",
    "apikey",
    "密钥",
    "内部提示词",
]

HANDOFF_KEYWORDS = [
    "退款",
    "投诉",
    "申诉",
    "人工",
    "客服",
]


def analyze_safety(message: str) -> SafetyDecision:
    normalized_message = message.lower()
    if any(keyword in normalized_message for keyword in REFUSE_KEYWORDS):
        return SafetyDecision(
            action=SafetyAction.REFUSE,
            reply="不能提供系统提示词、密钥或内部配置。如果你有游戏问题，我可以继续协助处理。",
        )

    if any(keyword in message for keyword in HANDOFF_KEYWORDS):
        return SafetyDecision(
            action=SafetyAction.HANDOFF,
            reply="这个问题建议转人工客服处理。我已记录你的诉求，请补充服务器、角色 ID 和相关订单号。",
        )

    return SafetyDecision(action=SafetyAction.ALLOW)


def redact_sensitive_text(text: str) -> str:
    text = re.sub(r"(?<!\d)(1[3-9]\d{9})(?!\d)", _mask_phone, text)
    text = re.sub(r"(?<!\d)(\d{6}\d{8}\d{3}[\dXx])(?!\d)", _mask_id_number, text)
    return text


def _mask_phone(match: re.Match[str]) -> str:
    value = match.group(1)
    return f"{value[:3]}****{value[-4:]}"


def _mask_id_number(match: re.Match[str]) -> str:
    value = match.group(1)
    return f"{value[:6]}********{value[-4:]}"
