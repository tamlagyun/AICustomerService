from dataclasses import dataclass
from enum import StrEnum
import json


class AgentAction(StrEnum):
    KNOWLEDGE_BASE = "knowledge_base"
    MYSQL_PLAYER_PROFILE = "mysql_player_profile"
    MYSQL_PLAYERS_LIST = "mysql_players_list"
    AVATAR_GENERATE = "avatar_generate"
    AMAP_PLACE_SEARCH = "amap_place_search"
    AMAP_GEO = "amap_geo"
    AMAP_ROUTE = "amap_route"
    AMAP_NAVIGATION = "amap_navigation"
    AMAP_WEATHER = "amap_weather"
    ASK_CLARIFICATION = "ask_clarification"
    HANDOFF = "handoff"
    DIRECT_ANSWER = "direct_answer"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class AgentDecision:
    action: AgentAction
    reason: str
    arguments: dict[str, object] | None = None
    final_task: str = ""
    direct_reply: str = ""


def parse_agent_decision(raw_content: str) -> AgentDecision:
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError:
        return AgentDecision(
            action=AgentAction.FALLBACK,
            reason="无法解析模型 JSON 决策。",
        )

    raw_action = payload.get("action")
    try:
        action = AgentAction(raw_action)
    except ValueError:
        return AgentDecision(
            action=AgentAction.FALLBACK,
            reason=f"不支持的模型动作：{raw_action}",
        )

    arguments = _parse_arguments(payload.get("arguments"))
    validation = _validate_arguments(action, arguments)
    if not validation.valid:
        return AgentDecision(
            action=AgentAction.FALLBACK,
            reason=f"工具参数不合法：{'; '.join(validation.errors)}",
        )

    return AgentDecision(
        action=action,
        reason=str(payload.get("reason", "")),
        arguments=validation.arguments,
        final_task=str(payload.get("final_task", "")),
        direct_reply=str(payload.get("direct_reply", "")),
    )


def _parse_arguments(raw_arguments: object) -> dict[str, object]:
    if isinstance(raw_arguments, dict):
        return raw_arguments
    return {}


def _validate_arguments(action: AgentAction, arguments: dict[str, object]):
    from app.tools.registry import validate_tool_arguments

    return validate_tool_arguments(action, arguments)
