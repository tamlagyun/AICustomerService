from __future__ import annotations

import json
import logging

from app.agent.decision import AgentAction, AgentDecision
from app.agent.state import CustomerServiceState
from app.config import Settings, get_settings
from app.conversation_memory import ConversationMessage
from app.map_tools import MapToolResult
from app.prompt_registry import get_prompt_versions, load_prompt

logger = logging.getLogger("app.agent.customer_service")


def log_prompt_versions(session_id: str, settings: Settings) -> None:
    versions = get_prompt_versions(settings)
    logger.info(
        "Prompt versions selected; session_id=%s decision=%s planner=%s followup_decision=%s final_reply=%s",
        session_id,
        versions["decision"],
        versions["planner"],
        versions["followup_decision"],
        versions["final_reply"],
    )


def planner_messages(
    message: str,
    conversation_history: list[ConversationMessage],
) -> list[dict[str, str]]:
    versions = get_prompt_versions(get_settings())
    return [
        {
            "role": "system",
            "content": load_prompt("planner", versions["planner"]),
        },
        {
            "role": "user",
            "content": (
                f"历史对话（最近 10 条）：\n{format_conversation_history(conversation_history)}\n"
                f"当前玩家问题：{message}\n"
                "请输出执行计划 JSON。"
            ),
        },
    ]


def decision_messages(
    message: str,
    conversation_history: list[ConversationMessage],
) -> list[dict[str, str]]:
    versions = get_prompt_versions(get_settings())
    return [
        {
            "role": "system",
            "content": load_prompt("decision", versions["decision"]),
        },
        {
            "role": "user",
            "content": (
                f"历史对话（最近 10 条）：\n{format_conversation_history(conversation_history)}\n"
                f"当前玩家问题：{message}\n"
                "请结合历史对话和当前问题输出 JSON，例如："
                '{"action":"mysql_player_profile","arguments":{"player_id":"1"},'
                '"final_task":"根据玩家资料和 desc 字段分析总结玩家个性",'
                '"reason":"需要查询玩家资料后继续分析","direct_reply":""}'
                '；表格示例：{"action":"amap_place_search",'
                '"arguments":{"keywords":"旅游景点","city":"广州",'
                '"presentation":{"mode":"table","sort_by":"distance","sort_order":"asc"}},'
                '"final_task":"按距离整理地点结果","reason":"需要调用地图工具","direct_reply":""}'
            ),
        },
    ]


def followup_decision_messages(state: CustomerServiceState) -> list[dict[str, str]]:
    player_data_result = state.get("player_data_result")
    player_data = player_data_result.data if player_data_result is not None else {}
    versions = get_prompt_versions(get_settings())
    return [
        {
            "role": "system",
            "content": load_prompt("followup_decision", versions["followup_decision"]),
        },
        {
            "role": "user",
            "content": (
                f"玩家问题：{state.get('normalized_message', state.get('message', ''))}\n"
                f"已查询到的玩家资料：{json.dumps(player_data, ensure_ascii=False)}\n"
                "请输出 JSON，例如："
                '{"action":"amap_place_search","arguments":{"keywords":"旅游景点","city":"广州",'
                '"presentation":{"mode":"table"}},"final_task":"结合玩家资料和广州景点结果生成推荐",'
                '"reason":"需要真实景点数据","direct_reply":""}'
            ),
        },
    ]


def final_reply_messages(state: CustomerServiceState) -> list[dict[str, str]]:
    sources = state.get("sources", [])
    source_text = "\n".join(
        f"- {source.title}: {source.reference}" for source in sources
    ) or "无"
    decision = state.get("llm_decision")
    final_task = final_task_for_prompt(state, decision)
    tool_data = tool_data_for_prompt(state)
    conversation_history = format_conversation_history(state.get("conversation_history", []))
    table_instruction = (
        "结构化表格已经由后端生成，最终回复只需要说明重点和限制，不要再用 Markdown 或空格画表格。"
        if state.get("tables")
        else ""
    )
    versions = get_prompt_versions(get_settings())
    system_prompt = load_prompt("final_reply", versions["final_reply"])
    if table_instruction:
        system_prompt = f"{system_prompt}{table_instruction}"
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": (
                f"历史对话（最近 10 条）：\n{conversation_history}\n"
                f"玩家问题：{state.get('normalized_message', state.get('message', ''))}\n"
                f"最终任务：{final_task or '按玩家问题回复'}\n"
                f"工具结果：{state.get('reply', '')}\n"
                f"结构化工具数据：{tool_data}\n"
                f"来源：\n{source_text}\n"
                "请生成最终回复。"
            ),
        },
    ]


def should_use_llm_final_reply(state: CustomerServiceState) -> bool:
    decision = state.get("llm_decision")
    return (
        state.get("llm_client") is not None
        and decision is not None
        and decision.action != AgentAction.FALLBACK
    )


def format_conversation_history(messages: list[ConversationMessage]) -> str:
    if not messages:
        return "无"

    lines = []
    for message in messages:
        role_label = "玩家" if message.role == "user" else "客服"
        lines.append(f"{role_label}：{message.content}")
    return "\n".join(lines)


def final_task_for_prompt(
    state: CustomerServiceState,
    decision: AgentDecision | None,
) -> str:
    tasks = []
    if decision is not None and decision.final_task:
        tasks.append(decision.final_task)
    map_decision = state.get("map_decision")
    if map_decision is not None and map_decision.final_task:
        tasks.append(map_decision.final_task)
    return "；".join(dict.fromkeys(tasks))


def tool_data_for_prompt(state: CustomerServiceState) -> str:
    tools: list[dict[str, object]] = []
    player_data_result = state.get("player_data_result")
    if player_data_result is not None:
        tools.append(
            {
                "tool": tool_name_for_prompt(state),
                "status": player_data_result.status,
                "data": player_data_result.data,
            }
        )

    avatar_result = state.get("avatar_result")
    if avatar_result is not None:
        tools.append(
            {
                "tool": "avatar_generate",
                "status": avatar_result.status,
                "url": avatar_result.url,
                "alt": avatar_result.alt,
                "data": avatar_result.data,
            }
        )

    map_result = state.get("map_result")
    if map_result is not None:
        tools.append(
            {
                "tool": map_tool_name(map_result) or "amap_mcp",
                "status": map_result.status,
                "data": map_result.data,
            }
        )

    if not tools:
        return "{}"

    return json.dumps({"tools": tools}, ensure_ascii=False)


def tool_name_for_prompt(state: CustomerServiceState) -> str:
    decision = state.get("llm_decision")
    if decision is not None and decision.action == AgentAction.MYSQL_PLAYERS_LIST:
        return "mysql_players_list"
    if state.get("question_type") == "players_list":
        return "mysql_players_list"
    return "mysql_player_profile"


def map_tool_name(map_result: MapToolResult) -> str | None:
    if not map_result.data:
        return None
    tool_name = map_result.data.get("tool")
    if isinstance(tool_name, str) and tool_name.strip():
        return tool_name.strip()
    return None
