from __future__ import annotations

import logging

from app.agent.decision import AgentDecision
from app.agent.prompting import map_tool_name, tool_name_for_prompt
from app.agent.state import CustomerServiceState
from app.agent_audit import write_agent_audit_event
from app.config import Settings
from app.schemas import ChatResponse

logger = logging.getLogger(__name__)


def write_chat_audit_event(
    settings: Settings,
    *,
    session_id: str,
    player_id: str | None,
    message: str,
    response: ChatResponse,
    final_state: CustomerServiceState | None = None,
) -> None:
    try:
        write_agent_audit_event(
            settings,
            {
                "event_type": "chat_completed",
                "session_id": session_id,
                "player_id": player_id,
                "message": message,
                "reply": response.reply,
                "handoff": response.handoff,
                "knowledge_source": final_state.get("knowledge_source") if final_state else None,
                "use_planner": bool(final_state.get("use_planner")) if final_state else False,
                "llm_action": _audit_action(final_state.get("llm_decision") if final_state else None),
                "map_action": _audit_action(final_state.get("map_decision") if final_state else None),
                "plan_actions": _audit_plan_actions(final_state),
                "completed_plan_steps": final_state.get("completed_plan_steps", [])
                if final_state
                else [],
                "planner_fallback_reason": final_state.get("planner_fallback_reason")
                if final_state
                else None,
                "sources": [source.model_dump() for source in response.sources],
                "images": [image.model_dump() for image in response.images],
                "tables": [
                    {
                        "title": table.title,
                        "row_count": len(table.rows),
                        "columns": [column.model_dump() for column in table.columns],
                    }
                    for table in response.tables
                ],
                "tools": _audit_tools(final_state),
                **_audit_trace_summary(final_state),
            },
        )
    except Exception:
        logger.exception("Agent audit logging failed; session_id=%s", session_id)


def _audit_action(decision: AgentDecision | None) -> str | None:
    if decision is None:
        return None
    return str(decision.action)


def _audit_plan_actions(state: CustomerServiceState | None) -> list[str]:
    if state is None:
        return []
    plan = state.get("agent_plan")
    if plan is None:
        return []
    return plan.actions()


def _audit_tools(state: CustomerServiceState | None) -> list[dict[str, object]]:
    if state is None:
        return []

    tools: list[dict[str, object]] = []
    player_data_result = state.get("player_data_result")
    if player_data_result is not None:
        tools.append(
            {
                "tool": tool_name_for_prompt(state),
                "status": str(player_data_result.status),
                "summary": player_data_result.summary,
            }
        )

    map_result = state.get("map_result")
    if map_result is not None:
        tools.append(
            {
                "tool": map_tool_name(map_result) or "amap_mcp",
                "status": str(map_result.status),
                "summary": map_result.summary,
            }
        )

    avatar_result = state.get("avatar_result")
    if avatar_result is not None:
        tools.append(
            {
                "tool": "avatar_generate",
                "status": str(avatar_result.status),
                "summary": avatar_result.summary,
                "url": avatar_result.url,
            }
        )

    knowledge_results = state.get("knowledge_results", [])
    if knowledge_results:
        tools.append(
            {
                "tool": "knowledge_base",
                "status": "found",
                "count": len(knowledge_results),
                "sources": [chunk.reference for chunk in knowledge_results],
            }
        )

    return tools


def _audit_trace_summary(state: CustomerServiceState | None) -> dict[str, object]:
    if state is None:
        return {
            "trace_event_count": 0,
            "trace_errors": [],
            "trace_duration_ms": 0,
        }
    trace = state.get("agent_trace")
    if trace is None:
        return {
            "trace_event_count": 0,
            "trace_errors": [],
            "trace_duration_ms": 0,
        }
    return trace.summary()
