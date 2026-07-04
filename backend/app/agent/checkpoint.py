from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from threading import RLock
from typing import Any

from app.agent.decision import AgentDecision
from app.agent.state import CustomerServiceState
from app.config import Settings
from app.logging_config import resolve_log_dir
from app.schemas import ChatResponse

logger = logging.getLogger(__name__)
_checkpoint_lock = RLock()


@dataclass(frozen=True)
class AgentCheckpoint:
    session_id: str
    player_id: str | None
    message: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": self.session_id,
            "player_id": self.player_id,
            "message": self.message,
            **self.payload,
        }


class CheckpointStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def append(self, checkpoint: AgentCheckpoint) -> Path:
        checkpoint_file = checkpoint_log_path(self.settings)
        if not self.settings.agent_checkpoint_enabled:
            return checkpoint_file

        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with _checkpoint_lock, checkpoint_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(checkpoint.to_dict(), ensure_ascii=False, default=str))
            file.write("\n")
        return checkpoint_file

    def list_recent(self, *, session_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        checkpoint_file = checkpoint_log_path(self.settings)
        if not checkpoint_file.exists():
            return []

        checkpoints: list[dict[str, Any]] = []
        with _checkpoint_lock, checkpoint_file.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if session_id and payload.get("session_id") != session_id:
                    continue
                checkpoints.append(payload)
        return checkpoints[-max(1, limit) :]


def write_chat_checkpoint_event(
    settings: Settings,
    *,
    session_id: str,
    player_id: str | None,
    message: str,
    response: ChatResponse,
    final_state: CustomerServiceState | None = None,
) -> Path:
    store = CheckpointStore(settings)
    return store.append(
        AgentCheckpoint(
            session_id=session_id,
            player_id=player_id,
            message=message,
            payload=_checkpoint_payload(response, final_state),
        )
    )


def checkpoint_log_path(settings: Settings) -> Path:
    return resolve_log_dir(settings.log_dir) / settings.agent_checkpoint_file


def _checkpoint_payload(
    response: ChatResponse,
    state: CustomerServiceState | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "reply": response.reply,
        "handoff": response.handoff,
        "llm_action": _action_value(state.get("llm_decision") if state else None),
        "plan_actions": _plan_actions(state),
        "completed_plan_steps": state.get("completed_plan_steps", []) if state else [],
        "tools": _checkpoint_tools(state),
        **_trace_summary(state),
        **_context_budget_summary(state),
        **_llm_usage_summary(state),
    }
    return payload


def _action_value(decision: AgentDecision | None) -> str | None:
    if decision is None:
        return None
    return str(decision.action)


def _plan_actions(state: CustomerServiceState | None) -> list[str]:
    if state is None:
        return []
    plan = state.get("agent_plan")
    if plan is None:
        return []
    return plan.actions()


def _checkpoint_tools(state: CustomerServiceState | None) -> list[dict[str, Any]]:
    if state is None:
        return []

    tools: list[dict[str, Any]] = []
    player_data_result = state.get("player_data_result")
    if player_data_result is not None:
        tools.append(
            {
                "tool": "player_data",
                "status": str(player_data_result.status),
                "summary": player_data_result.summary,
            }
        )

    map_result = state.get("map_result")
    if map_result is not None:
        tools.append(
            {
                "tool": "map",
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
            }
        )

    if state.get("knowledge_results"):
        tools.append({"tool": "knowledge_base", "status": "found"})

    return tools


def _trace_summary(state: CustomerServiceState | None) -> dict[str, Any]:
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


def _context_budget_summary(state: CustomerServiceState | None) -> dict[str, Any]:
    if state is None:
        return {
            "context_budget_max_tokens": 0,
            "context_estimated_tokens_before": 0,
            "context_estimated_tokens_after": 0,
            "context_truncated": False,
        }
    budget_result = state.get("context_budget_result")
    if budget_result is None:
        return {
            "context_budget_max_tokens": 0,
            "context_estimated_tokens_before": 0,
            "context_estimated_tokens_after": 0,
            "context_truncated": False,
        }
    return budget_result.to_audit_payload()


def _llm_usage_summary(state: CustomerServiceState | None) -> dict[str, Any]:
    if state is None:
        return {
            "llm_prompt_tokens": 0,
            "llm_completion_tokens": 0,
            "llm_total_tokens": 0,
            "llm_estimated_cost": 0.0,
        }
    usage_summary = state.get("llm_usage_summary")
    if usage_summary is None:
        return {
            "llm_prompt_tokens": 0,
            "llm_completion_tokens": 0,
            "llm_total_tokens": 0,
            "llm_estimated_cost": 0.0,
        }
    return usage_summary.to_audit_payload()
