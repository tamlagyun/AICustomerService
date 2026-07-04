from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from app.agent.decision import AgentAction, AgentDecision
from app.agent.map_agent import MapAgentResult, run_map_agent
from app.agent.trace import AgentTrace, TraceEventType
from app.config import Settings
from app.player_data import PlayerDataResult, build_player_data_tools
from app.tools.registry import (
    ToolCategory,
    ToolDefinition,
    get_tool_by_action,
    missing_tool_dependencies,
    validate_tool_arguments,
)

StatusEmitter = Callable[[str], None]
MapAgentRunner = Callable[
    [AgentDecision],
    Awaitable[MapAgentResult],
]


class PlayerDataToolsProtocol(Protocol):
    def get_player_profile(self, player_id: str | None) -> PlayerDataResult:
        raise NotImplementedError

    def get_players(self, limit: int = 100) -> PlayerDataResult:
        raise NotImplementedError


class ToolExecutionStatus(StrEnum):
    SUCCESS = "success"
    INVALID_ARGUMENTS = "invalid_arguments"
    MISSING_DEPENDENCY = "missing_dependency"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


@dataclass(frozen=True)
class ToolExecutionContext:
    settings: Settings
    message: str = ""
    emit_status: StatusEmitter | None = None
    player_data_tools_factory: Callable[[], PlayerDataToolsProtocol] | None = None
    map_agent_runner: Callable[..., Awaitable[MapAgentResult]] | None = None
    enforce_dependencies: bool = True
    agent_trace: AgentTrace | None = None


@dataclass(frozen=True)
class ToolExecutionResult:
    status: ToolExecutionStatus
    action: AgentAction
    tool_name: str | None
    arguments: dict[str, object]
    output: object | None = None
    error: str = ""


async def execute_tool_action(
    decision: AgentDecision,
    context: ToolExecutionContext,
) -> ToolExecutionResult:
    tool = get_tool_by_action(decision.action)
    if tool is None:
        return ToolExecutionResult(
            status=ToolExecutionStatus.UNSUPPORTED,
            action=decision.action,
            tool_name=None,
            arguments=decision.arguments or {},
            error=f"Unsupported tool action: {decision.action}",
        )

    trace_started_at = _record_tool_started(context, tool, decision)
    validation = validate_tool_arguments(decision.action, decision.arguments)
    if not validation.valid:
        error = "; ".join(validation.errors)
        _record_tool_failed(context, tool, trace_started_at, error)
        return ToolExecutionResult(
            status=ToolExecutionStatus.INVALID_ARGUMENTS,
            action=decision.action,
            tool_name=tool.name,
            arguments=validation.arguments,
            error=error,
        )

    if context.enforce_dependencies:
        missing_dependencies = missing_tool_dependencies(context.settings, tool)
        if missing_dependencies:
            error = "Missing tool dependencies: " + ", ".join(
                str(dependency) for dependency in missing_dependencies
            )
            _record_tool_failed(context, tool, trace_started_at, error)
            return ToolExecutionResult(
                status=ToolExecutionStatus.MISSING_DEPENDENCY,
                action=decision.action,
                tool_name=tool.name,
                arguments=validation.arguments,
                error=error,
            )

    try:
        output = await _dispatch_tool(tool, decision, validation.arguments, context)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        _record_tool_failed(context, tool, trace_started_at, error)
        return ToolExecutionResult(
            status=ToolExecutionStatus.ERROR,
            action=decision.action,
            tool_name=tool.name,
            arguments=validation.arguments,
            error=error,
        )

    _record_tool_finished(context, tool, trace_started_at)
    return ToolExecutionResult(
        status=ToolExecutionStatus.SUCCESS,
        action=decision.action,
        tool_name=tool.name,
        arguments=validation.arguments,
        output=output,
    )


async def _dispatch_tool(
    tool: ToolDefinition,
    decision: AgentDecision,
    arguments: dict[str, object],
    context: ToolExecutionContext,
) -> object:
    if tool.action == AgentAction.MYSQL_PLAYER_PROFILE:
        return _player_data_tools(context).get_player_profile(_string_argument(arguments, "player_id"))

    if tool.action == AgentAction.MYSQL_PLAYERS_LIST:
        return _player_data_tools(context).get_players(_int_argument(arguments, "limit", 100))

    if tool.category == ToolCategory.MAP:
        return await _run_map_tool(decision, arguments, context)

    raise ValueError(f"Unsupported tool action: {decision.action}")


async def _run_map_tool(
    decision: AgentDecision,
    arguments: dict[str, object],
    context: ToolExecutionContext,
) -> MapAgentResult:
    sanitized_decision = AgentDecision(
        action=decision.action,
        reason=decision.reason,
        arguments=arguments,
        final_task=decision.final_task,
        direct_reply=decision.direct_reply,
    )
    runner = context.map_agent_runner or run_map_agent
    return await runner(
        sanitized_decision,
        message=context.message,
        emit_status=context.emit_status,
    )


def _player_data_tools(context: ToolExecutionContext) -> PlayerDataToolsProtocol:
    factory = context.player_data_tools_factory or build_player_data_tools
    return factory()


def _string_argument(arguments: dict[str, object], name: str) -> str | None:
    value = arguments.get(name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _int_argument(arguments: dict[str, object], name: str, default: int) -> int:
    value = arguments.get(name)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return default


def _record_tool_started(
    context: ToolExecutionContext,
    tool: ToolDefinition,
    decision: AgentDecision,
) -> float | None:
    if context.agent_trace is None:
        return None
    return context.agent_trace.record_started(
        TraceEventType.TOOL_STARTED,
        tool.name,
        {
            "action": str(decision.action),
        },
    )


def _record_tool_finished(
    context: ToolExecutionContext,
    tool: ToolDefinition,
    started_at: float | None,
) -> None:
    if context.agent_trace is None:
        return
    context.agent_trace.record_finished(
        TraceEventType.TOOL_FINISHED,
        tool.name,
        started_at,
        {
            "status": str(ToolExecutionStatus.SUCCESS),
        },
    )


def _record_tool_failed(
    context: ToolExecutionContext,
    tool: ToolDefinition,
    started_at: float | None,
    error: str,
) -> None:
    if context.agent_trace is None:
        return
    context.agent_trace.record_finished(
        TraceEventType.TOOL_FAILED,
        tool.name,
        started_at,
        {
            "error": error,
        },
        error=error,
    )
