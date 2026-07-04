import json

from app.agent.audit import write_chat_audit_event
from app.agent.decision import AgentAction, AgentDecision
from app.agent.trace import AgentTrace, TraceEventType
from app.config import Settings
from app.player_data import PlayerDataStatus
from app.schemas import ChatResponse
from app.tools.executor import (
    ToolExecutionContext,
    ToolExecutionStatus,
    execute_tool_action,
)
from tests.fakes import FakePlayerDataTools


def test_trace_records_ordered_events_and_summary() -> None:
    trace = AgentTrace()

    started_at = trace.record_started(TraceEventType.NODE_STARTED, "analyze_safety")
    trace.record_finished(TraceEventType.NODE_FINISHED, "analyze_safety", started_at)

    assert [event.event_type for event in trace.events] == [
        TraceEventType.NODE_STARTED,
        TraceEventType.NODE_FINISHED,
    ]
    assert [event.name for event in trace.events] == ["analyze_safety", "analyze_safety"]
    assert trace.events[1].duration_ms is not None

    summary = trace.summary()
    assert summary["trace_event_count"] == 2
    assert summary["trace_errors"] == []
    assert summary["trace_duration_ms"] >= 0


def test_trace_records_errors_in_summary() -> None:
    trace = AgentTrace()

    trace.record_error("mysql_players_list", "database timeout")

    summary = trace.summary()
    assert summary["trace_event_count"] == 1
    assert summary["trace_errors"] == [
        {"name": "mysql_players_list", "error": "database timeout"}
    ]


async def test_tool_executor_writes_trace_events() -> None:
    player_tools = FakePlayerDataTools()
    trace = AgentTrace()
    decision = AgentDecision(
        action=AgentAction.MYSQL_PLAYER_PROFILE,
        reason="query player",
        arguments={"player_id": "1"},
    )

    result = await execute_tool_action(
        decision,
        ToolExecutionContext(
            settings=Settings(mysql_enabled=True),
            player_data_tools_factory=lambda: player_tools,
            agent_trace=trace,
        ),
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.output.status == PlayerDataStatus.FOUND
    assert [event.event_type for event in trace.events] == [
        TraceEventType.TOOL_STARTED,
        TraceEventType.TOOL_FINISHED,
    ]
    assert [event.name for event in trace.events] == [
        "mysql_player_profile",
        "mysql_player_profile",
    ]


def test_chat_audit_event_includes_trace_summary(tmp_path) -> None:
    settings = Settings(log_dir=str(tmp_path), agent_audit_log_enabled=True)
    trace = AgentTrace()
    trace.record_error("retrieve_knowledge", "vector index not ready")

    write_chat_audit_event(
        settings,
        session_id="trace-session",
        player_id="1",
        message="查询知识库",
        response=ChatResponse(reply="已回复"),
        final_state={"agent_trace": trace},
    )

    audit_file = tmp_path / "agent_audit.jsonl"
    payload = json.loads(audit_file.read_text(encoding="utf-8").splitlines()[-1])
    assert payload["trace_event_count"] == 1
    assert payload["trace_errors"] == [
        {"name": "retrieve_knowledge", "error": "vector index not ready"}
    ]
    assert payload["trace_duration_ms"] >= 0
