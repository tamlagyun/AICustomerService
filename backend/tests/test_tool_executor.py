from app.agent.decision import AgentAction, AgentDecision
from app.agent.map_agent import MapAgentResult
from app.config import Settings
from app.map_tools import MapToolStatus
from app.player_data import PlayerDataStatus
from app.tools.executor import (
    ToolExecutionContext,
    ToolExecutionStatus,
    execute_tool_action,
)
from tests.fakes import FakeMapTools, FakePlayerDataTools


async def test_executor_runs_mysql_players_list_with_default_limit() -> None:
    player_tools = FakePlayerDataTools()
    decision = AgentDecision(
        action=AgentAction.MYSQL_PLAYERS_LIST,
        reason="query all players",
        arguments={},
    )

    result = await execute_tool_action(
        decision,
        ToolExecutionContext(
            settings=Settings(mysql_enabled=True),
            player_data_tools_factory=lambda: player_tools,
        ),
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.tool_name == "mysql_players_list"
    assert result.arguments == {"limit": 100}
    assert player_tools.requested_limit == 100
    assert result.output.status == PlayerDataStatus.FOUND


async def test_executor_clamps_mysql_players_list_limit() -> None:
    player_tools = FakePlayerDataTools()
    decision = AgentDecision(
        action=AgentAction.MYSQL_PLAYERS_LIST,
        reason="query all players",
        arguments={"limit": 3000},
    )

    result = await execute_tool_action(
        decision,
        ToolExecutionContext(
            settings=Settings(mysql_enabled=True),
            player_data_tools_factory=lambda: player_tools,
        ),
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.arguments == {"limit": 1000}
    assert player_tools.requested_limit == 1000


async def test_executor_rejects_invalid_map_arguments_before_calling_runner() -> None:
    called = False

    async def map_runner(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("map runner should not be called")

    decision = AgentDecision(
        action=AgentAction.AMAP_ROUTE,
        reason="bad route mode",
        arguments={
            "origin": "广州塔",
            "destination": "白云山",
            "mode": "spaceship",
        },
    )

    result = await execute_tool_action(
        decision,
        ToolExecutionContext(
            settings=Settings(amap_mcp_enabled=True),
            map_agent_runner=map_runner,
            message="广州塔到白云山怎么去",
        ),
    )

    assert called is False
    assert result.status == ToolExecutionStatus.INVALID_ARGUMENTS
    assert result.output is None
    assert result.error == "mode must be one of: bicycling, driving, transit, walking"


async def test_executor_runs_map_action_and_returns_map_result() -> None:
    map_tools = FakeMapTools()

    async def map_runner(decision, *, message, emit_status=None):
        return await __import__("app.agent.map_agent", fromlist=["run_map_agent"]).run_map_agent(
            decision,
            message=message,
            map_tools=map_tools,
            emit_status=emit_status,
        )

    decision = AgentDecision(
        action=AgentAction.AMAP_WEATHER,
        reason="weather",
        arguments={"city": "北京"},
    )

    result = await execute_tool_action(
        decision,
        ToolExecutionContext(
            settings=Settings(amap_mcp_enabled=True),
            map_agent_runner=map_runner,
            message="北京天气",
        ),
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.tool_name == "amap_weather"
    assert result.arguments == {"city": "北京"}
    assert isinstance(result.output, MapAgentResult)
    assert result.output.map_result.status == MapToolStatus.FOUND
    assert result.output.map_result.data["tool"] == "maps_weather"


async def test_executor_reports_missing_dependency_without_running_tool() -> None:
    player_tools = FakePlayerDataTools()
    decision = AgentDecision(
        action=AgentAction.MYSQL_PLAYERS_LIST,
        reason="query all players",
        arguments={"limit": 100},
    )

    result = await execute_tool_action(
        decision,
        ToolExecutionContext(
            settings=Settings(mysql_enabled=False),
            player_data_tools_factory=lambda: player_tools,
        ),
    )

    assert result.status == ToolExecutionStatus.MISSING_DEPENDENCY
    assert result.output is None
    assert result.error == "Missing tool dependencies: mysql"
    assert player_tools.requested_limit is None


async def test_executor_reports_unsupported_tool_action() -> None:
    decision = AgentDecision(
        action=AgentAction.DIRECT_ANSWER,
        reason="not a tool",
        arguments={"unused": "value"},
    )

    result = await execute_tool_action(
        decision,
        ToolExecutionContext(settings=Settings()),
    )

    assert result.status == ToolExecutionStatus.UNSUPPORTED
    assert result.output is None
    assert result.error == "Unsupported tool action: direct_answer"
