from app.agent.decision import AgentAction
from app.config import Settings
from app.tools import registry
from app.tools.registry import (
    ToolCategory,
    ToolDependency,
    get_tool_by_action,
    get_tool_by_name,
    is_map_tool_action,
    is_registered_tool_action,
    list_tool_definitions,
    missing_tool_dependencies,
)


def test_registry_contains_all_runtime_tool_actions() -> None:
    registered_actions = {tool.action for tool in list_tool_definitions()}

    assert AgentAction.KNOWLEDGE_BASE in registered_actions
    assert AgentAction.MYSQL_PLAYER_PROFILE in registered_actions
    assert AgentAction.MYSQL_PLAYERS_LIST in registered_actions
    assert AgentAction.AVATAR_GENERATE in registered_actions
    assert AgentAction.AMAP_PLACE_SEARCH in registered_actions
    assert AgentAction.AMAP_GEO in registered_actions
    assert AgentAction.AMAP_ROUTE in registered_actions
    assert AgentAction.AMAP_NAVIGATION in registered_actions
    assert AgentAction.AMAP_WEATHER in registered_actions
    assert AgentAction.DIRECT_ANSWER not in registered_actions


def test_lookup_tool_by_action_and_name() -> None:
    profile_tool = get_tool_by_action(AgentAction.MYSQL_PLAYER_PROFILE)

    assert profile_tool is not None
    assert profile_tool.name == "mysql_player_profile"
    assert profile_tool.category == ToolCategory.MYSQL
    assert get_tool_by_name("mysql_player_profile") == profile_tool
    assert get_tool_by_name("unknown") is None


def test_map_tool_action_classification() -> None:
    assert is_registered_tool_action(AgentAction.AMAP_WEATHER) is True
    assert is_map_tool_action(AgentAction.AMAP_WEATHER) is True
    assert is_map_tool_action(AgentAction.KNOWLEDGE_BASE) is False
    assert is_registered_tool_action(AgentAction.HANDOFF) is False


def test_missing_tool_dependencies_follow_settings() -> None:
    disabled_settings = Settings(mysql_enabled=False, amap_mcp_enabled=False)
    enabled_settings = Settings(mysql_enabled=True, amap_mcp_enabled=True)

    assert missing_tool_dependencies(disabled_settings, "mysql_players_list") == [
        ToolDependency.MYSQL
    ]
    assert missing_tool_dependencies(disabled_settings, AgentAction.AMAP_WEATHER) == [
        ToolDependency.AMAP
    ]
    assert missing_tool_dependencies(enabled_settings, "mysql_players_list") == []
    assert missing_tool_dependencies(enabled_settings, AgentAction.AMAP_WEATHER) == []
    assert missing_tool_dependencies(enabled_settings, "unknown") == []


def test_validate_tool_arguments_applies_default_limit() -> None:
    result = registry.validate_tool_arguments(AgentAction.MYSQL_PLAYERS_LIST, {})

    assert result.valid is True
    assert result.arguments == {"limit": 100}
    assert result.errors == []


def test_validate_tool_arguments_coerces_and_clamps_limit() -> None:
    result = registry.validate_tool_arguments(AgentAction.MYSQL_PLAYERS_LIST, {"limit": "1200"})

    assert result.valid is True
    assert result.arguments == {"limit": 1000}
    assert result.errors == []


def test_validate_tool_arguments_rejects_invalid_enum_value() -> None:
    result = registry.validate_tool_arguments(
        AgentAction.AMAP_ROUTE,
        {
            "origin": "广州塔",
            "destination": "白云山",
            "mode": "spaceship",
        },
    )

    assert result.valid is False
    assert result.arguments == {
        "origin": "广州塔",
        "destination": "白云山",
    }
    assert result.errors == ["mode must be one of: bicycling, driving, transit, walking"]


def test_validate_tool_arguments_ignores_unknown_arguments() -> None:
    result = registry.validate_tool_arguments(
        AgentAction.AMAP_WEATHER,
        {
            "city": "北京",
            "sql": "drop table players",
        },
    )

    assert result.valid is True
    assert result.arguments == {"city": "北京"}
    assert result.errors == []


def test_validate_tool_arguments_skips_control_actions() -> None:
    sentinel = object()
    result = registry.validate_tool_arguments(AgentAction.DIRECT_ANSWER, {"unused": sentinel})

    assert result.valid is True
    assert result.arguments == {"unused": sentinel}
    assert result.errors == []
